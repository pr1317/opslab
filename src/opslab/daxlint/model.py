"""Read a Power BI tabular model.

Two formats are supported, because both are in daily use:

* **TMSL** (``.bim`` / ``model.json``) - the JSON serialisation used by
  Analysis Services and by older Power BI Desktop exports.
* **TMDL** (``*.tmdl`` files in a ``definition`` folder) - the indentation-based
  text format PBIP projects use, which is what a model under source control
  looks like now.

Only the subset the lint rules need is read.  Anything unrecognised is ignored
rather than rejected, so the linter keeps working against model files produced
by newer versions of the tooling.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

__all__ = ["Column", "Measure", "Relationship", "Table", "TabularModel", "load_model"]


@dataclass
class Column:
    """A column in a table."""

    name: str
    table: str
    data_type: str = ""
    #: ``summarizeBy``: the default aggregation Power BI applies in a visual.
    summarize_by: str = "none"
    is_hidden: bool = False
    is_key: bool = False
    format_string: str = ""
    description: str = ""
    #: Set when the column is a DAX calculated column.
    expression: str = ""
    source_column: str = ""

    @property
    def is_calculated(self) -> bool:
        """True for a DAX calculated column."""
        return bool(self.expression)

    @property
    def qualified_name(self) -> str:
        """``Table[Column]``."""
        return "%s[%s]" % (self.table, self.name)


@dataclass
class Measure:
    """A DAX measure."""

    name: str
    table: str
    expression: str = ""
    display_folder: str = ""
    description: str = ""
    format_string: str = ""
    is_hidden: bool = False

    @property
    def qualified_name(self) -> str:
        """``Table[Measure]``."""
        return "%s[%s]" % (self.table, self.name)


@dataclass
class Relationship:
    """A relationship between two tables."""

    name: str = ""
    from_table: str = ""
    from_column: str = ""
    to_table: str = ""
    to_column: str = ""
    #: ``oneDirection`` or ``bothDirections``.
    cross_filtering_behavior: str = "oneDirection"
    is_active: bool = True

    @property
    def is_bidirectional(self) -> bool:
        """True when the filter propagates both ways."""
        return self.cross_filtering_behavior.lower() == "bothdirections"

    def describe(self) -> str:
        """A readable rendering of the relationship."""
        return "%s[%s] -> %s[%s]" % (
            self.from_table, self.from_column, self.to_table, self.to_column
        )


@dataclass
class Table:
    """A table with its columns and measures."""

    name: str
    columns: List[Column] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)
    is_hidden: bool = False
    description: str = ""
    #: ``import``, ``directQuery``, ``calculated`` and so on.
    mode: str = ""
    #: True when the table is flagged as the model's date table.
    is_date_table: bool = False
    #: True for Power BI's automatically generated date tables.
    is_auto_date_table: bool = False

    def column(self, name: str) -> Optional[Column]:
        """Look up a column by name, case-insensitively."""
        lowered = name.lower()
        for candidate in self.columns:
            if candidate.name.lower() == lowered:
                return candidate
        return None


@dataclass
class TabularModel:
    """The parts of a tabular model the linter reasons about."""

    name: str = "model"
    tables: List[Table] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    annotations: Dict[str, str] = field(default_factory=dict)
    source_path: str = ""

    # -- lookups -----------------------------------------------------------
    def measures(self) -> List[Measure]:
        """Every measure in the model."""
        return [m for t in self.tables for m in t.measures]

    def columns(self) -> List[Column]:
        """Every column in the model."""
        return [c for t in self.tables for c in t.columns]

    def table(self, name: str) -> Optional[Table]:
        """Look up a table by name, case-insensitively."""
        lowered = name.lower()
        for candidate in self.tables:
            if candidate.name.lower() == lowered:
                return candidate
        return None

    def measure_names(self) -> Dict[str, Measure]:
        """Measures keyed by lower-cased name."""
        return {m.name.lower(): m for m in self.measures()}

    def column_names(self) -> Dict[str, List[Column]]:
        """Columns grouped by lower-cased name, since names repeat across tables."""
        grouped: Dict[str, List[Column]] = {}
        for column in self.columns():
            grouped.setdefault(column.name.lower(), []).append(column)
        return grouped

    @property
    def auto_date_tables(self) -> List[Table]:
        """Power BI's hidden per-column date tables, if auto date/time is on."""
        return [t for t in self.tables if t.is_auto_date_table]


# --------------------------------------------------------------------------
# TMSL (.bim / model.json)
# --------------------------------------------------------------------------
_AUTO_DATE_PATTERN = re.compile(r"^(LocalDateTable_|DateTableTemplate_)", re.IGNORECASE)


def _annotations(node: dict) -> Dict[str, str]:
    return {
        str(a.get("name", "")): str(a.get("value", ""))
        for a in node.get("annotations", [])
        if isinstance(a, dict)
    }


def _expression_text(value) -> str:
    """TMSL writes multi-line expressions as a list of lines."""
    if isinstance(value, list):
        return "\n".join(str(part) for part in value)
    return "" if value is None else str(value)


def load_tmsl(path: str) -> TabularModel:
    """Load a model from a TMSL ``.bim`` or ``model.json`` file."""
    with open(path, encoding="utf-8-sig") as handle:
        document = json.load(handle)

    root = document.get("model", document)
    model = TabularModel(
        name=str(document.get("name", os.path.basename(path))),
        annotations=_annotations(root),
        source_path=path,
    )

    for table_node in root.get("tables", []):
        table_name = str(table_node.get("name", ""))
        table_annotations = _annotations(table_node)
        partitions = table_node.get("partitions", []) or []
        mode = ""
        if partitions and isinstance(partitions[0], dict):
            mode = str(partitions[0].get("mode", ""))

        table = Table(
            name=table_name,
            is_hidden=bool(table_node.get("isHidden", False)),
            description=_expression_text(table_node.get("description")),
            mode=mode,
            is_date_table=str(table_node.get("dataCategory", "")).lower() == "time",
            is_auto_date_table=bool(_AUTO_DATE_PATTERN.match(table_name))
            or table_annotations.get("__PBI_LocalDateTable", "").lower() == "true",
        )

        for column_node in table_node.get("columns", []):
            table.columns.append(
                Column(
                    name=str(column_node.get("name", "")),
                    table=table_name,
                    data_type=str(column_node.get("dataType", "")),
                    summarize_by=str(column_node.get("summarizeBy", "none")),
                    is_hidden=bool(column_node.get("isHidden", False)),
                    is_key=bool(column_node.get("isKey", False)),
                    format_string=str(column_node.get("formatString", "")),
                    description=_expression_text(column_node.get("description")),
                    expression=_expression_text(column_node.get("expression")),
                    source_column=str(column_node.get("sourceColumn", "")),
                )
            )

        for measure_node in table_node.get("measures", []):
            table.measures.append(
                Measure(
                    name=str(measure_node.get("name", "")),
                    table=table_name,
                    expression=_expression_text(measure_node.get("expression")),
                    display_folder=str(measure_node.get("displayFolder", "")),
                    description=_expression_text(measure_node.get("description")),
                    format_string=str(measure_node.get("formatString", "")),
                    is_hidden=bool(measure_node.get("isHidden", False)),
                )
            )
        model.tables.append(table)

    for relationship_node in root.get("relationships", []):
        model.relationships.append(
            Relationship(
                name=str(relationship_node.get("name", "")),
                from_table=str(relationship_node.get("fromTable", "")),
                from_column=str(relationship_node.get("fromColumn", "")),
                to_table=str(relationship_node.get("toTable", "")),
                to_column=str(relationship_node.get("toColumn", "")),
                cross_filtering_behavior=str(
                    relationship_node.get("crossFilteringBehavior", "oneDirection")
                ),
                is_active=bool(relationship_node.get("isActive", True)),
            )
        )
    return model


# --------------------------------------------------------------------------
# TMDL (PBIP projects)
# --------------------------------------------------------------------------
@dataclass
class _TmdlNode:
    """One block of a TMDL file: a header line plus its indented children."""

    header: str
    indent: int
    lines: List[str] = field(default_factory=list)
    children: List["_TmdlNode"] = field(default_factory=list)

    def property_value(self, name: str) -> str:
        """Read a ``name: value`` property from this block's own lines."""
        prefix = name.lower() + ":"
        for line in self.lines:
            stripped = line.strip()
            if stripped.lower().startswith(prefix):
                return stripped[len(prefix):].strip()
        return ""

    def has_flag(self, name: str) -> bool:
        """True when a bare flag line such as ``isHidden`` is present."""
        lowered = name.lower()
        for line in self.lines:
            stripped = line.strip().lower()
            if stripped == lowered or stripped == lowered + ": true":
                return True
        return False


def _parse_tmdl_blocks(text: str) -> List[_TmdlNode]:
    """Build a shallow block tree from TMDL's significant indentation."""
    lines = text.splitlines()
    roots: List[_TmdlNode] = []
    stack: List[_TmdlNode] = []

    for raw in lines:
        if not raw.strip() or raw.strip().startswith("///"):
            continue
        indent = len(raw) - len(raw.lstrip("\t "))
        stripped = raw.strip()
        is_header = bool(
            re.match(
                r"^(table|column|measure|hierarchy|partition|relationship|"
                r"annotation|model|expression|role|perspective|culture)\b",
                stripped,
                re.IGNORECASE,
            )
        )
        if is_header:
            node = _TmdlNode(header=stripped, indent=indent)
            while stack and stack[-1].indent >= indent:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
        elif stack:
            stack[-1].lines.append(raw)
    return roots


def _tmdl_name_and_expression(header: str, node: _TmdlNode) -> "tuple":
    """Split ``measure 'X' = expr`` into its name and expression."""
    body = header.split(None, 1)[1] if " " in header else ""
    expression = ""
    if "=" in body:
        name_part, expression = body.split("=", 1)
    else:
        name_part = body
    name = name_part.strip().strip("'").replace("''", "'")
    expression = expression.strip()
    # A multi-line expression continues in the block's indented lines, until
    # the first line that looks like a property assignment.
    continuation: List[str] = []
    for line in node.lines:
        stripped = line.strip()
        if re.match(r"^[A-Za-z_]+:\s", stripped) or stripped in {
            "isHidden", "isKey", "isActive",
        }:
            break
        continuation.append(stripped)
    if continuation:
        expression = (expression + "\n" + "\n".join(continuation)).strip()
    return name, expression


def load_tmdl(path: str) -> TabularModel:
    """Load a model from a TMDL file or a PBIP ``definition`` folder."""
    files: List[str] = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            files.extend(
                os.path.join(root, n) for n in sorted(names) if n.lower().endswith(".tmdl")
            )
    else:
        files = [path]
    if not files:
        raise ValueError("no .tmdl files found under %r" % path)

    model = TabularModel(name=os.path.basename(path.rstrip(os.sep)), source_path=path)

    for file_path in files:
        with open(file_path, encoding="utf-8-sig") as handle:
            blocks = _parse_tmdl_blocks(handle.read())
        for block in blocks:
            keyword = block.header.split(None, 1)[0].lower()
            if keyword == "table":
                model.tables.append(_tmdl_table(block))
            elif keyword == "relationship":
                model.relationships.append(_tmdl_relationship(block))
    return model


def _tmdl_table(block: _TmdlNode) -> Table:
    name = block.header.split(None, 1)[1].strip().strip("'") if " " in block.header else ""
    table = Table(
        name=name,
        is_hidden=block.has_flag("isHidden"),
        description=block.property_value("description"),
        is_date_table=block.property_value("dataCategory").lower() == "time",
        is_auto_date_table=bool(_AUTO_DATE_PATTERN.match(name)),
    )
    for child in block.children:
        keyword = child.header.split(None, 1)[0].lower()
        if keyword == "column":
            column_name, expression = _tmdl_name_and_expression(child.header, child)
            table.columns.append(
                Column(
                    name=column_name,
                    table=name,
                    data_type=child.property_value("dataType"),
                    summarize_by=child.property_value("summarizeBy") or "none",
                    is_hidden=child.has_flag("isHidden"),
                    is_key=child.has_flag("isKey"),
                    format_string=child.property_value("formatString"),
                    description=child.property_value("description"),
                    expression=expression,
                    source_column=child.property_value("sourceColumn"),
                )
            )
        elif keyword == "measure":
            measure_name, expression = _tmdl_name_and_expression(child.header, child)
            table.measures.append(
                Measure(
                    name=measure_name,
                    table=name,
                    expression=expression,
                    display_folder=child.property_value("displayFolder"),
                    description=child.property_value("description"),
                    format_string=child.property_value("formatString"),
                    is_hidden=child.has_flag("isHidden"),
                )
            )
        elif keyword == "partition":
            table.mode = child.property_value("mode") or table.mode
    return table


def _tmdl_relationship(block: _TmdlNode) -> Relationship:
    name = block.header.split(None, 1)[1].strip() if " " in block.header else ""

    def split_reference(reference: str) -> "tuple":
        if "." not in reference:
            return reference.strip().strip("'"), ""
        table_part, column_part = reference.split(".", 1)
        return table_part.strip().strip("'"), column_part.strip().strip("'")

    from_table, from_column = split_reference(block.property_value("fromColumn"))
    to_table, to_column = split_reference(block.property_value("toColumn"))
    behavior = block.property_value("crossFilteringBehavior") or "oneDirection"
    is_active = block.property_value("isActive").lower() != "false"
    return Relationship(
        name=name,
        from_table=from_table,
        from_column=from_column,
        to_table=to_table,
        to_column=to_column,
        cross_filtering_behavior=behavior,
        is_active=is_active,
    )


def load_model(path: str) -> TabularModel:
    """Load a tabular model, choosing the reader from the path."""
    if os.path.isdir(path):
        return load_tmdl(path)
    lowered = path.lower()
    if lowered.endswith((".bim", ".json")):
        return load_tmsl(path)
    if lowered.endswith(".tmdl"):
        return load_tmdl(path)
    raise ValueError(
        "unsupported model file %r: expected a .bim/.json (TMSL) file, a .tmdl "
        "file, or a PBIP definition folder" % path
    )
