"""A DAX tokenizer.

Enough of a lexer to answer the questions the rules ask - which functions are
called, which columns and measures are referenced, and whether a reference was
qualified with its table - without pretending to be a full DAX parser.  The
distinction matters: comments and string literals are skipped properly, so a
rule never fires on the word ``FILTER`` inside a label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

__all__ = ["TokenKind", "Token", "tokenize", "function_calls", "column_references"]


class TokenKind(Enum):
    """The token categories the rules distinguish."""

    IDENTIFIER = "identifier"
    #: A bracketed reference: ``[Sales Amount]``.
    BRACKET = "bracket"
    #: A single-quoted table name: ``'Fact Sales'``.
    QUOTED = "quoted"
    STRING = "string"
    NUMBER = "number"
    OPERATOR = "operator"
    PUNCTUATION = "punctuation"
    COMMENT = "comment"
    WHITESPACE = "whitespace"


@dataclass(frozen=True)
class Token:
    """One lexical token, with its source position."""

    kind: TokenKind
    text: str
    line: int
    column: int

    @property
    def value(self) -> str:
        """The token's content with any delimiters stripped."""
        if self.kind is TokenKind.BRACKET:
            return self.text[1:-1]
        if self.kind is TokenKind.QUOTED:
            return self.text[1:-1].replace("''", "'")
        if self.kind is TokenKind.STRING:
            return self.text[1:-1].replace('""', '"')
        return self.text

    @property
    def upper(self) -> str:
        """Upper-cased text, for case-insensitive function matching."""
        return self.text.upper()


_PATTERNS = (
    (TokenKind.COMMENT, re.compile(r"/\*.*?\*/", re.DOTALL)),
    (TokenKind.COMMENT, re.compile(r"(?://|--)[^\n]*")),
    (TokenKind.WHITESPACE, re.compile(r"\s+")),
    (TokenKind.STRING, re.compile(r'"(?:[^"]|"")*"')),
    (TokenKind.QUOTED, re.compile(r"'(?:[^']|'')*'")),
    (TokenKind.BRACKET, re.compile(r"\[[^\]]*\]")),
    (TokenKind.NUMBER, re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")),
    (TokenKind.IDENTIFIER, re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*")),
    (TokenKind.OPERATOR, re.compile(r"<>|>=|<=|&&|\|\||[-+*/^=<>&]")),
    (TokenKind.PUNCTUATION, re.compile(r"[(),;{}]")),
)


def tokenize(expression: str, keep_trivia: bool = False) -> List[Token]:
    """Tokenize a DAX expression.

    Whitespace and comments are dropped unless ``keep_trivia`` is set, which
    is what the rules want: they reason about code, not layout.
    """
    tokens: List[Token] = []
    position = 0
    line = 1
    column = 1
    length = len(expression)

    while position < length:
        for kind, pattern in _PATTERNS:
            match = pattern.match(expression, position)
            if not match:
                continue
            text = match.group(0)
            if keep_trivia or kind not in (TokenKind.WHITESPACE, TokenKind.COMMENT):
                tokens.append(Token(kind, text, line, column))
            newlines = text.count("\n")
            if newlines:
                line += newlines
                column = len(text) - text.rfind("\n")
            else:
                column += len(text)
            position = match.end()
            break
        else:
            # An unrecognised character: emit it as an operator and move on
            # rather than failing, so one odd symbol cannot block a whole lint.
            tokens.append(Token(TokenKind.OPERATOR, expression[position], line, column))
            position += 1
            column += 1
    return tokens


def function_calls(tokens: List[Token]) -> List[Token]:
    """Identifier tokens immediately followed by an opening parenthesis."""
    calls: List[Token] = []
    for index, token in enumerate(tokens):
        if token.kind is not TokenKind.IDENTIFIER:
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if following is not None and following.text == "(":
            calls.append(token)
    return calls


@dataclass(frozen=True)
class ColumnReference:
    """A ``[Name]`` reference, with the table qualifier when one was written."""

    name: str
    table: Optional[str]
    token: Token

    @property
    def qualified(self) -> bool:
        """True when the reference was written as ``Table[Name]``."""
        return self.table is not None


def column_references(tokens: List[Token]) -> List[ColumnReference]:
    """Every bracketed reference, paired with its qualifier if present.

    A qualifier is the identifier or quoted name directly before the bracket
    with no whitespace token between them - which is exactly how DAX
    distinguishes ``Sales[Amount]`` from ``Sales`` followed by ``[Amount]``.
    """
    references: List[ColumnReference] = []
    for index, token in enumerate(tokens):
        if token.kind is not TokenKind.BRACKET:
            continue
        table = None
        if index > 0:
            previous = tokens[index - 1]
            if previous.kind in (TokenKind.IDENTIFIER, TokenKind.QUOTED):
                # Guard against a function call such as SUM([x]) being read as
                # a qualifier by checking the token is adjacent in the source.
                adjacent = (
                    previous.line == token.line
                    and previous.column + len(previous.text) == token.column
                )
                if adjacent:
                    table = previous.value
        references.append(ColumnReference(token.value, table, token))
    return references
