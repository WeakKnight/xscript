"""
XScript Token Definitions

Defines all token types and the Token class for lexical analysis.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional


class TokenType(Enum):
    """All token types in XScript."""
    
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    
    # Keywords
    VAR = auto()
    FUNC = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    DO = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    NIL = auto()
    TRUE = auto()
    FALSE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Operators
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()          # *
    SLASH = auto()         # /
    PERCENT = auto()       # %
    CARET = auto()         # ^
    HASH = auto()          # #
    
    # Comparison
    EQ = auto()            # ==
    NE = auto()            # !=
    LT = auto()            # <
    LE = auto()            # <=
    GT = auto()            # >
    GE = auto()            # >=
    
    # Assignment
    ASSIGN = auto()        # =
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN = auto()  # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN = auto()  # /=
    PERCENT_ASSIGN = auto() # %=
    
    # Delimiters
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    COMMA = auto()         # ,
    DOT = auto()           # .
    COLON = auto()         # :
    SEMICOLON = auto()     # ;
    QUESTION = auto()      # ?
    
    # Special
    EOF = auto()
    NEWLINE = auto()
    ERROR = auto()


# Keyword mapping
KEYWORDS = {
    'var': TokenType.VAR,
    'func': TokenType.FUNC,
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'for': TokenType.FOR,
    'while': TokenType.WHILE,
    'do': TokenType.DO,
    'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE,
    'return': TokenType.RETURN,
    'nil': TokenType.NIL,
    'true': TokenType.TRUE,
    'false': TokenType.FALSE,
    'and': TokenType.AND,
    'or': TokenType.OR,
    'not': TokenType.NOT,
}


@dataclass
class Token:
    """Represents a single token from the source code."""
    
    type: TokenType
    lexeme: str
    value: Any
    line: int
    column: int
    
    def __repr__(self) -> str:
        if self.value is not None:
            return f"Token({self.type.name}, {self.lexeme!r}, {self.value!r}, line={self.line})"
        return f"Token({self.type.name}, {self.lexeme!r}, line={self.line})"
    
    def is_keyword(self) -> bool:
        """Check if this token is a keyword."""
        return self.type in KEYWORDS.values()
    
    def is_literal(self) -> bool:
        """Check if this token is a literal value."""
        return self.type in (TokenType.NUMBER, TokenType.STRING, 
                            TokenType.TRUE, TokenType.FALSE, TokenType.NIL)
    
    def is_operator(self) -> bool:
        """Check if this token is an operator."""
        return self.type in (
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.PERCENT, TokenType.CARET, TokenType.EQ, TokenType.NE,
            TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE,
            TokenType.AND, TokenType.OR, TokenType.NOT
        )
    
    def is_assignment(self) -> bool:
        """Check if this token is an assignment operator."""
        return self.type in (
            TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
            TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN, TokenType.PERCENT_ASSIGN
        )


# Operator precedence (higher = binds tighter)
PRECEDENCE = {
    TokenType.OR: 1,
    TokenType.AND: 2,
    TokenType.EQ: 3,
    TokenType.NE: 3,
    TokenType.LT: 4,
    TokenType.LE: 4,
    TokenType.GT: 4,
    TokenType.GE: 4,
    TokenType.PLUS: 5,
    TokenType.MINUS: 5,
    TokenType.STAR: 6,
    TokenType.SLASH: 6,
    TokenType.PERCENT: 6,
    TokenType.CARET: 7,  # Right-associative
}


def get_precedence(token_type: TokenType) -> int:
    """Get the precedence of an operator token type."""
    return PRECEDENCE.get(token_type, 0)

