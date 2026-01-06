"""
XScript Lexer

Tokenizes XScript source code into a stream of tokens.
"""

from typing import List, Optional
from .tokens import Token, TokenType, KEYWORDS
from .errors import SyntaxError


class Lexer:
    """Lexical analyzer for XScript source code."""
    
    def __init__(self, source: str):
        """
        Initialize the lexer.
        
        Args:
            source: XScript source code to tokenize
        """
        self.source = source
        self.tokens: List[Token] = []
        self.start = 0      # Start of current token
        self.current = 0    # Current position
        self.line = 1       # Current line number
        self.column = 1     # Current column number
        self.line_start = 0 # Position of current line start
    
    def tokenize(self) -> List[Token]:
        """
        Tokenize the entire source code.
        
        Returns:
            List of tokens
        """
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        
        self.tokens.append(Token(TokenType.EOF, "", None, self.line, self.column))
        return self.tokens
    
    def scan_token(self) -> None:
        """Scan the next token."""
        c = self.advance()
        
        # Skip whitespace
        if c in ' \t\r':
            return
        
        # Newline
        if c == '\n':
            self.line += 1
            self.column = 1
            self.line_start = self.current
            return
        
        # Comments
        if c == '/':
            if self.match('/'):
                # Line comment
                while self.peek() != '\n' and not self.is_at_end():
                    self.advance()
                return
            elif self.match('*'):
                # Block comment
                self.block_comment()
                return
            elif self.match('='):
                self.add_token(TokenType.SLASH_ASSIGN)
                return
            else:
                self.add_token(TokenType.SLASH)
                return
        
        # Single-character tokens
        if c == '(':
            self.add_token(TokenType.LPAREN)
        elif c == ')':
            self.add_token(TokenType.RPAREN)
        elif c == '{':
            self.add_token(TokenType.LBRACE)
        elif c == '}':
            self.add_token(TokenType.RBRACE)
        elif c == '[':
            self.add_token(TokenType.LBRACKET)
        elif c == ']':
            self.add_token(TokenType.RBRACKET)
        elif c == ',':
            self.add_token(TokenType.COMMA)
        elif c == '.':
            self.add_token(TokenType.DOT)
        elif c == ':':
            self.add_token(TokenType.COLON)
        elif c == ';':
            self.add_token(TokenType.SEMICOLON)
        elif c == '?':
            self.add_token(TokenType.QUESTION)
        elif c == '#':
            self.add_token(TokenType.HASH)
        elif c == '^':
            self.add_token(TokenType.CARET)
        
        # Operators with possible assignment
        elif c == '+':
            self.add_token(TokenType.PLUS_ASSIGN if self.match('=') else TokenType.PLUS)
        elif c == '-':
            self.add_token(TokenType.MINUS_ASSIGN if self.match('=') else TokenType.MINUS)
        elif c == '*':
            self.add_token(TokenType.STAR_ASSIGN if self.match('=') else TokenType.STAR)
        elif c == '%':
            self.add_token(TokenType.PERCENT_ASSIGN if self.match('=') else TokenType.PERCENT)
        
        # Comparison operators
        elif c == '=':
            self.add_token(TokenType.EQ if self.match('=') else TokenType.ASSIGN)
        elif c == '!':
            self.add_token(TokenType.NE if self.match('=') else TokenType.NOT)
        elif c == '<':
            self.add_token(TokenType.LE if self.match('=') else TokenType.LT)
        elif c == '>':
            self.add_token(TokenType.GE if self.match('=') else TokenType.GT)
        
        # String literals
        elif c == '"':
            self.string('"')
        elif c == "'":
            self.string("'")
        elif c == '`':
            self.template_string()
        
        # Numbers
        elif c.isdigit():
            self.number()
        
        # Identifiers and keywords
        elif c.isalpha() or c == '_':
            self.identifier()
        
        else:
            raise SyntaxError(f"Unexpected character: {c!r}", self.line, self.column)
    
    def advance(self) -> str:
        """Consume and return the current character."""
        c = self.source[self.current]
        self.current += 1
        self.column += 1
        return c
    
    def peek(self) -> str:
        """Return the current character without consuming it."""
        if self.is_at_end():
            return '\0'
        return self.source[self.current]
    
    def peek_next(self) -> str:
        """Return the next character without consuming it."""
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]
    
    def match(self, expected: str) -> bool:
        """Consume the current character if it matches expected."""
        if self.is_at_end():
            return False
        if self.source[self.current] != expected:
            return False
        self.current += 1
        self.column += 1
        return True
    
    def is_at_end(self) -> bool:
        """Check if we've reached the end of the source."""
        return self.current >= len(self.source)
    
    def add_token(self, type: TokenType, value: any = None) -> None:
        """Add a token to the token list."""
        lexeme = self.source[self.start:self.current]
        col = self.start - self.line_start + 1
        self.tokens.append(Token(type, lexeme, value, self.line, col))
    
    def string(self, quote: str) -> None:
        """Scan a string literal."""
        value = []
        
        while self.peek() != quote and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.column = 1
                self.line_start = self.current + 1
            
            if self.peek() == '\\':
                self.advance()
                c = self.advance()
                if c == 'n':
                    value.append('\n')
                elif c == 't':
                    value.append('\t')
                elif c == 'r':
                    value.append('\r')
                elif c == '\\':
                    value.append('\\')
                elif c == quote:
                    value.append(quote)
                elif c == '0':
                    value.append('\0')
                else:
                    value.append(c)
            else:
                value.append(self.advance())
        
        if self.is_at_end():
            raise SyntaxError("Unterminated string", self.line, self.column)
        
        # Consume closing quote
        self.advance()
        
        self.add_token(TokenType.STRING, ''.join(value))
    
    def template_string(self) -> None:
        """Scan a template string literal (backtick string)."""
        # For simplicity, treat as regular string for now
        # Full implementation would handle ${...} interpolation
        value = []
        
        while self.peek() != '`' and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.column = 1
                self.line_start = self.current + 1
            value.append(self.advance())
        
        if self.is_at_end():
            raise SyntaxError("Unterminated template string", self.line, self.column)
        
        # Consume closing backtick
        self.advance()
        
        self.add_token(TokenType.STRING, ''.join(value))
    
    def number(self) -> None:
        """Scan a number literal."""
        # Check for hex or binary
        if self.source[self.start] == '0':
            if self.match('x') or self.match('X'):
                self.hex_number()
                return
            elif self.match('b') or self.match('B'):
                self.binary_number()
                return
        
        # Decimal number
        while self.peek().isdigit():
            self.advance()
        
        # Fractional part
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()  # Consume '.'
            while self.peek().isdigit():
                self.advance()
        
        # Exponent part
        if self.peek() in 'eE':
            self.advance()
            if self.peek() in '+-':
                self.advance()
            while self.peek().isdigit():
                self.advance()
        
        value = float(self.source[self.start:self.current])
        self.add_token(TokenType.NUMBER, value)
    
    def hex_number(self) -> None:
        """Scan a hexadecimal number."""
        while self.peek() in '0123456789abcdefABCDEF':
            self.advance()
        
        hex_str = self.source[self.start + 2:self.current]
        value = float(int(hex_str, 16))
        self.add_token(TokenType.NUMBER, value)
    
    def binary_number(self) -> None:
        """Scan a binary number."""
        while self.peek() in '01':
            self.advance()
        
        bin_str = self.source[self.start + 2:self.current]
        value = float(int(bin_str, 2))
        self.add_token(TokenType.NUMBER, value)
    
    def identifier(self) -> None:
        """Scan an identifier or keyword."""
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()
        
        text = self.source[self.start:self.current]
        
        # Check if it's a keyword
        token_type = KEYWORDS.get(text, TokenType.IDENTIFIER)
        
        if token_type == TokenType.TRUE:
            self.add_token(token_type, True)
        elif token_type == TokenType.FALSE:
            self.add_token(token_type, False)
        elif token_type == TokenType.NIL:
            self.add_token(token_type, None)
        else:
            self.add_token(token_type)
    
    def block_comment(self) -> None:
        """Skip a block comment /* ... */."""
        depth = 1
        
        while depth > 0 and not self.is_at_end():
            if self.peek() == '/' and self.peek_next() == '*':
                self.advance()
                self.advance()
                depth += 1
            elif self.peek() == '*' and self.peek_next() == '/':
                self.advance()
                self.advance()
                depth -= 1
            else:
                if self.peek() == '\n':
                    self.line += 1
                    self.column = 1
                    self.line_start = self.current + 1
                self.advance()
        
        if depth > 0:
            raise SyntaxError("Unterminated block comment", self.line, self.column)

