"""
XScript Parser

Recursive descent parser that produces an AST from tokens.
"""

from typing import List, Optional, Callable
from .tokens import Token, TokenType, get_precedence
from .ast import *
from .errors import SyntaxError


class Parser:
    """Recursive descent parser for XScript."""
    
    def __init__(self, tokens: List[Token]):
        """
        Initialize the parser.
        
        Args:
            tokens: List of tokens from the lexer
        """
        self.tokens = tokens
        self.current = 0
    
    def parse(self) -> Program:
        """
        Parse the token stream into an AST.
        
        Returns:
            Program AST node
        """
        statements = []
        
        while not self.is_at_end():
            stmt = self.declaration()
            if stmt:
                statements.append(stmt)
        
        return Program(statements)
    
    # =========================================================================
    # Declarations
    # =========================================================================
    
    def declaration(self) -> Optional[Statement]:
        """Parse a declaration or statement."""
        try:
            if self.match(TokenType.VAR):
                return self.var_declaration()
            if self.match(TokenType.FUNC):
                return self.function_declaration()
            return self.statement()
        except SyntaxError as e:
            self.synchronize()
            raise
    
    def var_declaration(self) -> VarDeclStmt:
        """Parse a variable declaration."""
        name = self.consume(TokenType.IDENTIFIER, "Expected variable name")
        
        initializer = None
        if self.match(TokenType.ASSIGN):
            initializer = self.expression()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after variable declaration")
        return VarDeclStmt(name, initializer)
    
    def function_declaration(self) -> FunctionDeclStmt:
        """Parse a function declaration."""
        name = self.consume(TokenType.IDENTIFIER, "Expected function name")
        
        self.consume(TokenType.LPAREN, "Expected '(' after function name")
        params = self.parameters()
        self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        
        self.consume(TokenType.LBRACE, "Expected '{' before function body")
        body = self.block()
        
        return FunctionDeclStmt(name, params, body)
    
    def parameters(self) -> List[Token]:
        """Parse function parameters."""
        params = []
        
        if not self.check(TokenType.RPAREN):
            params.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name"))
            
            while self.match(TokenType.COMMA):
                params.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name"))
        
        return params
    
    # =========================================================================
    # Statements
    # =========================================================================
    
    def statement(self) -> Statement:
        """Parse a statement."""
        if self.match(TokenType.IF):
            return self.if_statement()
        if self.match(TokenType.WHILE):
            return self.while_statement()
        if self.match(TokenType.DO):
            return self.do_while_statement()
        if self.match(TokenType.FOR):
            return self.for_statement()
        if self.match(TokenType.BREAK):
            return self.break_statement()
        if self.match(TokenType.CONTINUE):
            return self.continue_statement()
        if self.match(TokenType.RETURN):
            return self.return_statement()
        if self.match(TokenType.LBRACE):
            return self.block()
        
        return self.expression_statement()
    
    def if_statement(self) -> IfStmt:
        """Parse an if statement."""
        self.consume(TokenType.LPAREN, "Expected '(' after 'if'")
        condition = self.expression()
        self.consume(TokenType.RPAREN, "Expected ')' after if condition")
        
        then_branch = self.statement()
        
        else_branch = None
        if self.match(TokenType.ELSE):
            else_branch = self.statement()
        
        return IfStmt(condition, then_branch, else_branch)
    
    def while_statement(self) -> WhileStmt:
        """Parse a while statement."""
        self.consume(TokenType.LPAREN, "Expected '(' after 'while'")
        condition = self.expression()
        self.consume(TokenType.RPAREN, "Expected ')' after while condition")
        
        body = self.statement()
        
        return WhileStmt(condition, body)
    
    def do_while_statement(self) -> DoWhileStmt:
        """Parse a do-while statement."""
        body = self.statement()
        
        self.consume(TokenType.WHILE, "Expected 'while' after do block")
        self.consume(TokenType.LPAREN, "Expected '(' after 'while'")
        condition = self.expression()
        self.consume(TokenType.RPAREN, "Expected ')' after condition")
        self.consume(TokenType.SEMICOLON, "Expected ';' after do-while")
        
        return DoWhileStmt(body, condition)
    
    def for_statement(self) -> ForStmt:
        """Parse a for statement."""
        self.consume(TokenType.LPAREN, "Expected '(' after 'for'")
        
        # Initializer
        initializer = None
        if self.match(TokenType.SEMICOLON):
            pass
        elif self.match(TokenType.VAR):
            initializer = self.var_declaration()
        else:
            initializer = self.expression_statement()
        
        # Condition
        condition = None
        if not self.check(TokenType.SEMICOLON):
            condition = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after for condition")
        
        # Increment
        increment = None
        if not self.check(TokenType.RPAREN):
            increment = self.expression()
        self.consume(TokenType.RPAREN, "Expected ')' after for clauses")
        
        body = self.statement()
        
        return ForStmt(initializer, condition, increment, body)
    
    def break_statement(self) -> BreakStmt:
        """Parse a break statement."""
        keyword = self.previous()
        self.consume(TokenType.SEMICOLON, "Expected ';' after 'break'")
        return BreakStmt(keyword)
    
    def continue_statement(self) -> ContinueStmt:
        """Parse a continue statement."""
        keyword = self.previous()
        self.consume(TokenType.SEMICOLON, "Expected ';' after 'continue'")
        return ContinueStmt(keyword)
    
    def return_statement(self) -> ReturnStmt:
        """Parse a return statement."""
        keyword = self.previous()
        
        values = []
        if not self.check(TokenType.SEMICOLON):
            values.append(self.expression())
            while self.match(TokenType.COMMA):
                values.append(self.expression())
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after return value")
        return ReturnStmt(keyword, values)
    
    def block(self) -> BlockStmt:
        """Parse a block of statements."""
        statements = []
        
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            stmt = self.declaration()
            if stmt:
                statements.append(stmt)
        
        self.consume(TokenType.RBRACE, "Expected '}' after block")
        return BlockStmt(statements)
    
    def expression_statement(self) -> ExpressionStmt:
        """Parse an expression statement."""
        expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
        return ExpressionStmt(expr)
    
    # =========================================================================
    # Expressions
    # =========================================================================
    
    def expression(self) -> Expression:
        """Parse an expression."""
        return self.assignment()
    
    def assignment(self) -> Expression:
        """Parse an assignment expression."""
        expr = self.ternary()
        
        if self.match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                     TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN, TokenType.PERCENT_ASSIGN):
            operator = self.previous()
            value = self.assignment()
            
            # Check if target is valid
            if isinstance(expr, (IdentifierExpr, IndexExpr, DotExpr)):
                return AssignExpr(expr, operator, value)
            
            raise SyntaxError("Invalid assignment target", operator.line, operator.column)
        
        return expr
    
    def ternary(self) -> Expression:
        """Parse a ternary conditional expression."""
        expr = self.or_expr()
        
        if self.match(TokenType.QUESTION):
            then_expr = self.expression()
            self.consume(TokenType.COLON, "Expected ':' in ternary expression")
            else_expr = self.ternary()
            return TernaryExpr(expr, then_expr, else_expr)
        
        return expr
    
    def or_expr(self) -> Expression:
        """Parse a logical OR expression."""
        expr = self.and_expr()
        
        while self.match(TokenType.OR):
            operator = self.previous()
            right = self.and_expr()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def and_expr(self) -> Expression:
        """Parse a logical AND expression."""
        expr = self.equality()
        
        while self.match(TokenType.AND):
            operator = self.previous()
            right = self.equality()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def equality(self) -> Expression:
        """Parse an equality expression."""
        expr = self.comparison()
        
        while self.match(TokenType.EQ, TokenType.NE):
            operator = self.previous()
            right = self.comparison()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def comparison(self) -> Expression:
        """Parse a comparison expression."""
        expr = self.term()
        
        while self.match(TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE):
            operator = self.previous()
            right = self.term()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def term(self) -> Expression:
        """Parse addition/subtraction."""
        expr = self.factor()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.previous()
            right = self.factor()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def factor(self) -> Expression:
        """Parse multiplication/division/modulo."""
        expr = self.power()
        
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            operator = self.previous()
            right = self.power()
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def power(self) -> Expression:
        """Parse exponentiation (right-associative)."""
        expr = self.unary()
        
        if self.match(TokenType.CARET):
            operator = self.previous()
            right = self.power()  # Right-associative
            expr = BinaryExpr(expr, operator, right)
        
        return expr
    
    def unary(self) -> Expression:
        """Parse unary expressions."""
        if self.match(TokenType.MINUS, TokenType.NOT, TokenType.HASH):
            operator = self.previous()
            operand = self.unary()
            return UnaryExpr(operator, operand)
        
        return self.call()
    
    def call(self) -> Expression:
        """Parse function calls and member access."""
        expr = self.primary()
        
        while True:
            if self.match(TokenType.LPAREN):
                expr = self.finish_call(expr)
            elif self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expected property name after '.'")
                expr = DotExpr(expr, name)
            elif self.match(TokenType.LBRACKET):
                bracket = self.previous()
                index = self.expression()
                self.consume(TokenType.RBRACKET, "Expected ']' after index")
                expr = IndexExpr(expr, index, bracket)
            else:
                break
        
        return expr
    
    def finish_call(self, callee: Expression) -> CallExpr:
        """Parse function call arguments."""
        paren = self.previous()
        arguments = []
        
        if not self.check(TokenType.RPAREN):
            arguments.append(self.expression())
            while self.match(TokenType.COMMA):
                arguments.append(self.expression())
        
        self.consume(TokenType.RPAREN, "Expected ')' after arguments")
        return CallExpr(callee, arguments, paren)
    
    def primary(self) -> Expression:
        """Parse primary expressions."""
        # Literals
        if self.match(TokenType.NIL):
            return LiteralExpr(None, self.previous())
        if self.match(TokenType.TRUE):
            return LiteralExpr(True, self.previous())
        if self.match(TokenType.FALSE):
            return LiteralExpr(False, self.previous())
        if self.match(TokenType.NUMBER, TokenType.STRING):
            return LiteralExpr(self.previous().value, self.previous())
        
        # Identifier
        if self.match(TokenType.IDENTIFIER):
            return IdentifierExpr(self.previous().lexeme, self.previous())
        
        # Grouped expression
        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expected ')' after expression")
            return GroupExpr(expr)
        
        # Table literal
        if self.match(TokenType.LBRACE):
            return self.table_literal()
        
        # Anonymous function
        if self.match(TokenType.FUNC):
            return self.function_expression()
        
        raise SyntaxError(
            f"Expected expression, got {self.peek().type.name}",
            self.peek().line,
            self.peek().column
        )
    
    def table_literal(self) -> TableExpr:
        """Parse a table literal."""
        brace = self.previous()
        entries = []
        array_index = 0
        
        if not self.check(TokenType.RBRACE):
            entries.append(self.table_entry(array_index))
            array_index += 1
            
            while self.match(TokenType.COMMA, TokenType.SEMICOLON):
                if self.check(TokenType.RBRACE):
                    break
                entries.append(self.table_entry(array_index))
                array_index += 1
        
        self.consume(TokenType.RBRACE, "Expected '}' after table entries")
        return TableExpr(entries, brace)
    
    def table_entry(self, array_index: int) -> TableEntry:
        """Parse a single table entry."""
        # [expr]: value
        if self.match(TokenType.LBRACKET):
            key = self.expression()
            self.consume(TokenType.RBRACKET, "Expected ']' after key")
            self.consume(TokenType.COLON, "Expected ':' after key")
            value = self.expression()
            return TableEntry(key, value)
        
        # Check for identifier: value
        if self.check(TokenType.IDENTIFIER) and self.check_next(TokenType.COLON):
            name_token = self.advance()
            self.advance()  # consume ':'
            key = LiteralExpr(name_token.lexeme, name_token)
            value = self.expression()
            return TableEntry(key, value)
        
        # Array-style entry (no key)
        value = self.expression()
        return TableEntry(None, value)
    
    def function_expression(self) -> FunctionExpr:
        """Parse an anonymous function expression."""
        self.consume(TokenType.LPAREN, "Expected '(' after 'func'")
        params = self.parameters()
        self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        
        self.consume(TokenType.LBRACE, "Expected '{' before function body")
        body = self.block()
        
        return FunctionExpr(params, body)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types and advance."""
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False
    
    def check(self, type: TokenType) -> bool:
        """Check if current token is of given type."""
        if self.is_at_end():
            return False
        return self.peek().type == type
    
    def check_next(self, type: TokenType) -> bool:
        """Check if next token is of given type."""
        if self.current + 1 >= len(self.tokens):
            return False
        return self.tokens[self.current + 1].type == type
    
    def advance(self) -> Token:
        """Consume and return the current token."""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def is_at_end(self) -> bool:
        """Check if we've reached the end of tokens."""
        return self.peek().type == TokenType.EOF
    
    def peek(self) -> Token:
        """Return the current token."""
        return self.tokens[self.current]
    
    def previous(self) -> Token:
        """Return the previous token."""
        return self.tokens[self.current - 1]
    
    def consume(self, type: TokenType, message: str) -> Token:
        """Consume a token of the expected type or raise an error."""
        if self.check(type):
            return self.advance()
        
        token = self.peek()
        raise SyntaxError(message, token.line, token.column)
    
    def synchronize(self) -> None:
        """Synchronize after an error to continue parsing."""
        self.advance()
        
        while not self.is_at_end():
            if self.previous().type == TokenType.SEMICOLON:
                return
            
            if self.peek().type in (
                TokenType.VAR, TokenType.FUNC, TokenType.IF, TokenType.WHILE,
                TokenType.FOR, TokenType.RETURN
            ):
                return
            
            self.advance()

