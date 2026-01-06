"""
XScript Abstract Syntax Tree

Defines AST node classes for the XScript language.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
from .tokens import Token


# =============================================================================
# Base Classes
# =============================================================================

class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    @abstractmethod
    def accept(self, visitor: 'ASTVisitor') -> Any:
        """Accept a visitor for traversal."""
        pass


class Expression(ASTNode):
    """Base class for expression nodes."""
    pass


class Statement(ASTNode):
    """Base class for statement nodes."""
    pass


# =============================================================================
# Expressions
# =============================================================================

@dataclass
class LiteralExpr(Expression):
    """Literal value expression (number, string, bool, nil)."""
    value: Any
    token: Token
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_literal(self)


@dataclass
class IdentifierExpr(Expression):
    """Variable or function name reference."""
    name: str
    token: Token
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_identifier(self)


@dataclass
class UnaryExpr(Expression):
    """Unary operator expression (-, not, #)."""
    operator: Token
    operand: Expression
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_unary(self)


@dataclass
class BinaryExpr(Expression):
    """Binary operator expression."""
    left: Expression
    operator: Token
    right: Expression
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_binary(self)


@dataclass
class TernaryExpr(Expression):
    """Ternary conditional expression (cond ? then : else)."""
    condition: Expression
    then_expr: Expression
    else_expr: Expression
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_ternary(self)


@dataclass
class GroupExpr(Expression):
    """Parenthesized expression."""
    expression: Expression
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_group(self)


@dataclass
class CallExpr(Expression):
    """Function call expression."""
    callee: Expression
    arguments: List[Expression]
    paren: Token  # For error reporting
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_call(self)


@dataclass
class IndexExpr(Expression):
    """Index/subscript expression (a[b])."""
    object: Expression
    index: Expression
    bracket: Token
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_index(self)


@dataclass
class DotExpr(Expression):
    """Property access expression (a.b)."""
    object: Expression
    name: Token
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_dot(self)


@dataclass
class AssignExpr(Expression):
    """Assignment expression."""
    target: Expression
    operator: Token
    value: Expression
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_assign(self)


@dataclass
class TableExpr(Expression):
    """Table literal expression."""
    entries: List['TableEntry']
    brace: Token
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_table(self)


@dataclass
class TableEntry:
    """Single entry in a table literal."""
    key: Optional[Expression]  # None for array-style entries
    value: Expression


@dataclass
class FunctionExpr(Expression):
    """Anonymous function expression."""
    params: List[Token]
    body: 'BlockStmt'
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_function_expr(self)


# =============================================================================
# Statements
# =============================================================================

@dataclass
class ExpressionStmt(Statement):
    """Expression as a statement."""
    expression: Expression
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_expression_stmt(self)


@dataclass
class VarDeclStmt(Statement):
    """Variable declaration statement."""
    name: Token
    initializer: Optional[Expression]
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_var_decl(self)


@dataclass
class FunctionDeclStmt(Statement):
    """Function declaration statement."""
    name: Token
    params: List[Token]
    body: 'BlockStmt'
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_function_decl(self)


@dataclass
class BlockStmt(Statement):
    """Block of statements."""
    statements: List[Statement]
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_block(self)


@dataclass
class IfStmt(Statement):
    """If/else statement."""
    condition: Expression
    then_branch: Statement
    else_branch: Optional[Statement]
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_if(self)


@dataclass
class WhileStmt(Statement):
    """While loop statement."""
    condition: Expression
    body: Statement
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_while(self)


@dataclass
class DoWhileStmt(Statement):
    """Do-while loop statement."""
    body: Statement
    condition: Expression
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_do_while(self)


@dataclass
class ForStmt(Statement):
    """For loop statement."""
    initializer: Optional[Statement]
    condition: Optional[Expression]
    increment: Optional[Expression]
    body: Statement
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_for(self)


@dataclass
class BreakStmt(Statement):
    """Break statement."""
    keyword: Token
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_break(self)


@dataclass
class ContinueStmt(Statement):
    """Continue statement."""
    keyword: Token
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_continue(self)


@dataclass
class ReturnStmt(Statement):
    """Return statement."""
    keyword: Token
    values: List[Expression]
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_return(self)


@dataclass
class Program(ASTNode):
    """Root node of the AST."""
    statements: List[Statement]
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        return visitor.visit_program(self)


# =============================================================================
# Visitor Interface
# =============================================================================

class ASTVisitor(ABC):
    """Visitor interface for AST traversal."""
    
    # Expressions
    @abstractmethod
    def visit_literal(self, node: LiteralExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_identifier(self, node: IdentifierExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_unary(self, node: UnaryExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_binary(self, node: BinaryExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_ternary(self, node: TernaryExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_group(self, node: GroupExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_call(self, node: CallExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_index(self, node: IndexExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_dot(self, node: DotExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_assign(self, node: AssignExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_table(self, node: TableExpr) -> Any:
        pass
    
    @abstractmethod
    def visit_function_expr(self, node: FunctionExpr) -> Any:
        pass
    
    # Statements
    @abstractmethod
    def visit_expression_stmt(self, node: ExpressionStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_var_decl(self, node: VarDeclStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_function_decl(self, node: FunctionDeclStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_block(self, node: BlockStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_if(self, node: IfStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_while(self, node: WhileStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_do_while(self, node: DoWhileStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_for(self, node: ForStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_break(self, node: BreakStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_continue(self, node: ContinueStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_return(self, node: ReturnStmt) -> Any:
        pass
    
    @abstractmethod
    def visit_program(self, node: Program) -> Any:
        pass


# =============================================================================
# AST Printer (for debugging)
# =============================================================================

class ASTPrinter(ASTVisitor):
    """Prints AST for debugging."""
    
    def __init__(self):
        self.indent = 0
    
    def print(self, node: ASTNode) -> str:
        return node.accept(self)
    
    def _indent(self) -> str:
        return "  " * self.indent
    
    def visit_literal(self, node: LiteralExpr) -> str:
        return f"{self._indent()}Literal({node.value!r})"
    
    def visit_identifier(self, node: IdentifierExpr) -> str:
        return f"{self._indent()}Identifier({node.name})"
    
    def visit_unary(self, node: UnaryExpr) -> str:
        self.indent += 1
        operand = node.operand.accept(self)
        self.indent -= 1
        return f"{self._indent()}Unary({node.operator.lexeme})\n{operand}"
    
    def visit_binary(self, node: BinaryExpr) -> str:
        self.indent += 1
        left = node.left.accept(self)
        right = node.right.accept(self)
        self.indent -= 1
        return f"{self._indent()}Binary({node.operator.lexeme})\n{left}\n{right}"
    
    def visit_ternary(self, node: TernaryExpr) -> str:
        self.indent += 1
        cond = node.condition.accept(self)
        then = node.then_expr.accept(self)
        else_ = node.else_expr.accept(self)
        self.indent -= 1
        return f"{self._indent()}Ternary\n{cond}\n{then}\n{else_}"
    
    def visit_group(self, node: GroupExpr) -> str:
        return node.expression.accept(self)
    
    def visit_call(self, node: CallExpr) -> str:
        self.indent += 1
        callee = node.callee.accept(self)
        args = [arg.accept(self) for arg in node.arguments]
        self.indent -= 1
        args_str = "\n".join(args) if args else ""
        return f"{self._indent()}Call\n{callee}\n{args_str}"
    
    def visit_index(self, node: IndexExpr) -> str:
        self.indent += 1
        obj = node.object.accept(self)
        idx = node.index.accept(self)
        self.indent -= 1
        return f"{self._indent()}Index\n{obj}\n{idx}"
    
    def visit_dot(self, node: DotExpr) -> str:
        self.indent += 1
        obj = node.object.accept(self)
        self.indent -= 1
        return f"{self._indent()}Dot(.{node.name.lexeme})\n{obj}"
    
    def visit_assign(self, node: AssignExpr) -> str:
        self.indent += 1
        target = node.target.accept(self)
        value = node.value.accept(self)
        self.indent -= 1
        return f"{self._indent()}Assign({node.operator.lexeme})\n{target}\n{value}"
    
    def visit_table(self, node: TableExpr) -> str:
        self.indent += 1
        entries = []
        for entry in node.entries:
            if entry.key:
                key = entry.key.accept(self)
                val = entry.value.accept(self)
                entries.append(f"{key}: {val}")
            else:
                entries.append(entry.value.accept(self))
        self.indent -= 1
        return f"{self._indent()}Table\n" + "\n".join(entries)
    
    def visit_function_expr(self, node: FunctionExpr) -> str:
        params = ", ".join(p.lexeme for p in node.params)
        self.indent += 1
        body = node.body.accept(self)
        self.indent -= 1
        return f"{self._indent()}Function({params})\n{body}"
    
    def visit_expression_stmt(self, node: ExpressionStmt) -> str:
        return node.expression.accept(self)
    
    def visit_var_decl(self, node: VarDeclStmt) -> str:
        if node.initializer:
            self.indent += 1
            init = node.initializer.accept(self)
            self.indent -= 1
            return f"{self._indent()}VarDecl({node.name.lexeme})\n{init}"
        return f"{self._indent()}VarDecl({node.name.lexeme})"
    
    def visit_function_decl(self, node: FunctionDeclStmt) -> str:
        params = ", ".join(p.lexeme for p in node.params)
        self.indent += 1
        body = node.body.accept(self)
        self.indent -= 1
        return f"{self._indent()}FunctionDecl({node.name.lexeme}({params}))\n{body}"
    
    def visit_block(self, node: BlockStmt) -> str:
        self.indent += 1
        stmts = [stmt.accept(self) for stmt in node.statements]
        self.indent -= 1
        return f"{self._indent()}Block\n" + "\n".join(stmts)
    
    def visit_if(self, node: IfStmt) -> str:
        self.indent += 1
        cond = node.condition.accept(self)
        then = node.then_branch.accept(self)
        else_ = node.else_branch.accept(self) if node.else_branch else ""
        self.indent -= 1
        result = f"{self._indent()}If\n{cond}\n{then}"
        if else_:
            result += f"\n{else_}"
        return result
    
    def visit_while(self, node: WhileStmt) -> str:
        self.indent += 1
        cond = node.condition.accept(self)
        body = node.body.accept(self)
        self.indent -= 1
        return f"{self._indent()}While\n{cond}\n{body}"
    
    def visit_do_while(self, node: DoWhileStmt) -> str:
        self.indent += 1
        body = node.body.accept(self)
        cond = node.condition.accept(self)
        self.indent -= 1
        return f"{self._indent()}DoWhile\n{body}\n{cond}"
    
    def visit_for(self, node: ForStmt) -> str:
        self.indent += 1
        parts = []
        if node.initializer:
            parts.append(node.initializer.accept(self))
        if node.condition:
            parts.append(node.condition.accept(self))
        if node.increment:
            parts.append(node.increment.accept(self))
        parts.append(node.body.accept(self))
        self.indent -= 1
        return f"{self._indent()}For\n" + "\n".join(parts)
    
    def visit_break(self, node: BreakStmt) -> str:
        return f"{self._indent()}Break"
    
    def visit_continue(self, node: ContinueStmt) -> str:
        return f"{self._indent()}Continue"
    
    def visit_return(self, node: ReturnStmt) -> str:
        if node.values:
            self.indent += 1
            vals = [v.accept(self) for v in node.values]
            self.indent -= 1
            return f"{self._indent()}Return\n" + "\n".join(vals)
        return f"{self._indent()}Return"
    
    def visit_program(self, node: Program) -> str:
        stmts = [stmt.accept(self) for stmt in node.statements]
        return "Program\n" + "\n".join(stmts)

