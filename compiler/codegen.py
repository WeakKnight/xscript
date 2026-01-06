"""
XScript Code Generator

Generates bytecode from an AST.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from .tokens import Token, TokenType
from .ast import *
from .bytecode import Bytecode, OpCode, Constant, FunctionInfo
from .errors import CompileError


@dataclass
class Local:
    """A local variable in the current scope."""
    name: str
    depth: int
    slot: int
    captured: bool = False


@dataclass
class Upvalue:
    """An upvalue (captured variable from enclosing scope)."""
    index: int
    is_local: bool  # True if captures local, False if captures upvalue


@dataclass
class LoopContext:
    """Context for loop compilation (for break/continue)."""
    start: int
    break_jumps: List[int] = field(default_factory=list)
    continue_jumps: List[int] = field(default_factory=list)


class Scope:
    """Represents a local scope for variable resolution."""
    
    def __init__(self, parent: Optional['Scope'] = None, is_function: bool = False):
        self.parent = parent
        self.is_function = is_function
        self.locals: List[Local] = []
        self.upvalues: List[Upvalue] = []
        self.depth = parent.depth + 1 if parent else 0
        self.next_slot = 0
        
        # Copy parent's slot count if not a function
        if parent and not is_function:
            self.next_slot = parent.next_slot
    
    def add_local(self, name: str) -> int:
        """Add a local variable, returning its slot."""
        slot = self.next_slot
        self.locals.append(Local(name, self.depth, slot))
        self.next_slot += 1
        return slot
    
    def resolve_local(self, name: str) -> Optional[int]:
        """Resolve a local variable, returning its slot or None."""
        for local in reversed(self.locals):
            if local.name == name:
                return local.slot
        return None
    
    def resolve_upvalue(self, name: str) -> Optional[int]:
        """Resolve an upvalue, returning its index or None."""
        if self.parent is None:
            return None
        
        # Check for existing upvalue
        for i, uv in enumerate(self.upvalues):
            # Can't easily check name here, would need to store it
            pass
        
        # Try to capture from parent's locals
        parent_local = self.parent.resolve_local(name)
        if parent_local is not None:
            # Mark as captured
            for local in self.parent.locals:
                if local.slot == parent_local and local.name == name:
                    local.captured = True
                    break
            return self._add_upvalue(parent_local, True)
        
        # Try to capture from parent's upvalues
        parent_upvalue = self.parent.resolve_upvalue(name)
        if parent_upvalue is not None:
            return self._add_upvalue(parent_upvalue, False)
        
        return None
    
    def _add_upvalue(self, index: int, is_local: bool) -> int:
        """Add an upvalue, returning its index."""
        # Check if already exists
        for i, uv in enumerate(self.upvalues):
            if uv.index == index and uv.is_local == is_local:
                return i
        
        idx = len(self.upvalues)
        self.upvalues.append(Upvalue(index, is_local))
        return idx
    
    def local_count(self) -> int:
        """Get the number of local variables in this scope."""
        if self.is_function:
            return self.next_slot
        elif self.parent:
            return self.next_slot - self.parent.next_slot
        return self.next_slot


class CodeGenerator(ASTVisitor):
    """Generates bytecode from an AST."""
    
    def __init__(self):
        self.bytecode = Bytecode()
        self.scope: Optional[Scope] = None
        self.loop_stack: List[LoopContext] = []
        self.current_function: Optional[FunctionInfo] = None
    
    def generate(self, program: Program) -> Bytecode:
        """Generate bytecode from a program AST."""
        self.bytecode = Bytecode()
        self.scope = Scope()  # Global scope
        
        # Visit all statements
        program.accept(self)
        
        # Add halt instruction
        self.bytecode.emit(OpCode.HALT)
        
        return self.bytecode
    
    # =========================================================================
    # Scope Management
    # =========================================================================
    
    def begin_scope(self, is_function: bool = False) -> None:
        """Begin a new scope."""
        self.scope = Scope(self.scope, is_function)
    
    def end_scope(self) -> int:
        """End the current scope, returning local count."""
        local_count = self.scope.local_count()
        
        # Pop locals
        for _ in range(local_count):
            self.bytecode.emit(OpCode.POP)
        
        self.scope = self.scope.parent
        return local_count
    
    def add_local(self, name: str) -> int:
        """Add a local variable in the current scope."""
        return self.scope.add_local(name)
    
    def resolve_variable(self, name: str) -> Tuple[str, int]:
        """
        Resolve a variable name.
        Returns: ('local', slot), ('upvalue', index), or ('global', index)
        """
        # Check locals
        local = self.scope.resolve_local(name)
        if local is not None:
            return ('local', local)
        
        # Check upvalues
        upvalue = self.scope.resolve_upvalue(name)
        if upvalue is not None:
            return ('upvalue', upvalue)
        
        # Must be global
        return ('global', self.bytecode.add_global(name))
    
    # =========================================================================
    # Expression Visitors
    # =========================================================================
    
    def visit_literal(self, node: LiteralExpr) -> None:
        """Generate code for literal expression."""
        value = node.value
        line = node.token.line
        
        if value is None:
            self.bytecode.emit(OpCode.PUSH_NIL, line)
        elif value is True:
            self.bytecode.emit(OpCode.PUSH_TRUE, line)
        elif value is False:
            self.bytecode.emit(OpCode.PUSH_FALSE, line)
        elif isinstance(value, (int, float)):
            idx = self.bytecode.add_constant(Constant.number(float(value)))
            self.bytecode.emit_with_u16(OpCode.PUSH_NUM, idx, line)
        elif isinstance(value, str):
            idx = self.bytecode.add_constant(Constant.string(value))
            self.bytecode.emit_with_u16(OpCode.PUSH_STR, idx, line)
        else:
            raise CompileError(f"Unknown literal type: {type(value)}", line)
    
    def visit_identifier(self, node: IdentifierExpr) -> None:
        """Generate code for identifier expression."""
        kind, index = self.resolve_variable(node.name)
        line = node.token.line
        
        if kind == 'local':
            self.bytecode.emit_with_u8(OpCode.GET_LOCAL, index, line)
        elif kind == 'upvalue':
            self.bytecode.emit_with_u8(OpCode.GET_UPVALUE, index, line)
        else:
            self.bytecode.emit_with_u16(OpCode.GET_GLOBAL, index, line)
    
    def visit_unary(self, node: UnaryExpr) -> None:
        """Generate code for unary expression."""
        # Generate operand
        node.operand.accept(self)
        
        line = node.operator.line
        op = node.operator.type
        
        if op == TokenType.MINUS:
            self.bytecode.emit(OpCode.NEG, line)
        elif op == TokenType.NOT:
            self.bytecode.emit(OpCode.NOT, line)
        elif op == TokenType.HASH:
            # Length operator - implement as method call later
            # For now, emit a placeholder
            pass
        else:
            raise CompileError(f"Unknown unary operator: {op}", line)
    
    def visit_binary(self, node: BinaryExpr) -> None:
        """Generate code for binary expression."""
        op = node.operator.type
        line = node.operator.line
        
        # Short-circuit operators
        if op == TokenType.AND:
            node.left.accept(self)
            jump = self.bytecode.emit_jump(OpCode.JMP_IF_NOT, line)
            self.bytecode.emit(OpCode.POP)
            node.right.accept(self)
            self.bytecode.patch_jump(jump)
            return
        
        if op == TokenType.OR:
            node.left.accept(self)
            jump = self.bytecode.emit_jump(OpCode.JMP_IF, line)
            self.bytecode.emit(OpCode.POP)
            node.right.accept(self)
            self.bytecode.patch_jump(jump)
            return
        
        # Regular binary operators
        node.left.accept(self)
        node.right.accept(self)
        
        ops = {
            TokenType.PLUS: OpCode.ADD,
            TokenType.MINUS: OpCode.SUB,
            TokenType.STAR: OpCode.MUL,
            TokenType.SLASH: OpCode.DIV,
            TokenType.PERCENT: OpCode.MOD,
            TokenType.CARET: OpCode.POW,
            TokenType.EQ: OpCode.EQ,
            TokenType.NE: OpCode.NE,
            TokenType.LT: OpCode.LT,
            TokenType.LE: OpCode.LE,
            TokenType.GT: OpCode.GT,
            TokenType.GE: OpCode.GE,
        }
        
        if op in ops:
            self.bytecode.emit(ops[op], line)
        else:
            raise CompileError(f"Unknown binary operator: {op}", line)
    
    def visit_ternary(self, node: TernaryExpr) -> None:
        """Generate code for ternary expression."""
        # condition
        node.condition.accept(self)
        
        # Jump to else if false
        else_jump = self.bytecode.emit_jump(OpCode.JMP_IF_NOT)
        self.bytecode.emit(OpCode.POP)
        
        # then expression
        node.then_expr.accept(self)
        
        # Jump over else
        end_jump = self.bytecode.emit_jump(OpCode.JMP)
        
        self.bytecode.patch_jump(else_jump)
        self.bytecode.emit(OpCode.POP)
        
        # else expression
        node.else_expr.accept(self)
        
        self.bytecode.patch_jump(end_jump)
    
    def visit_group(self, node: GroupExpr) -> None:
        """Generate code for grouped expression."""
        node.expression.accept(self)
    
    # ECS builtin functions that emit special opcodes
    ECS_BUILTINS: Dict[str, Tuple[OpCode, int]] = {
        # name: (opcode, expected_arg_count)
        "spawn": (OpCode.SPAWN_ENTITY, 1),       # spawn(table) -> entity_id
        "destroy": (OpCode.DESTROY_ENTITY, 1),   # destroy(entity_id)
        "get_entity": (OpCode.GET_ENTITY, 0),    # get_entity() -> entity table
        "get_entity_id": (OpCode.GET_ENTITY_ID, 0),  # get_entity_id() -> entity id
        "has_component": (OpCode.HAS_COMPONENT, 2),  # has_component(entity, key) -> bool
        "add_component": (OpCode.ADD_COMPONENT, 3),  # add_component(entity, key, value)
        "remove_component": (OpCode.REMOVE_COMPONENT, 2),  # remove_component(entity, key)
    }
    
    def visit_call(self, node: CallExpr) -> None:
        """Generate code for function call."""
        # Check for ECS builtin functions
        if isinstance(node.callee, IdentifierExpr):
            func_name = node.callee.name
            if func_name in self.ECS_BUILTINS:
                opcode, expected_args = self.ECS_BUILTINS[func_name]
                
                # Validate argument count
                if len(node.arguments) != expected_args:
                    raise CompileError(
                        f"ECS builtin '{func_name}' expects {expected_args} argument(s), "
                        f"got {len(node.arguments)}",
                        node.paren.line
                    )
                
                # Generate arguments (pushed in order)
                for arg in node.arguments:
                    arg.accept(self)
                
                # Emit ECS opcode
                self.bytecode.emit(opcode, node.paren.line)
                return
        
        # Regular function call
        # Generate callee
        node.callee.accept(self)
        
        # Generate arguments
        for arg in node.arguments:
            arg.accept(self)
        
        # Emit call
        self.bytecode.emit_with_u8(OpCode.CALL, len(node.arguments), node.paren.line)
    
    def visit_index(self, node: IndexExpr) -> None:
        """Generate code for index expression."""
        node.object.accept(self)
        node.index.accept(self)
        self.bytecode.emit(OpCode.GET_TABLE, node.bracket.line)
    
    def visit_dot(self, node: DotExpr) -> None:
        """Generate code for dot expression."""
        node.object.accept(self)
        
        # Add field name as constant
        idx = self.bytecode.add_constant(Constant.string(node.name.lexeme))
        self.bytecode.emit_with_u16(OpCode.GET_FIELD, idx, node.name.line)
    
    def visit_assign(self, node: AssignExpr) -> None:
        """Generate code for assignment expression."""
        line = node.operator.line
        op = node.operator.type
        
        # Handle compound assignment
        if op != TokenType.ASSIGN:
            # Load current value
            node.target.accept(self)
            # Load new value
            node.value.accept(self)
            # Apply operator
            compound_ops = {
                TokenType.PLUS_ASSIGN: OpCode.ADD,
                TokenType.MINUS_ASSIGN: OpCode.SUB,
                TokenType.STAR_ASSIGN: OpCode.MUL,
                TokenType.SLASH_ASSIGN: OpCode.DIV,
                TokenType.PERCENT_ASSIGN: OpCode.MOD,
            }
            self.bytecode.emit(compound_ops[op], line)
        else:
            # Simple assignment - just evaluate value
            node.value.accept(self)
        
        # Duplicate for expression result
        self.bytecode.emit(OpCode.DUP)
        
        # Store to target
        if isinstance(node.target, IdentifierExpr):
            kind, index = self.resolve_variable(node.target.name)
            if kind == 'local':
                self.bytecode.emit_with_u8(OpCode.SET_LOCAL, index, line)
            elif kind == 'upvalue':
                self.bytecode.emit_with_u8(OpCode.SET_UPVALUE, index, line)
            else:
                self.bytecode.emit_with_u16(OpCode.SET_GLOBAL, index, line)
        
        elif isinstance(node.target, IndexExpr):
            # Re-evaluate object and index
            node.target.object.accept(self)
            node.target.index.accept(self)
            # Stack: value, table, key -> need to reorder
            # For now, emit a simpler version
            self.bytecode.emit(OpCode.SET_TABLE, line)
        
        elif isinstance(node.target, DotExpr):
            node.target.object.accept(self)
            idx = self.bytecode.add_constant(Constant.string(node.target.name.lexeme))
            self.bytecode.emit_with_u16(OpCode.SET_FIELD, idx, line)
    
    def visit_table(self, node: TableExpr) -> None:
        """Generate code for table literal."""
        # Create new table
        capacity = max(8, len(node.entries) * 2)
        self.bytecode.emit_with_u8(OpCode.NEW_TABLE, min(capacity, 255), node.brace.line)
        
        array_idx = 0
        for entry in node.entries:
            # Duplicate table reference
            self.bytecode.emit(OpCode.DUP)
            
            # Key
            if entry.key is None:
                # Array-style entry
                idx = self.bytecode.add_constant(Constant.number(float(array_idx)))
                self.bytecode.emit_with_u16(OpCode.PUSH_NUM, idx)
                array_idx += 1
            else:
                entry.key.accept(self)
            
            # Value
            entry.value.accept(self)
            
            # Set table entry
            self.bytecode.emit(OpCode.SET_TABLE)
    
    def visit_function_expr(self, node: FunctionExpr) -> None:
        """Generate code for anonymous function expression."""
        # Create function info
        func_name = f"<anon_{len(self.bytecode.functions)}>"
        func_start = self.bytecode.current_offset()
        
        # Begin function scope
        self.begin_scope(is_function=True)
        
        # Add parameters as locals
        for param in node.params:
            self.add_local(param.lexeme)
        
        # Generate body
        for stmt in node.body.statements:
            stmt.accept(self)
        
        # Implicit return nil
        self.bytecode.emit(OpCode.PUSH_NIL)
        self.bytecode.emit_with_u8(OpCode.RETURN, 1)
        
        func_end = self.bytecode.current_offset()
        local_count = self.scope.local_count()
        upvalue_count = len(self.scope.upvalues)
        
        self.end_scope()
        
        # Add function info
        func_idx = len(self.bytecode.functions)
        self.bytecode.functions.append(FunctionInfo(
            name=func_name,
            arity=len(node.params),
            local_count=local_count,
            upvalue_count=upvalue_count,
            code_offset=func_start,
            code_length=func_end - func_start
        ))
        
        # Push function value
        idx = self.bytecode.add_constant(Constant.number(float(func_idx)))
        self.bytecode.emit_with_u16(OpCode.PUSH_NUM, idx)
    
    # =========================================================================
    # Statement Visitors
    # =========================================================================
    
    def visit_expression_stmt(self, node: ExpressionStmt) -> None:
        """Generate code for expression statement."""
        node.expression.accept(self)
        self.bytecode.emit(OpCode.POP)
    
    def visit_var_decl(self, node: VarDeclStmt) -> None:
        """Generate code for variable declaration."""
        line = node.name.line
        
        if node.initializer:
            node.initializer.accept(self)
        else:
            self.bytecode.emit(OpCode.PUSH_NIL, line)
        
        # If in local scope, add as local
        if self.scope.depth > 0:
            self.add_local(node.name.lexeme)
            # Value is already on stack, becomes the local
        else:
            # Global variable
            idx = self.bytecode.add_global(node.name.lexeme)
            self.bytecode.emit_with_u16(OpCode.SET_GLOBAL, idx, line)
    
    def visit_function_decl(self, node: FunctionDeclStmt) -> None:
        """Generate code for function declaration."""
        line = node.name.line
        func_name = node.name.lexeme
        
        # Record start position
        func_start = self.bytecode.current_offset()
        
        # Begin function scope
        self.begin_scope(is_function=True)
        
        # Add parameters as locals
        for param in node.params:
            self.add_local(param.lexeme)
        
        # Generate body
        for stmt in node.body.statements:
            stmt.accept(self)
        
        # Implicit return nil
        self.bytecode.emit(OpCode.PUSH_NIL)
        self.bytecode.emit_with_u8(OpCode.RETURN, 1)
        
        func_end = self.bytecode.current_offset()
        local_count = self.scope.local_count()
        upvalue_count = len(self.scope.upvalues)
        
        self.end_scope()
        
        # Add function info
        func_idx = len(self.bytecode.functions)
        self.bytecode.functions.append(FunctionInfo(
            name=func_name,
            arity=len(node.params),
            local_count=local_count,
            upvalue_count=upvalue_count,
            code_offset=func_start,
            code_length=func_end - func_start
        ))
        
        # Store function in global
        idx = self.bytecode.add_constant(Constant.number(float(func_idx)))
        self.bytecode.emit_with_u16(OpCode.PUSH_NUM, idx, line)
        
        global_idx = self.bytecode.add_global(func_name)
        self.bytecode.emit_with_u16(OpCode.SET_GLOBAL, global_idx, line)
    
    def visit_block(self, node: BlockStmt) -> None:
        """Generate code for block statement."""
        self.begin_scope()
        
        for stmt in node.statements:
            stmt.accept(self)
        
        self.end_scope()
    
    def visit_if(self, node: IfStmt) -> None:
        """Generate code for if statement."""
        # Condition
        node.condition.accept(self)
        
        # Jump to else if false
        else_jump = self.bytecode.emit_jump(OpCode.JMP_IF_NOT)
        self.bytecode.emit(OpCode.POP)
        
        # Then branch
        node.then_branch.accept(self)
        
        if node.else_branch:
            # Jump over else
            end_jump = self.bytecode.emit_jump(OpCode.JMP)
            
            self.bytecode.patch_jump(else_jump)
            self.bytecode.emit(OpCode.POP)
            
            # Else branch
            node.else_branch.accept(self)
            
            self.bytecode.patch_jump(end_jump)
        else:
            self.bytecode.patch_jump(else_jump)
            self.bytecode.emit(OpCode.POP)
    
    def visit_while(self, node: WhileStmt) -> None:
        """Generate code for while statement."""
        loop_start = self.bytecode.current_offset()
        
        # Push loop context
        loop_ctx = LoopContext(loop_start)
        self.loop_stack.append(loop_ctx)
        
        # Condition
        node.condition.accept(self)
        
        # Exit if false
        exit_jump = self.bytecode.emit_jump(OpCode.JMP_IF_NOT)
        self.bytecode.emit(OpCode.POP)
        
        # Body
        node.body.accept(self)
        
        # Loop back
        self.emit_loop(loop_start)
        
        # Exit point
        self.bytecode.patch_jump(exit_jump)
        self.bytecode.emit(OpCode.POP)
        
        # Patch break jumps
        for jump in loop_ctx.break_jumps:
            self.bytecode.patch_jump(jump)
        
        self.loop_stack.pop()
    
    def visit_do_while(self, node: DoWhileStmt) -> None:
        """Generate code for do-while statement."""
        loop_start = self.bytecode.current_offset()
        
        # Push loop context
        loop_ctx = LoopContext(loop_start)
        self.loop_stack.append(loop_ctx)
        
        # Body
        node.body.accept(self)
        
        # Patch continue jumps to condition
        condition_start = self.bytecode.current_offset()
        for jump in loop_ctx.continue_jumps:
            self.bytecode.patch_jump(jump)
        
        # Condition
        node.condition.accept(self)
        
        # Loop if true
        self.bytecode.emit_jump(OpCode.JMP_IF)
        # Need to calculate offset back to loop_start
        offset = loop_start - self.bytecode.current_offset()
        self.bytecode.code[-2] = offset & 0xFF
        self.bytecode.code[-1] = (offset >> 8) & 0xFF
        
        self.bytecode.emit(OpCode.POP)
        
        # Patch break jumps
        for jump in loop_ctx.break_jumps:
            self.bytecode.patch_jump(jump)
        
        self.loop_stack.pop()
    
    def visit_for(self, node: ForStmt) -> None:
        """Generate code for for statement."""
        # For loop is transformed to:
        # {
        #   initializer
        #   while (condition) {
        #     body
        #     increment
        #   }
        # }
        
        self.begin_scope()
        
        # Initializer
        if node.initializer:
            node.initializer.accept(self)
        
        loop_start = self.bytecode.current_offset()
        
        # Push loop context
        loop_ctx = LoopContext(loop_start)
        self.loop_stack.append(loop_ctx)
        
        # Condition
        exit_jump = None
        if node.condition:
            node.condition.accept(self)
            exit_jump = self.bytecode.emit_jump(OpCode.JMP_IF_NOT)
            self.bytecode.emit(OpCode.POP)
        
        # Body
        node.body.accept(self)
        
        # Increment (this is where continue jumps to)
        increment_start = self.bytecode.current_offset()
        for jump in loop_ctx.continue_jumps:
            self.bytecode.patch_jump(jump)
        
        if node.increment:
            node.increment.accept(self)
            self.bytecode.emit(OpCode.POP)
        
        # Loop back
        self.emit_loop(loop_start)
        
        # Exit point
        if exit_jump:
            self.bytecode.patch_jump(exit_jump)
            self.bytecode.emit(OpCode.POP)
        
        # Patch break jumps
        for jump in loop_ctx.break_jumps:
            self.bytecode.patch_jump(jump)
        
        self.loop_stack.pop()
        self.end_scope()
    
    def visit_break(self, node: BreakStmt) -> None:
        """Generate code for break statement."""
        if not self.loop_stack:
            raise CompileError("'break' outside of loop", node.keyword.line)
        
        jump = self.bytecode.emit_jump(OpCode.JMP, node.keyword.line)
        self.loop_stack[-1].break_jumps.append(jump)
    
    def visit_continue(self, node: ContinueStmt) -> None:
        """Generate code for continue statement."""
        if not self.loop_stack:
            raise CompileError("'continue' outside of loop", node.keyword.line)
        
        jump = self.bytecode.emit_jump(OpCode.JMP, node.keyword.line)
        self.loop_stack[-1].continue_jumps.append(jump)
    
    def visit_return(self, node: ReturnStmt) -> None:
        """Generate code for return statement."""
        if node.values:
            for value in node.values:
                value.accept(self)
            self.bytecode.emit_with_u8(OpCode.RETURN, len(node.values), node.keyword.line)
        else:
            self.bytecode.emit(OpCode.PUSH_NIL, node.keyword.line)
            self.bytecode.emit_with_u8(OpCode.RETURN, 1, node.keyword.line)
    
    def visit_program(self, node: Program) -> None:
        """Generate code for program."""
        for stmt in node.statements:
            stmt.accept(self)
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def emit_loop(self, loop_start: int) -> None:
        """Emit a loop instruction back to loop_start."""
        offset = self.bytecode.current_offset() - loop_start + 3
        self.bytecode.emit_with_u16(OpCode.LOOP, offset)

