"""
XScript CPU Interpreter

A Python-based interpreter for XScript bytecode.
Used for debugging and as a fallback when GPU is not available.
"""

from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field

from .types import XValue, XTable, XFunction, TYPE_NIL, TYPE_BOOL, TYPE_NUMBER, TYPE_STRING, TYPE_TABLE, TYPE_FUNCTION
from compiler.bytecode import Bytecode, OpCode, Constant


@dataclass
class CallFrame:
    """A function call frame."""
    return_pc: int
    base_slot: int
    func_name: str


class Interpreter:
    """
    CPU-based interpreter for XScript bytecode.
    
    This provides a reference implementation and debugging capability.
    """
    
    def __init__(self, context):
        """
        Initialize the interpreter.
        
        Args:
            context: XScript Context object
        """
        self.context = context
        self.stack: List[XValue] = []
        self.pc = 0
        self.call_stack: List[CallFrame] = []
        self.bytecode: Optional[Bytecode] = None
    
    def run(self, bytecode: Bytecode) -> XValue:
        """
        Execute bytecode.
        
        Args:
            bytecode: Compiled bytecode
            
        Returns:
            Result value
        """
        self.bytecode = bytecode
        self.stack = []
        self.pc = 0
        self.call_stack = []
        
        try:
            while True:
                result = self.step()
                if result is not None:
                    return result
        except Exception as e:
            raise RuntimeError(f"Runtime error at PC {self.pc}: {e}")
    
    def step(self) -> Optional[XValue]:
        """
        Execute one instruction.
        
        Returns:
            Result value if execution is complete, None otherwise
        """
        if self.pc >= len(self.bytecode.code):
            return XValue.nil()
        
        opcode = OpCode(self.bytecode.code[self.pc])
        self.pc += 1
        
        # Stack operations
        if opcode == OpCode.NOP:
            pass
        
        elif opcode == OpCode.PUSH_NIL:
            self.push(XValue.nil())
        
        elif opcode == OpCode.PUSH_TRUE:
            self.push(XValue.boolean(True))
        
        elif opcode == OpCode.PUSH_FALSE:
            self.push(XValue.boolean(False))
        
        elif opcode == OpCode.PUSH_NUM:
            idx = self.read_u16()
            const = self.bytecode.constants[idx]
            self.push(XValue.number(const.value))
        
        elif opcode == OpCode.PUSH_STR:
            idx = self.read_u16()
            const = self.bytecode.constants[idx]
            self.push(XValue.string(const.value))
        
        elif opcode == OpCode.POP:
            self.pop()
        
        elif opcode == OpCode.DUP:
            self.push(self.peek())
        
        elif opcode == OpCode.SWAP:
            a = self.pop()
            b = self.pop()
            self.push(a)
            self.push(b)
        
        # Local/Global variables
        elif opcode == OpCode.GET_LOCAL:
            slot = self.read_u8()
            base = self.call_stack[-1].base_slot if self.call_stack else 0
            self.push(self.stack[base + slot])
        
        elif opcode == OpCode.SET_LOCAL:
            slot = self.read_u8()
            base = self.call_stack[-1].base_slot if self.call_stack else 0
            self.stack[base + slot] = self.pop()
        
        elif opcode == OpCode.GET_GLOBAL:
            idx = self.read_u16()
            name = self._get_global_name(idx)
            value = self.context.get_global(name)
            self.push(value)
        
        elif opcode == OpCode.SET_GLOBAL:
            idx = self.read_u16()
            name = self._get_global_name(idx)
            value = self.pop()
            self.context.set_global(name, value.to_python())
        
        # Arithmetic
        elif opcode == OpCode.ADD:
            b = self.pop()
            a = self.pop()
            self.push(self._binary_op(a, b, lambda x, y: x + y))
        
        elif opcode == OpCode.SUB:
            b = self.pop()
            a = self.pop()
            self.push(self._binary_op(a, b, lambda x, y: x - y))
        
        elif opcode == OpCode.MUL:
            b = self.pop()
            a = self.pop()
            self.push(self._binary_op(a, b, lambda x, y: x * y))
        
        elif opcode == OpCode.DIV:
            b = self.pop()
            a = self.pop()
            self.push(self._binary_op(a, b, lambda x, y: x / y if y != 0 else float('inf')))
        
        elif opcode == OpCode.MOD:
            b = self.pop()
            a = self.pop()
            self.push(self._binary_op(a, b, lambda x, y: x % y if y != 0 else float('nan')))
        
        elif opcode == OpCode.NEG:
            a = self.pop()
            if a.type == TYPE_NUMBER:
                self.push(XValue.number(-a.data))
            else:
                self.push(XValue.nil())
        
        elif opcode == OpCode.POW:
            b = self.pop()
            a = self.pop()
            self.push(self._binary_op(a, b, lambda x, y: x ** y))
        
        # Comparison
        elif opcode == OpCode.EQ:
            b = self.pop()
            a = self.pop()
            self.push(XValue.boolean(self._equals(a, b)))
        
        elif opcode == OpCode.NE:
            b = self.pop()
            a = self.pop()
            self.push(XValue.boolean(not self._equals(a, b)))
        
        elif opcode == OpCode.LT:
            b = self.pop()
            a = self.pop()
            self.push(self._compare_op(a, b, lambda x, y: x < y))
        
        elif opcode == OpCode.LE:
            b = self.pop()
            a = self.pop()
            self.push(self._compare_op(a, b, lambda x, y: x <= y))
        
        elif opcode == OpCode.GT:
            b = self.pop()
            a = self.pop()
            self.push(self._compare_op(a, b, lambda x, y: x > y))
        
        elif opcode == OpCode.GE:
            b = self.pop()
            a = self.pop()
            self.push(self._compare_op(a, b, lambda x, y: x >= y))
        
        elif opcode == OpCode.NOT:
            a = self.pop()
            self.push(XValue.boolean(not a.is_truthy()))
        
        # Control flow
        elif opcode == OpCode.JMP:
            offset = self.read_i16()
            self.pc += offset
        
        elif opcode == OpCode.JMP_IF:
            offset = self.read_i16()
            cond = self.pop()
            if cond.is_truthy():
                self.pc += offset
        
        elif opcode == OpCode.JMP_IF_NOT:
            offset = self.read_i16()
            cond = self.pop()
            if not cond.is_truthy():
                self.pc += offset
        
        elif opcode == OpCode.LOOP:
            offset = self.read_u16()
            self.pc -= offset
        
        # Function calls
        elif opcode == OpCode.CALL:
            arg_count = self.read_u8()
            func = self.stack[-(arg_count + 1)]
            
            if func.type == TYPE_FUNCTION:
                func_obj = func.data
                if isinstance(func_obj, XFunction):
                    if func_obj.is_host:
                        # Call host function
                        args = [self.pop().to_python() for _ in range(arg_count)]
                        args.reverse()
                        self.pop()  # Pop function
                        result = func_obj.host_func(*args)
                        self.push(XValue.from_python(result))
                    else:
                        # Call script function
                        frame = CallFrame(
                            return_pc=self.pc,
                            base_slot=len(self.stack) - arg_count - 1,
                            func_name=func_obj.name
                        )
                        self.call_stack.append(frame)
                        self.pc = func_obj.code_offset
        
        elif opcode == OpCode.RETURN:
            ret_count = self.read_u8()
            
            # Get return values
            returns = []
            for _ in range(ret_count):
                returns.append(self.pop())
            returns.reverse()
            
            if not self.call_stack:
                # Return from main
                return returns[0] if returns else XValue.nil()
            
            # Pop to base
            frame = self.call_stack.pop()
            while len(self.stack) > frame.base_slot:
                self.pop()
            
            # Push return values
            for ret in returns:
                self.push(ret)
            
            # Restore PC
            self.pc = frame.return_pc
        
        elif opcode == OpCode.CALL_HOST:
            func_idx = self.read_u16()
            arg_count = self.read_u8()
            
            func = self.context.get_host_function(func_idx)
            if func:
                args = [self.pop().to_python() for _ in range(arg_count)]
                args.reverse()
                result = func.host_func(*args)
                self.push(XValue.from_python(result))
            else:
                self.push(XValue.nil())
        
        # Table operations
        elif opcode == OpCode.NEW_TABLE:
            capacity = self.read_u8()
            self.push(XTable().to_xvalue())
        
        elif opcode == OpCode.GET_TABLE:
            key = self.pop()
            table = self.pop()
            if table.type == TYPE_TABLE:
                result = table.data.get(key.to_python())
                self.push(result)
            else:
                self.push(XValue.nil())
        
        elif opcode == OpCode.SET_TABLE:
            value = self.pop()
            key = self.pop()
            table = self.pop()
            if table.type == TYPE_TABLE:
                table.data.set(key.to_python(), value)
        
        elif opcode == OpCode.GET_FIELD:
            idx = self.read_u16()
            const = self.bytecode.constants[idx]
            key = const.value
            
            table = self.pop()
            if table.type == TYPE_TABLE:
                result = table.data.get(key)
                self.push(result)
            else:
                self.push(XValue.nil())
        
        elif opcode == OpCode.SET_FIELD:
            idx = self.read_u16()
            const = self.bytecode.constants[idx]
            key = const.value
            
            value = self.pop()
            table = self.pop()
            if table.type == TYPE_TABLE:
                table.data.set(key, value)
        
        # Metatable operations
        elif opcode == OpCode.GET_META:
            table = self.peek()
            if table.type == TYPE_TABLE:
                mt = table.data.get_metatable()
                self.stack[-1] = mt.to_xvalue() if mt else XValue.nil()
            else:
                self.stack[-1] = XValue.nil()
        
        elif opcode == OpCode.SET_META:
            mt = self.pop()
            table = self.pop()
            if table.type == TYPE_TABLE and mt.type == TYPE_TABLE:
                table.data.set_metatable(mt.data)
            self.push(table)
        
        # Special
        elif opcode == OpCode.HALT:
            return self.pop() if self.stack else XValue.nil()
        
        else:
            raise RuntimeError(f"Unknown opcode: {opcode}")
        
        return None
    
    def push(self, value: XValue) -> None:
        """Push a value onto the stack."""
        self.stack.append(value)
    
    def pop(self) -> XValue:
        """Pop a value from the stack."""
        if not self.stack:
            return XValue.nil()
        return self.stack.pop()
    
    def peek(self, offset: int = 0) -> XValue:
        """Peek at stack value."""
        idx = -(offset + 1)
        if abs(idx) <= len(self.stack):
            return self.stack[idx]
        return XValue.nil()
    
    def read_u8(self) -> int:
        """Read unsigned 8-bit integer."""
        value = self.bytecode.code[self.pc]
        self.pc += 1
        return value
    
    def read_u16(self) -> int:
        """Read unsigned 16-bit integer (little-endian)."""
        lo = self.bytecode.code[self.pc]
        hi = self.bytecode.code[self.pc + 1]
        self.pc += 2
        return lo | (hi << 8)
    
    def read_i16(self) -> int:
        """Read signed 16-bit integer (little-endian)."""
        value = self.read_u16()
        if value >= 0x8000:
            value -= 0x10000
        return value
    
    def _get_global_name(self, idx: int) -> str:
        """Get global variable name by index."""
        for name, i in self.bytecode.globals.items():
            if i == idx:
                return name
        return f"_g{idx}"
    
    def _binary_op(self, a: XValue, b: XValue, op) -> XValue:
        """Perform binary operation on numbers."""
        if a.type == TYPE_NUMBER and b.type == TYPE_NUMBER:
            return XValue.number(op(a.data, b.data))
        return XValue.nil()
    
    def _compare_op(self, a: XValue, b: XValue, op) -> XValue:
        """Perform comparison operation."""
        if a.type == TYPE_NUMBER and b.type == TYPE_NUMBER:
            return XValue.boolean(op(a.data, b.data))
        if a.type == TYPE_STRING and b.type == TYPE_STRING:
            return XValue.boolean(op(a.data, b.data))
        return XValue.boolean(False)
    
    def _equals(self, a: XValue, b: XValue) -> bool:
        """Check equality of two values."""
        if a.type != b.type:
            return False
        if a.type == TYPE_NIL:
            return True
        return a.data == b.data

