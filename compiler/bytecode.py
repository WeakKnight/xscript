"""
XScript Bytecode Format

Defines bytecode instructions and the compiled bytecode container.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import struct


class OpCode(IntEnum):
    """XScript VM opcodes."""
    
    # Stack operations
    NOP = 0x00
    PUSH_NIL = 0x01
    PUSH_TRUE = 0x02
    PUSH_FALSE = 0x03
    PUSH_NUM = 0x04      # operand: constant index (u16)
    PUSH_STR = 0x05      # operand: constant index (u16)
    POP = 0x06
    DUP = 0x07
    SWAP = 0x08
    
    # Local/Global variables
    GET_LOCAL = 0x10     # operand: slot (u8)
    SET_LOCAL = 0x11     # operand: slot (u8)
    GET_GLOBAL = 0x12    # operand: global index (u16)
    SET_GLOBAL = 0x13    # operand: global index (u16)
    GET_UPVALUE = 0x14   # operand: upvalue index (u8)
    SET_UPVALUE = 0x15   # operand: upvalue index (u8)
    
    # Arithmetic
    ADD = 0x20
    SUB = 0x21
    MUL = 0x22
    DIV = 0x23
    MOD = 0x24
    NEG = 0x25
    POW = 0x26
    
    # Comparison
    EQ = 0x30
    NE = 0x31
    LT = 0x32
    LE = 0x33
    GT = 0x34
    GE = 0x35
    NOT = 0x36
    AND = 0x37
    OR = 0x38
    
    # Control flow
    JMP = 0x40           # operand: offset (i16)
    JMP_IF = 0x41        # operand: offset (i16)
    JMP_IF_NOT = 0x42    # operand: offset (i16)
    LOOP = 0x43          # operand: offset (i16)
    
    # Function calls
    CALL = 0x50          # operand: arg count (u8)
    RETURN = 0x51        # operand: return count (u8)
    CALL_HOST = 0x52     # operand: host func index (u16), arg count (u8)
    
    # Table operations
    NEW_TABLE = 0x60     # operand: initial capacity (u8)
    GET_TABLE = 0x61
    SET_TABLE = 0x62
    GET_FIELD = 0x63     # operand: field name index (u16)
    SET_FIELD = 0x64     # operand: field name index (u16)
    
    # Metatable operations
    GET_META = 0x70
    SET_META = 0x71
    INVOKE_META = 0x72   # operand: meta index (u8), arg count (u8)
    
    # ECS operations
    SPAWN_ENTITY = 0x80      # Pop table, push entity ID
    DESTROY_ENTITY = 0x81    # Pop entity ID, mark for destruction
    GET_ENTITY = 0x82        # Push current dispatch entity table
    GET_ENTITY_ID = 0x83     # Push current dispatch entity ID
    HAS_COMPONENT = 0x84     # Pop key, pop entity, push bool
    ADD_COMPONENT = 0x85     # Pop value, pop key, pop entity
    REMOVE_COMPONENT = 0x86  # Pop key, pop entity
    
    # Special
    HALT = 0xFF


# Instruction size information
OPCODE_SIZES = {
    OpCode.NOP: 1,
    OpCode.PUSH_NIL: 1,
    OpCode.PUSH_TRUE: 1,
    OpCode.PUSH_FALSE: 1,
    OpCode.PUSH_NUM: 3,
    OpCode.PUSH_STR: 3,
    OpCode.POP: 1,
    OpCode.DUP: 1,
    OpCode.SWAP: 1,
    OpCode.GET_LOCAL: 2,
    OpCode.SET_LOCAL: 2,
    OpCode.GET_GLOBAL: 3,
    OpCode.SET_GLOBAL: 3,
    OpCode.GET_UPVALUE: 2,
    OpCode.SET_UPVALUE: 2,
    OpCode.ADD: 1,
    OpCode.SUB: 1,
    OpCode.MUL: 1,
    OpCode.DIV: 1,
    OpCode.MOD: 1,
    OpCode.NEG: 1,
    OpCode.POW: 1,
    OpCode.EQ: 1,
    OpCode.NE: 1,
    OpCode.LT: 1,
    OpCode.LE: 1,
    OpCode.GT: 1,
    OpCode.GE: 1,
    OpCode.NOT: 1,
    OpCode.AND: 1,
    OpCode.OR: 1,
    OpCode.JMP: 3,
    OpCode.JMP_IF: 3,
    OpCode.JMP_IF_NOT: 3,
    OpCode.LOOP: 3,
    OpCode.CALL: 2,
    OpCode.RETURN: 2,
    OpCode.CALL_HOST: 4,
    OpCode.NEW_TABLE: 2,
    OpCode.GET_TABLE: 1,
    OpCode.SET_TABLE: 1,
    OpCode.GET_FIELD: 3,
    OpCode.SET_FIELD: 3,
    OpCode.GET_META: 1,
    OpCode.SET_META: 1,
    OpCode.INVOKE_META: 3,
    OpCode.SPAWN_ENTITY: 1,
    OpCode.DESTROY_ENTITY: 1,
    OpCode.GET_ENTITY: 1,
    OpCode.GET_ENTITY_ID: 1,
    OpCode.HAS_COMPONENT: 1,
    OpCode.ADD_COMPONENT: 1,
    OpCode.REMOVE_COMPONENT: 1,
    OpCode.HALT: 1,
}


@dataclass
class Constant:
    """A constant value in the constant pool."""
    
    TYPE_NIL = 0
    TYPE_BOOL = 1
    TYPE_NUMBER = 2
    TYPE_STRING = 3
    
    type: int
    value: Any
    
    @classmethod
    def nil(cls) -> 'Constant':
        return cls(cls.TYPE_NIL, None)
    
    @classmethod
    def boolean(cls, value: bool) -> 'Constant':
        return cls(cls.TYPE_BOOL, value)
    
    @classmethod
    def number(cls, value: float) -> 'Constant':
        return cls(cls.TYPE_NUMBER, value)
    
    @classmethod
    def string(cls, value: str) -> 'Constant':
        return cls(cls.TYPE_STRING, value)


@dataclass
class FunctionInfo:
    """Information about a compiled function."""
    
    name: str
    arity: int              # Number of parameters
    local_count: int        # Number of local variables
    upvalue_count: int      # Number of upvalues
    code_offset: int        # Offset in bytecode
    code_length: int        # Length of function code
    
    # Debug info
    source_line: int = 0
    

@dataclass
class Bytecode:
    """Container for compiled XScript bytecode."""
    
    # Magic number for file format
    MAGIC = b'XSC\x00'
    VERSION = 1
    
    code: bytearray = field(default_factory=bytearray)
    constants: List[Constant] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    globals: Dict[str, int] = field(default_factory=dict)
    strings: List[str] = field(default_factory=list)
    
    # Debug information
    line_numbers: List[int] = field(default_factory=list)  # Line number per instruction
    
    def add_constant(self, constant: Constant) -> int:
        """Add a constant to the pool, returning its index."""
        # Check for existing identical constant
        for i, c in enumerate(self.constants):
            if c.type == constant.type and c.value == constant.value:
                return i
        
        index = len(self.constants)
        self.constants.append(constant)
        return index
    
    def add_string(self, s: str) -> int:
        """Add a string to the string pool, returning its index."""
        if s in self.strings:
            return self.strings.index(s)
        
        index = len(self.strings)
        self.strings.append(s)
        return index
    
    def add_global(self, name: str) -> int:
        """Add or get a global variable index."""
        if name not in self.globals:
            self.globals[name] = len(self.globals)
        return self.globals[name]
    
    def emit(self, opcode: OpCode, line: int = 0) -> int:
        """Emit a single-byte instruction."""
        offset = len(self.code)
        self.code.append(opcode)
        self.line_numbers.append(line)
        return offset
    
    def emit_byte(self, byte: int) -> int:
        """Emit a raw byte."""
        offset = len(self.code)
        self.code.append(byte & 0xFF)
        return offset
    
    def emit_u16(self, value: int) -> int:
        """Emit a 16-bit unsigned integer (little-endian)."""
        offset = len(self.code)
        self.code.append(value & 0xFF)
        self.code.append((value >> 8) & 0xFF)
        return offset
    
    def emit_i16(self, value: int) -> int:
        """Emit a 16-bit signed integer (little-endian)."""
        if value < 0:
            value = value + 0x10000
        return self.emit_u16(value)
    
    def emit_with_u8(self, opcode: OpCode, operand: int, line: int = 0) -> int:
        """Emit instruction with 8-bit operand."""
        offset = self.emit(opcode, line)
        self.emit_byte(operand)
        return offset
    
    def emit_with_u16(self, opcode: OpCode, operand: int, line: int = 0) -> int:
        """Emit instruction with 16-bit operand."""
        offset = self.emit(opcode, line)
        self.emit_u16(operand)
        return offset
    
    def emit_with_i16(self, opcode: OpCode, operand: int, line: int = 0) -> int:
        """Emit instruction with signed 16-bit operand."""
        offset = self.emit(opcode, line)
        self.emit_i16(operand)
        return offset
    
    def emit_jump(self, opcode: OpCode, line: int = 0) -> int:
        """Emit a jump instruction with placeholder offset."""
        offset = self.emit(opcode, line)
        self.emit_u16(0xFFFF)  # Placeholder
        return offset
    
    def patch_jump(self, offset: int) -> None:
        """Patch a jump instruction with the current offset."""
        jump = len(self.code) - offset - 3  # -3 for opcode + 2-byte offset
        if jump > 0x7FFF or jump < -0x8000:
            raise ValueError("Jump offset too large")
        
        if jump < 0:
            jump = jump + 0x10000
        
        self.code[offset + 1] = jump & 0xFF
        self.code[offset + 2] = (jump >> 8) & 0xFF
    
    def patch_i16(self, offset: int, value: int) -> None:
        """Patch a 16-bit signed integer at the given offset."""
        if value > 0x7FFF or value < -0x8000:
            raise ValueError("Value too large for i16")
        
        if value < 0:
            value = value + 0x10000
        
        self.code[offset] = value & 0xFF
        self.code[offset + 1] = (value >> 8) & 0xFF
    
    def emit_with_i16(self, opcode: OpCode, value: int, line: int = 0) -> int:
        """Emit an instruction with a signed 16-bit operand."""
        offset = self.emit(opcode, line)
        if value < 0:
            value = value + 0x10000
        self.emit_u16(value)
        return offset
    
    def current_offset(self) -> int:
        """Get the current code offset."""
        return len(self.code)
    
    def serialize(self) -> bytes:
        """Serialize bytecode to binary format."""
        output = bytearray()
        
        # Header
        output.extend(self.MAGIC)
        output.extend(struct.pack('<H', self.VERSION))
        output.extend(struct.pack('<H', 0))  # Flags
        
        # Constant pool offset (placeholder)
        const_offset_pos = len(output)
        output.extend(struct.pack('<I', 0))
        
        # Code offset (placeholder)
        code_offset_pos = len(output)
        output.extend(struct.pack('<I', 0))
        
        # Functions offset (placeholder)
        func_offset_pos = len(output)
        output.extend(struct.pack('<I', 0))
        
        # Write constant pool
        const_offset = len(output)
        struct.pack_into('<I', output, const_offset_pos, const_offset)
        
        output.extend(struct.pack('<I', len(self.constants)))
        for const in self.constants:
            output.append(const.type)
            if const.type == Constant.TYPE_NIL:
                pass
            elif const.type == Constant.TYPE_BOOL:
                output.append(1 if const.value else 0)
            elif const.type == Constant.TYPE_NUMBER:
                output.extend(struct.pack('<d', const.value))
            elif const.type == Constant.TYPE_STRING:
                encoded = const.value.encode('utf-8')
                output.extend(struct.pack('<H', len(encoded)))
                output.extend(encoded)
        
        # Write functions
        func_offset = len(output)
        struct.pack_into('<I', output, func_offset_pos, func_offset)
        
        output.extend(struct.pack('<I', len(self.functions)))
        for func in self.functions:
            name_encoded = func.name.encode('utf-8')
            output.extend(struct.pack('<H', len(name_encoded)))
            output.extend(name_encoded)
            output.extend(struct.pack('<B', func.arity))
            output.extend(struct.pack('<B', func.local_count))
            output.extend(struct.pack('<B', func.upvalue_count))
            output.extend(struct.pack('<I', func.code_offset))
            output.extend(struct.pack('<I', func.code_length))
        
        # Write code
        code_offset = len(output)
        struct.pack_into('<I', output, code_offset_pos, code_offset)
        
        output.extend(struct.pack('<I', len(self.code)))
        output.extend(self.code)
        
        return bytes(output)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'Bytecode':
        """Deserialize bytecode from binary format."""
        offset = 0
        
        # Header
        magic = data[offset:offset+4]
        if magic != cls.MAGIC:
            raise ValueError("Invalid bytecode magic number")
        offset += 4
        
        version = struct.unpack_from('<H', data, offset)[0]
        if version != cls.VERSION:
            raise ValueError(f"Unsupported bytecode version: {version}")
        offset += 2
        
        flags = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        const_offset = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        code_offset = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        func_offset = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        bc = cls()
        
        # Read constants
        offset = const_offset
        const_count = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        for _ in range(const_count):
            const_type = data[offset]
            offset += 1
            
            if const_type == Constant.TYPE_NIL:
                bc.constants.append(Constant.nil())
            elif const_type == Constant.TYPE_BOOL:
                bc.constants.append(Constant.boolean(data[offset] != 0))
                offset += 1
            elif const_type == Constant.TYPE_NUMBER:
                value = struct.unpack_from('<d', data, offset)[0]
                bc.constants.append(Constant.number(value))
                offset += 8
            elif const_type == Constant.TYPE_STRING:
                length = struct.unpack_from('<H', data, offset)[0]
                offset += 2
                value = data[offset:offset+length].decode('utf-8')
                bc.constants.append(Constant.string(value))
                offset += length
        
        # Read functions
        offset = func_offset
        func_count = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        for _ in range(func_count):
            name_len = struct.unpack_from('<H', data, offset)[0]
            offset += 2
            name = data[offset:offset+name_len].decode('utf-8')
            offset += name_len
            arity = data[offset]
            offset += 1
            local_count = data[offset]
            offset += 1
            upvalue_count = data[offset]
            offset += 1
            code_off = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            code_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            
            bc.functions.append(FunctionInfo(
                name=name,
                arity=arity,
                local_count=local_count,
                upvalue_count=upvalue_count,
                code_offset=code_off,
                code_length=code_len
            ))
        
        # Read code
        offset = code_offset
        code_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        bc.code = bytearray(data[offset:offset+code_len])
        
        return bc
    
    def disassemble(self) -> str:
        """Disassemble bytecode to human-readable format."""
        lines = []
        lines.append("=== XScript Bytecode ===")
        lines.append("")
        
        # Constants
        lines.append("Constants:")
        for i, const in enumerate(self.constants):
            if const.type == Constant.TYPE_NIL:
                lines.append(f"  [{i:4d}] nil")
            elif const.type == Constant.TYPE_BOOL:
                lines.append(f"  [{i:4d}] bool: {const.value}")
            elif const.type == Constant.TYPE_NUMBER:
                lines.append(f"  [{i:4d}] number: {const.value}")
            elif const.type == Constant.TYPE_STRING:
                lines.append(f"  [{i:4d}] string: {const.value!r}")
        lines.append("")
        
        # Functions
        lines.append("Functions:")
        for func in self.functions:
            lines.append(f"  {func.name}(arity={func.arity}, locals={func.local_count})")
            lines.append(f"    offset: {func.code_offset}, length: {func.code_length}")
        lines.append("")
        
        # Globals
        lines.append("Globals:")
        for name, idx in self.globals.items():
            lines.append(f"  [{idx:4d}] {name}")
        lines.append("")
        
        # Code
        lines.append("Code:")
        offset = 0
        while offset < len(self.code):
            line = self._disassemble_instruction(offset)
            lines.append(line)
            opcode = OpCode(self.code[offset])
            offset += OPCODE_SIZES.get(opcode, 1)
        
        return "\n".join(lines)
    
    def _disassemble_instruction(self, offset: int) -> str:
        """Disassemble a single instruction."""
        opcode = OpCode(self.code[offset])
        name = opcode.name
        
        if opcode in (OpCode.GET_LOCAL, OpCode.SET_LOCAL, OpCode.CALL, 
                     OpCode.RETURN, OpCode.NEW_TABLE, OpCode.GET_UPVALUE,
                     OpCode.SET_UPVALUE):
            operand = self.code[offset + 1]
            return f"  {offset:04x}: {name:16s} {operand}"
        
        elif opcode in (OpCode.PUSH_NUM, OpCode.PUSH_STR, OpCode.GET_GLOBAL,
                       OpCode.SET_GLOBAL, OpCode.GET_FIELD, OpCode.SET_FIELD):
            operand = self.code[offset + 1] | (self.code[offset + 2] << 8)
            if opcode in (OpCode.PUSH_NUM, OpCode.PUSH_STR):
                const = self.constants[operand] if operand < len(self.constants) else None
                const_str = f" ; {const.value!r}" if const else ""
                return f"  {offset:04x}: {name:16s} {operand}{const_str}"
            return f"  {offset:04x}: {name:16s} {operand}"
        
        elif opcode in (OpCode.JMP, OpCode.JMP_IF, OpCode.JMP_IF_NOT, OpCode.LOOP):
            raw = self.code[offset + 1] | (self.code[offset + 2] << 8)
            if raw >= 0x8000:
                operand = raw - 0x10000
            else:
                operand = raw
            target = offset + 3 + operand
            return f"  {offset:04x}: {name:16s} {operand:+d} -> {target:04x}"
        
        elif opcode == OpCode.CALL_HOST:
            func_idx = self.code[offset + 1] | (self.code[offset + 2] << 8)
            arg_count = self.code[offset + 3]
            return f"  {offset:04x}: {name:16s} func={func_idx}, args={arg_count}"
        
        elif opcode == OpCode.INVOKE_META:
            meta_idx = self.code[offset + 1]
            arg_count = self.code[offset + 2]
            return f"  {offset:04x}: {name:16s} meta={meta_idx}, args={arg_count}"
        
        else:
            return f"  {offset:04x}: {name}"

