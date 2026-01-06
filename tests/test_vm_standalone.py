"""
XScript VM Standalone Tests

This module runs the Slang-based VM tests without SlangPy (CPU simulation).
For full GPU testing, SlangPy would be required.

This provides a CPU-based simulation for initial debugging.
"""

import struct
import sys
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


# =============================================================================
# Type Constants (matching Slang)
# =============================================================================

class XType(IntEnum):
    NIL = 0
    BOOL = 1
    NUMBER = 2
    STRING = 3
    TABLE = 4
    FUNCTION = 5
    USERDATA = 6
    THREAD = 7


class VMStatus(IntEnum):
    RUNNING = 0
    PAUSED = 1
    WAITING_HOST = 2
    ERROR = 3
    COMPLETED = 4


class VMError(IntEnum):
    NONE = 0
    STACK_OVERFLOW = 1
    STACK_UNDERFLOW = 2
    TYPE_ERROR = 3
    DIV_BY_ZERO = 4
    INVALID_OPCODE = 5
    OUT_OF_MEMORY = 6
    CALL_DEPTH = 7


# =============================================================================
# Opcodes (matching Slang)
# =============================================================================

class Op(IntEnum):
    NOP = 0x00
    PUSH_NIL = 0x01
    PUSH_TRUE = 0x02
    PUSH_FALSE = 0x03
    PUSH_NUM = 0x04
    PUSH_STR = 0x05
    POP = 0x06
    DUP = 0x07
    SWAP = 0x08
    
    GET_LOCAL = 0x10
    SET_LOCAL = 0x11
    GET_GLOBAL = 0x12
    SET_GLOBAL = 0x13
    
    ADD = 0x20
    SUB = 0x21
    MUL = 0x22
    DIV = 0x23
    MOD = 0x24
    NEG = 0x25
    POW = 0x26
    
    EQ = 0x30
    NE = 0x31
    LT = 0x32
    LE = 0x33
    GT = 0x34
    GE = 0x35
    NOT = 0x36
    
    JMP = 0x40
    JMP_IF = 0x41
    JMP_IF_NOT = 0x42
    
    HALT = 0xFF


# =============================================================================
# XValue (32-bit version)
# =============================================================================

@dataclass
class XValue:
    type: int
    flags: int
    data: int  # 32-bit unsigned
    
    @classmethod
    def nil(cls) -> 'XValue':
        return cls(XType.NIL, 0, 0)
    
    @classmethod
    def boolean(cls, b: bool) -> 'XValue':
        return cls(XType.BOOL, 0, 1 if b else 0)
    
    @classmethod
    def number(cls, n: float) -> 'XValue':
        # Pack float32 as uint32
        data = struct.unpack('I', struct.pack('f', n))[0]
        return cls(XType.NUMBER, 0, data)
    
    def is_nil(self) -> bool:
        return self.type == XType.NIL
    
    def is_truthy(self) -> bool:
        if self.type == XType.NIL:
            return False
        if self.type == XType.BOOL:
            return self.data != 0
        return True
    
    def is_falsy(self) -> bool:
        return not self.is_truthy()
    
    def as_number(self) -> float:
        # Unpack uint32 as float32
        return struct.unpack('f', struct.pack('I', self.data))[0]
    
    def as_bool(self) -> bool:
        return self.data != 0
    
    def __repr__(self):
        type_names = ['nil', 'bool', 'number', 'string', 'table', 'function']
        tname = type_names[self.type] if self.type < len(type_names) else f'type{self.type}'
        if self.type == XType.NIL:
            return 'nil'
        elif self.type == XType.BOOL:
            return 'true' if self.data else 'false'
        elif self.type == XType.NUMBER:
            return f'{self.as_number()}'
        else:
            return f'{tname}:{self.data}'


# =============================================================================
# Simple VM (CPU simulation)
# =============================================================================

class SimpleVM:
    MAX_STACK = 256
    
    def __init__(self):
        self.stack: List[XValue] = []
        self.sp = 0
        self.pc = 0
        self.status = VMStatus.RUNNING
        self.error = VMError.NONE
        
        self.bytecode: bytes = b''
        self.constants: List[XValue] = []
        self.globals: List[XValue] = [XValue.nil() for _ in range(256)]
    
    def push(self, v: XValue) -> bool:
        if len(self.stack) >= self.MAX_STACK:
            self.status = VMStatus.ERROR
            self.error = VMError.STACK_OVERFLOW
            return False
        self.stack.append(v)
        return True
    
    def pop(self) -> Optional[XValue]:
        if len(self.stack) == 0:
            self.status = VMStatus.ERROR
            self.error = VMError.STACK_UNDERFLOW
            return None
        return self.stack.pop()
    
    def peek(self, offset: int = 0) -> XValue:
        if len(self.stack) <= offset:
            return XValue.nil()
        return self.stack[-(1 + offset)]
    
    def read_byte(self) -> int:
        if self.pc >= len(self.bytecode):
            return 0
        b = self.bytecode[self.pc]
        self.pc += 1
        return b
    
    def read_uint16(self) -> int:
        lo = self.read_byte()
        hi = self.read_byte()
        return lo | (hi << 8)
    
    def read_int16(self) -> int:
        u = self.read_uint16()
        if u >= 0x8000:
            return u - 0x10000
        return u
    
    def step(self):
        if self.status != VMStatus.RUNNING:
            return
        
        opcode = self.read_byte()
        
        if opcode == Op.NOP:
            pass
        
        elif opcode == Op.PUSH_NIL:
            self.push(XValue.nil())
        
        elif opcode == Op.PUSH_TRUE:
            self.push(XValue.boolean(True))
        
        elif opcode == Op.PUSH_FALSE:
            self.push(XValue.boolean(False))
        
        elif opcode == Op.PUSH_NUM or opcode == Op.PUSH_STR:
            idx = self.read_uint16()
            if idx < len(self.constants):
                self.push(self.constants[idx])
            else:
                self.push(XValue.nil())
        
        elif opcode == Op.POP:
            self.pop()
        
        elif opcode == Op.DUP:
            v = self.peek(0)
            self.push(XValue(v.type, v.flags, v.data))
        
        elif opcode == Op.SWAP:
            if len(self.stack) >= 2:
                self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
        
        elif opcode == Op.GET_GLOBAL:
            idx = self.read_uint16()
            if idx < len(self.globals):
                self.push(self.globals[idx])
            else:
                self.push(XValue.nil())
        
        elif opcode == Op.SET_GLOBAL:
            idx = self.read_uint16()
            v = self.pop()
            if v and idx < len(self.globals):
                self.globals[idx] = v
        
        elif opcode == Op.ADD:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                self.push(XValue.number(a.as_number() + b.as_number()))
            else:
                self.push(XValue.nil())
        
        elif opcode == Op.SUB:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                self.push(XValue.number(a.as_number() - b.as_number()))
            else:
                self.push(XValue.nil())
        
        elif opcode == Op.MUL:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                self.push(XValue.number(a.as_number() * b.as_number()))
            else:
                self.push(XValue.nil())
        
        elif opcode == Op.DIV:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                if b.as_number() == 0:
                    self.push(XValue.number(float('inf')))
                else:
                    self.push(XValue.number(a.as_number() / b.as_number()))
            else:
                self.push(XValue.nil())
        
        elif opcode == Op.MOD:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                na, nb = a.as_number(), b.as_number()
                import math
                self.push(XValue.number(na - math.floor(na / nb) * nb))
            else:
                self.push(XValue.nil())
        
        elif opcode == Op.NEG:
            a = self.pop()
            if a and a.type == XType.NUMBER:
                self.push(XValue.number(-a.as_number()))
            else:
                self.push(XValue.nil())
        
        elif opcode == Op.POW:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                self.push(XValue.number(a.as_number() ** b.as_number()))
            else:
                self.push(XValue.nil())
        
        elif opcode == Op.EQ:
            b = self.pop()
            a = self.pop()
            if a and b:
                if a.type != b.type:
                    self.push(XValue.boolean(False))
                elif a.type == XType.NUMBER:
                    self.push(XValue.boolean(a.as_number() == b.as_number()))
                else:
                    self.push(XValue.boolean(a.data == b.data))
            else:
                self.push(XValue.boolean(False))
        
        elif opcode == Op.NE:
            b = self.pop()
            a = self.pop()
            if a and b:
                if a.type != b.type:
                    self.push(XValue.boolean(True))
                elif a.type == XType.NUMBER:
                    self.push(XValue.boolean(a.as_number() != b.as_number()))
                else:
                    self.push(XValue.boolean(a.data != b.data))
            else:
                self.push(XValue.boolean(True))
        
        elif opcode == Op.LT:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                self.push(XValue.boolean(a.as_number() < b.as_number()))
            else:
                self.push(XValue.boolean(False))
        
        elif opcode == Op.LE:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                self.push(XValue.boolean(a.as_number() <= b.as_number()))
            else:
                self.push(XValue.boolean(False))
        
        elif opcode == Op.GT:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                self.push(XValue.boolean(a.as_number() > b.as_number()))
            else:
                self.push(XValue.boolean(False))
        
        elif opcode == Op.GE:
            b = self.pop()
            a = self.pop()
            if a and b and a.type == XType.NUMBER and b.type == XType.NUMBER:
                self.push(XValue.boolean(a.as_number() >= b.as_number()))
            else:
                self.push(XValue.boolean(False))
        
        elif opcode == Op.NOT:
            a = self.pop()
            if a:
                self.push(XValue.boolean(a.is_falsy()))
            else:
                self.push(XValue.boolean(True))
        
        elif opcode == Op.JMP:
            offset = self.read_int16()
            self.pc += offset
        
        elif opcode == Op.JMP_IF:
            offset = self.read_int16()
            a = self.pop()
            if a and a.is_truthy():
                self.pc += offset
        
        elif opcode == Op.JMP_IF_NOT:
            offset = self.read_int16()
            a = self.pop()
            if a and a.is_falsy():
                self.pc += offset
        
        elif opcode == Op.HALT:
            self.status = VMStatus.COMPLETED
        
        else:
            self.status = VMStatus.ERROR
            self.error = VMError.INVALID_OPCODE
    
    def run(self, max_steps: int = 1000):
        for _ in range(max_steps):
            if self.status != VMStatus.RUNNING:
                break
            self.step()


# =============================================================================
# Test Cases
# =============================================================================

@dataclass
class TestResult:
    test_id: int
    name: str
    passed: bool
    expected: Any
    actual: Any
    message: str = ""


def run_test(name: str, test_id: int, bytecode: bytes, 
             constants: List[XValue], 
             expected_result: Any,
             check_fn=None) -> TestResult:
    """Run a single VM test."""
    vm = SimpleVM()
    vm.bytecode = bytecode
    vm.constants = constants
    
    vm.run(100)
    
    if vm.status == VMStatus.ERROR:
        return TestResult(
            test_id, name, False, expected_result, None,
            f"VM Error: {VMError(vm.error).name}"
        )
    
    if vm.status != VMStatus.COMPLETED:
        return TestResult(
            test_id, name, False, expected_result, None,
            f"VM did not complete, status: {VMStatus(vm.status).name}"
        )
    
    if len(vm.stack) == 0:
        return TestResult(
            test_id, name, False, expected_result, None,
            "Stack is empty after execution"
        )
    
    result = vm.stack[-1]
    
    if check_fn:
        passed, actual = check_fn(result)
    else:
        if isinstance(expected_result, float):
            actual = result.as_number()
            passed = abs(actual - expected_result) < 0.0001
        elif isinstance(expected_result, bool):
            actual = result.as_bool()
            passed = actual == expected_result
        elif expected_result is None:
            actual = result
            passed = result.is_nil()
        else:
            actual = result
            passed = False
    
    return TestResult(test_id, name, passed, expected_result, actual)


def run_all_tests() -> List[TestResult]:
    """Run all VM tests."""
    results = []
    
    # Test 1: Push nil
    results.append(run_test(
        "push_nil", 1,
        bytes([Op.PUSH_NIL, Op.HALT]),
        [],
        None
    ))
    
    # Test 2: Push true
    results.append(run_test(
        "push_true", 2,
        bytes([Op.PUSH_TRUE, Op.HALT]),
        [],
        True
    ))
    
    # Test 3: Push false
    results.append(run_test(
        "push_false", 3,
        bytes([Op.PUSH_FALSE, Op.HALT]),
        [],
        False
    ))
    
    # Test 4: Push number constant
    results.append(run_test(
        "push_number", 4,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.HALT]),
        [XValue.number(42.0)],
        42.0
    ))
    
    # Test 5: Push float constant
    results.append(run_test(
        "push_float", 5,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.HALT]),
        [XValue.number(3.14159)],
        3.14159
    ))
    
    # Test 6: Push negative number
    results.append(run_test(
        "push_negative", 6,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.HALT]),
        [XValue.number(-100.5)],
        -100.5
    ))
    
    # Test 10: Add two numbers (10 + 20 = 30)
    results.append(run_test(
        "add_integers", 10,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.ADD, Op.HALT]),
        [XValue.number(10.0), XValue.number(20.0)],
        30.0
    ))
    
    # Test 11: Add floats (1.5 + 2.5 = 4.0)
    results.append(run_test(
        "add_floats", 11,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.ADD, Op.HALT]),
        [XValue.number(1.5), XValue.number(2.5)],
        4.0
    ))
    
    # Test 12: Subtract (30 - 10 = 20)
    results.append(run_test(
        "subtract", 12,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.SUB, Op.HALT]),
        [XValue.number(30.0), XValue.number(10.0)],
        20.0
    ))
    
    # Test 13: Multiply (6 * 7 = 42)
    results.append(run_test(
        "multiply", 13,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.MUL, Op.HALT]),
        [XValue.number(6.0), XValue.number(7.0)],
        42.0
    ))
    
    # Test 14: Divide (20 / 4 = 5)
    results.append(run_test(
        "divide", 14,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.DIV, Op.HALT]),
        [XValue.number(20.0), XValue.number(4.0)],
        5.0
    ))
    
    # Test 15: Modulo (17 % 5 = 2)
    results.append(run_test(
        "modulo", 15,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.MOD, Op.HALT]),
        [XValue.number(17.0), XValue.number(5.0)],
        2.0
    ))
    
    # Test 16: Negate (-42)
    results.append(run_test(
        "negate", 16,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.NEG, Op.HALT]),
        [XValue.number(42.0)],
        -42.0
    ))
    
    # Test 17: Power (5^2 = 25)
    results.append(run_test(
        "power", 17,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.POW, Op.HALT]),
        [XValue.number(5.0), XValue.number(2.0)],
        25.0
    ))
    
    # Test 20: Complex expression (5 + 3) * 2 = 16
    results.append(run_test(
        "complex_expr", 20,
        bytes([
            Op.PUSH_NUM, 0x00, 0x00,  # push 5
            Op.PUSH_NUM, 0x01, 0x00,  # push 3
            Op.ADD,                    # 5 + 3 = 8
            Op.PUSH_NUM, 0x02, 0x00,  # push 2
            Op.MUL,                    # 8 * 2 = 16
            Op.HALT
        ]),
        [XValue.number(5.0), XValue.number(3.0), XValue.number(2.0)],
        16.0
    ))
    
    # Test 30: Equality (10 == 10)
    results.append(run_test(
        "equality_true", 30,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x00, 0x00, Op.EQ, Op.HALT]),
        [XValue.number(10.0)],
        True
    ))
    
    # Test 31: Equality (10 == 20) = false
    results.append(run_test(
        "equality_false", 31,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.EQ, Op.HALT]),
        [XValue.number(10.0), XValue.number(20.0)],
        False
    ))
    
    # Test 32: Less than (5 < 10)
    results.append(run_test(
        "less_than_true", 32,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.LT, Op.HALT]),
        [XValue.number(5.0), XValue.number(10.0)],
        True
    ))
    
    # Test 33: Less than (10 < 5) = false
    results.append(run_test(
        "less_than_false", 33,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.LT, Op.HALT]),
        [XValue.number(10.0), XValue.number(5.0)],
        False
    ))
    
    # Test 34: Less or equal (10 <= 10)
    results.append(run_test(
        "less_equal", 34,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x00, 0x00, Op.LE, Op.HALT]),
        [XValue.number(10.0)],
        True
    ))
    
    # Test 35: Not (not true = false)
    results.append(run_test(
        "not_true", 35,
        bytes([Op.PUSH_TRUE, Op.NOT, Op.HALT]),
        [],
        False
    ))
    
    # Test 36: Not (not false = true)
    results.append(run_test(
        "not_false", 36,
        bytes([Op.PUSH_FALSE, Op.NOT, Op.HALT]),
        [],
        True
    ))
    
    # Test 37: Not (not nil = true)
    results.append(run_test(
        "not_nil", 37,
        bytes([Op.PUSH_NIL, Op.NOT, Op.HALT]),
        [],
        True
    ))
    
    # Test 40: DUP
    results.append(run_test(
        "dup", 40,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.DUP, Op.ADD, Op.HALT]),
        [XValue.number(21.0)],
        42.0  # 21 + 21
    ))
    
    # Test 41: SWAP
    results.append(run_test(
        "swap", 41,
        bytes([Op.PUSH_NUM, 0x00, 0x00, Op.PUSH_NUM, 0x01, 0x00, Op.SWAP, Op.SUB, Op.HALT]),
        [XValue.number(10.0), XValue.number(3.0)],
        -7.0  # 3 - 10 (after swap)
    ))
    
    # Test 50: Set and get global
    def check_global_test(result):
        return result.as_number() == 100.0, result.as_number()
    
    results.append(run_test(
        "set_get_global", 50,
        bytes([
            Op.PUSH_NUM, 0x00, 0x00,  # push 100
            Op.SET_GLOBAL, 0x00, 0x00,  # global[0] = 100
            Op.GET_GLOBAL, 0x00, 0x00,  # push global[0]
            Op.HALT
        ]),
        [XValue.number(100.0)],
        100.0,
        check_global_test
    ))
    
    return results


def print_results(results: List[TestResult]):
    """Print test results in a formatted way."""
    print("\n" + "=" * 60)
    print("XScript VM Test Results (CPU Simulation)")
    print("=" * 60 + "\n")
    
    passed = 0
    failed = 0
    
    for r in results:
        status = "[PASS]" if r.passed else "[FAIL]"
        print(f"[{r.test_id:3d}] {r.name:25s} {status}")
        
        if not r.passed:
            print(f"      Expected: {r.expected}")
            print(f"      Actual:   {r.actual}")
            if r.message:
                print(f"      Message:  {r.message}")
        
        if r.passed:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 60)
    print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    print("-" * 60)
    
    return failed == 0


def main():
    """Main entry point."""
    print("Running XScript VM tests...")
    
    results = run_all_tests()
    success = print_results(results)
    
    if success:
        print("\n[OK] All tests passed!")
        return 0
    else:
        print("\n[ERROR] Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

