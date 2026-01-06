# XScript AI Agent Development Guide

Project context and development guidelines for AI-assisted development.

## Project Overview

XScript is a GPU-native scripting language for games with:

- **Stack VM on GPU**: Runs on compute shaders via Slang/HLSL
- **SIMT/ECS**: Native Entity-Component-System with parallel dispatch
- **32-bit Value System**: All values use 32-bit storage (float32, uint32)
- **Dynamic Typing**: Lua-like types (nil, bool, number, string, table, function)
- **Metatables**: Operator overloading via metamethods
- **Reference Counting GC**: Deterministic memory management
- **C-Style Syntax**: Familiar syntax for developers

## Slang And Slangpy Reference
Use SlangGuide.md and SlangPyGuide.md 

## Directory Structure

```
xscript/
├── runtime/              # Slang GPU Runtime
│   ├── value.slang       # XValue 32-bit type system
│   ├── heap.slang        # Heap memory allocator
│   ├── table.slang       # Hash table + metatables
│   ├── string.slang      # String pool
│   ├── gc.slang          # Reference counting GC
│   ├── ops.slang         # Opcode definitions + VMState
│   ├── vm.slang          # VM main loop
│   ├── entity.slang      # ECS entity pool
│   ├── spawn.slang       # GPU spawn buffer
│   ├── dispatch.slang    # System dispatch kernel
│   └── tests/            # Slang unit tests
│
├── compiler/             # Python Compiler
│   ├── tokens.py         # Token definitions
│   ├── lexer.py          # Lexer
│   ├── ast.py            # AST node definitions
│   ├── parser.py         # Parser
│   ├── bytecode.py       # Bytecode definitions
│   ├── codegen.py        # Code generator
│   └── errors.py         # Error handling
│
├── api/                  # Python API
│   ├── types.py          # XValue/XTable wrappers
│   ├── context.py        # Script execution context + GPU dispatch
│   └── interpreter.py    # CPU interpreter (debug)
│
├── tests/                # Python tests (pytest)
│   ├── test_compiler.py
│   ├── test_gpu_dispatch.py  # Main GPU dispatch tests
│   ├── test_integration.py
│   └── ...
│
├── demo.py               # Working GPU dispatch demo
└── examples/             # Example scripts (.xs)
```

## Core Data Structures

### XValue (32-bit)

```slang
struct XValue {
    uint type;    // Type tag (0-7)
    uint flags;   // Flags (GC marks, etc.)
    uint data;    // 32-bit data (value or pointer)
};
```

| Type | type | data storage |
|------|------|--------------|
| nil | 0 | 0 |
| bool | 1 | 0 or 1 |
| number | 2 | float32 bit pattern (asuint/asfloat) |
| string | 3 | String pool index |
| table | 4 | Heap word offset |
| function | 5 | Function index |

### Opcodes

| Range | Category | Examples |
|-------|----------|----------|
| 0x00-0x0F | Stack | NOP, PUSH_NIL, POP, DUP, SWAP |
| 0x10-0x1F | Variables | GET_LOCAL, SET_LOCAL, GET_GLOBAL |
| 0x20-0x2F | Arithmetic | ADD, SUB, MUL, DIV, MOD, POW |
| 0x30-0x3F | Comparison | EQ, NE, LT, LE, GT, GE, NOT |
| 0x40-0x4F | Control | JMP, JMP_IF, JMP_IF_NOT, LOOP |
| 0x50-0x5F | Functions | CALL, RETURN, CALL_HOST |
| 0x60-0x6F | Tables | NEW_TABLE, GET_TABLE, SET_TABLE |
| 0x70-0x7F | Metatables | GET_META, SET_META |
| 0x80-0x8F | ECS | SPAWN_ENTITY, DESTROY_ENTITY, GET_ENTITY |
| 0xFF | Special | HALT |

### Key GPU Buffers

| Buffer | Purpose |
|--------|---------|
| `g_bytecode` | Compiled bytecode (uint32 words) |
| `g_constants` | Constant pool (XValue array) |
| `g_functions` | Function descriptors |
| `g_heapMemory` | Shared heap for tables |
| `g_entityPool` | Entity slots (tablePtr, generation, flags) |
| `g_vmStates` | Per-thread VM state |
| `g_dispatchState` | Stats: processed, skipped, errors |

## Coding Standards

### Slang Code

1. **Naming**: Constants `TYPE_NIL`, Structs `XValue`, Functions `xvalue_add`
2. **32-bit Constraint**: Use `uint` and `float` only, no 64-bit types
3. **VM_STACK_SIZE**: Currently 32 (reduced for GPU resource limits)

### Python Code

1. **Type Hints**: All function parameters and returns
2. **Tests**: Use pytest, class `Test*`, method `test_*`

## Common Tasks

### Add New Opcode

1. Add constant in `runtime/ops.slang`
2. Add case in `runtime/vm.slang` `vm_step()`
3. Add to `compiler/bytecode.py` `OpCode` enum
4. Generate bytecode in `compiler/codegen.py`
5. Add CPU implementation in `api/interpreter.py`
6. Add tests

### Run Tests

```bash
# All Python tests
python -m pytest tests/ -v

# GPU dispatch tests only
python -m pytest tests/test_gpu_dispatch.py -v

# Quick demo
python demo.py
```

## Notes

1. **Heap Pointer**: Uses word offset (not byte offset) in XValue.data
2. **Entity Pool**: Entity 0 with tablePtr=0 is treated as invalid
3. **GPU Filtering**: Component filtering runs on GPU (`dispatch_entity_matches`)
4. **FunctionDescriptor Order**: [codeOffset, paramCount, localCount, upvalueCount, nameIndex]
5. **Buffer Sizes**: HeapAllocator state = 32 bytes, EntityPoolState = 32 bytes

## TODO

- [ ] String concatenation
- [ ] Table iterators (pairs/ipairs)

## Resources

- [Slang](https://shader-slang.org/)
- [SlangPy](https://github.com/shader-slang/slangpy)
- [Lua Reference](https://www.lua.org/manual/5.4/)
