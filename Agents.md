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

## Directory Structure

```
xscript/
├── runtime/              # Slang GPU Runtime
│   ├── value.slang       # XValue 32-bit type system
│   ├── heap.slang        # Heap memory allocator
│   ├── table.slang       # Hash table + metatables
│   ├── string.slang      # String pool
│   ├── gc.slang          # Reference counting GC
│   ├── ops.slang         # Opcode implementations
│   ├── vm.slang          # VM main loop
│   ├── entity.slang      # ECS entity pool
│   ├── spawn.slang       # GPU spawn buffer
│   ├── dispatch.slang    # System dispatch kernel
│   └── tests/            # Slang unit tests
│       ├── test_value.slang
│       ├── test_table.slang
│       ├── test_entity.slang
│       ├── test_spawn.slang
│       ├── test_dispatch.slang
│       └── ...
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
│   ├── context.py        # Script execution context
│   └── interpreter.py    # CPU interpreter (debug)
│
├── tests/                # Python tests
│   ├── test_compiler.py
│   ├── test_interpreter.py
│   └── test_vm_standalone.py
│
└── examples/             # Example scripts
    ├── hello_world.xs
    ├── game_npc.xs
    └── vector_math.xs
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
| table | 4 | Heap offset |
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
| 0x80-0x8F | ECS | SPAWN_ENTITY, DESTROY_ENTITY, GET_ENTITY, HAS_COMPONENT |
| 0xFF | Special | HALT |

### ECS Structures

```slang
struct EntitySlot {
    uint tablePtr;      // Entity table in heap
    uint generation;    // Handle validation
    uint flags;         // ACTIVE, DESTROYED
    uint reserved;
};

struct SpawnRequest {
    uint tablePtr;      // Spawned table
    uint sourceEntityId;
    uint sourceThreadId;
    uint status;        // PENDING, COMMITTED, FAILED
};

struct DispatchConfig {
    uint functionIndex;
    uint entityCount;
    uint requiredKeyCount;
    uint requiredKeys[8];
    float dt;
    uint flags;
};
```

## Coding Standards

### Slang Code

1. **File Organization**: One module per file, use `// =====` section dividers
2. **Naming**:
   - Constants: `TYPE_NIL`, `OP_ADD` (UPPER_SNAKE)
   - Structs: `XValue`, `VMState` (PascalCase)
   - Functions: `xvalue_add`, `vm_push` (snake_case)
3. **32-bit Constraint**: Use `uint` and `float` only, no 64-bit types
4. **Comments**: Use `//` single-line, document important functions

### Python Code

1. **Type Hints**: All function parameters and returns
2. **Docstrings**: Google-style for public APIs
3. **Tests**: Each module has corresponding `test_*.py`

### Tests

1. **Slang Tests**:
   - Use `RWStructuredBuffer<uint> g_testResults` for results
   - Each test has unique ID
   - Provide `test_*` entry points

2. **Python Tests**:
   - Use pytest framework
   - Class names: `Test*`
   - Method names: `test_*`

## Common Tasks

### Add New Opcode

1. Add constant in `runtime/ops.slang`
2. Add case in `runtime/vm.slang` `vm_step()`
3. Add to `compiler/bytecode.py` `OpCode` enum
4. Generate bytecode in `compiler/codegen.py`
5. Add CPU implementation in `api/interpreter.py`
6. Add tests

### Add ECS Opcode

1. Add constant in `runtime/ops.slang` (0x80-0x8F range)
2. Add ECS operation function in `runtime/ops.slang`
3. Add case in `runtime/vm.slang` `vm_step()`
4. Add tests in `runtime/tests/test_ops.slang`

### Add Builtin Function

1. Add in `api/context.py` `_register_builtins()`
2. Use `@self.register("name")` decorator
3. Add tests

### Add Metamethod

1. Add `META_*` constant in `runtime/table.slang`
2. Handle in `runtime/ops.slang`
3. Add Python support in `XTable` class

## Debug Tips

### View Generated Bytecode

```python
from compiler.codegen import CodeGenerator
from compiler.parser import Parser
from compiler.lexer import Lexer

source = "var x = 10 + 20;"
lexer = Lexer(source)
parser = Parser(lexer.tokenize())
ast = parser.parse()
codegen = CodeGenerator()
bytecode = codegen.generate(ast)
print(bytecode.disassemble())
```

### Run Tests

```bash
# Python tests
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_compiler.py -v
```

## TODO

- [ ] Closure and upvalue support
- [ ] String concatenation
- [ ] Table iterators (pairs/ipairs)
- [ ] SlangPy GPU integration
- [ ] Standard library

## Notes

1. **32-bit Float Precision**: Using float32, large values may lose precision
2. **GPU Memory**: Heap is fixed size, requires pre-allocation
3. **Reference Counting**: Watch for circular references
4. **Byte Order**: Bytecode uses little-endian

## Resources

- [Slang](https://shader-slang.org/)
- [SlangPy](https://github.com/shader-slang/slangpy)
- [Lua Reference](https://www.lua.org/manual/5.4/)
