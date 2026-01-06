# XScript Programming Language Design

> XScript - GPU-native scripting language for games

## Overview

XScript is a game scripting language with a stack-based bytecode VM running entirely on GPU compute shaders (via Slang/HLSL). It combines Lua-like flexibility with C-style syntax, optimized for massively parallel entity processing.

### Core Features

- **GPU Stack VM**: Bytecode interpreter in compute shaders
- **SIMT/ECS Model**: 1 thread = 1 entity, parallel dispatch
- **32-bit Value System**: All values use 32-bit storage
- **Dynamic Typing**: nil, bool, number, string, table, function
- **Metatables**: Operator overloading via metamethods
- **Reference Counting GC**: Atomic operations for GPU
- **GPU Spawning**: Create entities without CPU round-trip
- **Python API**: Host integration via SlangPy

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Layer (Python)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Script    │  │   XScript   │  │     Bytecode        │  │
│  │   Context   │──│   Compiler  │──│     Generator       │  │
│  │     API     │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Device Layer (Slang/GPU)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Per-Thread │  │   Shared    │  │   ECS Dispatch      │  │
│  │   VMState   │──│    Heap     │──│     Kernel          │  │
│  │             │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Core Architecture

### 1.1 SIMT/ECS Dispatch Model

The key innovation: GPU threads map directly to ECS entities.

| ECS Concept | XScript | GPU Execution |
|-------------|---------|---------------|
| Entity | Table | 1 thread per entity |
| Component | Key | Table field |
| System | Function | Parallel kernel dispatch |

```
┌─────────────────────────────────────────────────────┐
│  GPU Dispatch (1 thread = 1 entity)                 │
├─────────────────────────────────────────────────────┤
│  Thread 0    │  Thread 1    │  Thread 2    │ ...    │
│  VMState[0]  │  VMState[1]  │  VMState[2]  │        │
│  Entity 0    │  Entity 1    │  Entity 2    │        │
├─────────────────────────────────────────────────────┤
│  Shared: bytecode, constants, heap, entity pool     │
└─────────────────────────────────────────────────────┘
```

### 1.2 XValue (32-bit)

All values use a unified 32-bit representation:

```slang
struct XValue {
    uint type;       // Type tag (0-7)
    uint flags;      // GC flags
    uint data;       // 32-bit value or heap offset
};
```

| Type | ID | Data Storage |
|------|-----|--------------|
| nil | 0 | 0 |
| bool | 1 | 0 or 1 |
| number | 2 | float32 bit pattern (asuint/asfloat) |
| string | 3 | String pool index |
| table | 4 | Heap word offset |
| function | 5 | Function index |

### 1.3 VM State

Each GPU thread has isolated VM state:

```slang
static const uint VM_STACK_SIZE = 32;  // Optimized for GPU

struct VMState {
    XValue stack[VM_STACK_SIZE];
    uint sp;        // Stack pointer
    uint fp;        // Frame pointer
    uint pc;        // Program counter
    uint status;    // running, paused, error, completed
    uint error;     // Error code
};
```

### 1.4 Instruction Set

| Range | Category | Examples |
|-------|----------|----------|
| 0x00-0x0F | Stack | NOP, PUSH_NIL, POP, DUP, SWAP |
| 0x10-0x1F | Variables | GET_LOCAL, SET_LOCAL, GET_GLOBAL |
| 0x20-0x2F | Arithmetic | ADD, SUB, MUL, DIV, MOD, POW |
| 0x30-0x3F | Comparison | EQ, NE, LT, LE, GT, GE, NOT |
| 0x40-0x4F | Control | JMP, JMP_IF, JMP_IF_NOT, LOOP |
| 0x50-0x5F | Functions | CALL, RETURN, CALL_HOST |
| 0x60-0x6F | Tables | NEW_TABLE, GET_TABLE, SET_TABLE |
| 0x70-0x7F | Metatables | GET_META, SET_META, INVOKE_META |
| 0x80-0x8F | ECS | SPAWN_ENTITY, DESTROY_ENTITY, GET_ENTITY |
| 0xFF | Special | HALT |

---

## 2. ECS System

### 2.1 Entity Pool

```slang
struct EntitySlot {
    uint tablePtr;      // Table in heap (word offset)
    uint generation;    // Handle validation
    uint flags;         // ACTIVE, DESTROYED, PENDING_SPAWN
    uint reserved;
};

struct EntityPoolState {
    uint activeCount;
    uint highWaterMark;
    uint freeListHead;
    uint freeListCount;
    uint destroyedCount;
    // ...
};
```

Entity IDs pack index + generation for safe references:
- 20 bits: index (up to ~1M entities)
- 12 bits: generation (4096 reuse cycles)

### 2.2 Dispatch Configuration

```slang
struct DispatchConfig {
    uint functionIndex;      // System function to execute
    uint entityCount;        // Entities to process
    uint requiredKeyCount;   // Component filter count
    uint requiredKeys[8];    // Required component keys (string indices)
    float dt;                // Delta time
    uint flags;
};

struct DispatchState {
    uint processedCount;     // Entities processed
    uint skippedCount;       // Filtered out (missing components)
    uint errorCount;         // Runtime errors
    uint spawnCount;         // Entities spawned
    uint destroyCount;       // Entities destroyed
    uint status;
};
```

### 2.3 GPU-side Filtering

Component filtering runs on GPU:

```slang
bool dispatch_entity_matches(XValue entityTable, uint requiredKeyCount) {
    for (uint i = 0; i < requiredKeyCount; i++) {
        uint keyIndex = g_dispatchConfig[0].requiredKeys[i];
        XValue key = XValue::string(keyIndex);
        XValue value = table_get(entityTable, key);
        if (value.type == TYPE_NIL) {
            return false;
        }
    }
    return true;
}
```

### 2.4 GPU Spawning

Entities can be created during dispatch without CPU sync:

```slang
struct SpawnRequest {
    uint tablePtr;           // Spawned entity table
    uint sourceEntityId;     // Parent entity
    uint sourceThreadId;     // Thread that spawned
    uint status;             // PENDING, COMMITTED, FAILED
};
```

---

## 3. Memory Management

### 3.1 Heap Allocator

GPU-side pool allocator with atomic operations:

```slang
struct HeapAllocator {
    uint nextOffset;     // Next free offset
    uint maxSize;        // Total heap size
    uint allocCount;     // Allocation count
    uint freeCount;      // Free count
    // ...
};
```

All heap offsets are **word indices** (not byte offsets).

### 3.2 String Pool

Interned strings with hash-based deduplication:

```slang
struct StringHeader {
    uint hash;           // String hash
    uint length;         // Character count
    uint refCount;       // Reference count
    uint next;           // Hash chain
    // chars follow...
};
```

### 3.3 Reference Counting

Atomic reference counting for heap objects:

```slang
void xvalue_incref(inout XValue v) {
    if (v.type >= TYPE_STRING) {
        InterlockedAdd(g_heapMemory[v.data], 1);
    }
}

void xvalue_decref(inout XValue v) {
    if (v.type >= TYPE_STRING) {
        uint oldCount;
        InterlockedAdd(g_heapMemory[v.data], -1, oldCount);
        if (oldCount == 1) {
            // Free memory
        }
    }
}
```

---

## 4. Language Syntax

### 4.1 Keywords

```
var func if else for while do break continue return
nil true false and or not
setmetatable getmetatable
```

### 4.2 Basic Syntax

```c
// Variables
var x = 10;
var name = "player";

// Tables
var player = {
    name: "Hero",
    hp: 100,
    position: { x: 0, y: 0 }
};

// Functions
func add(a, b) {
    return a + b;
}

// Control flow
if (x > 5) {
    print("large");
} else {
    print("small");
}

for (var i = 0; i < 10; i += 1) {
    print(i);
}

// Table access
player.name = "Warrior";
player["level"] = 10;
```

### 4.3 Metatables

```c
var VectorMeta = {
    __add: func(a, b) {
        return { x: a.x + b.x, y: a.y + b.y };
    }
};

var v1 = { x: 1, y: 2 };
setmetatable(v1, VectorMeta);

var v2 = { x: 3, y: 4 };
setmetatable(v2, VectorMeta);

var v3 = v1 + v2;  // Uses __add
```

---

## 5. Python API

### 5.1 Basic Usage

```python
import xscript as xs

ctx = xs.Context(device="cuda")

# Compile system function
systems = ctx.compile('''
    func movement(entity, dt) {
        entity.position.x += entity.velocity.x * dt;
        entity.position.y += entity.velocity.y * dt;
    }
''')

# Spawn entities
for i in range(10000):
    ctx.spawn({
        "position": {"x": i, "y": 0},
        "velocity": {"x": 1, "y": 0.5}
    })

# Dispatch - all entities processed in parallel
stats = ctx.dispatch(
    systems, 
    "movement", 
    ctx.filter("position", "velocity"),
    dt=0.016
)

print(f"Processed: {stats.processed}, Skipped: {stats.skipped}")
```

### 5.2 GPU Spawning

```c
func weapon_fire(entity, dt) {
    if (entity.cooldown <= 0) {
        spawn_entity({
            position: entity.position,
            velocity: {x: 50, y: 0},
            damage: 10
        });
        entity.cooldown = 0.1;
    }
}
```

---

## 6. Project Structure

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
│   ├── lexer.py          # Lexer
│   ├── parser.py         # Parser
│   ├── ast.py            # AST nodes
│   ├── codegen.py        # Bytecode generator
│   └── bytecode.py       # Bytecode definitions
│
├── api/                  # Python API
│   ├── context.py        # Script context + GPU dispatch
│   ├── types.py          # XValue/XTable wrappers
│   └── interpreter.py    # CPU interpreter (debug)
│
├── tests/                # Python tests (pytest)
│   ├── test_gpu_dispatch.py
│   ├── test_compiler.py
│   └── ...
│
├── demo.py               # Working GPU dispatch demo
└── examples/             # Example scripts (.xs)
```

---

## 7. Key GPU Buffers

| Buffer | Type | Purpose |
|--------|------|---------|
| `g_bytecode` | `RWStructuredBuffer<uint>` | Compiled bytecode |
| `g_constants` | `RWStructuredBuffer<XValue>` | Constant pool |
| `g_functions` | `RWStructuredBuffer<FunctionDescriptor>` | Function table |
| `g_heapMemory` | `RWStructuredBuffer<uint>` | Shared heap |
| `g_heapState` | `RWStructuredBuffer<HeapAllocator>` | Heap state |
| `g_entityPool` | `RWStructuredBuffer<EntitySlot>` | Entity slots |
| `g_entityPoolState` | `RWStructuredBuffer<EntityPoolState>` | Pool state |
| `g_vmStates` | `RWStructuredBuffer<VMState>` | Per-thread VM |
| `g_dispatchConfig` | `RWStructuredBuffer<DispatchConfig>` | Dispatch config |
| `g_dispatchState` | `RWStructuredBuffer<DispatchState>` | Dispatch stats |
| `g_dispatchEntityList` | `RWStructuredBuffer<uint>` | Entity IDs |

---

## 8. Implementation Notes

### Critical Details

1. **Heap Pointers**: Use word offset (data * 4 = byte offset)
2. **Entity ID 0**: With tablePtr=0 is treated as invalid
3. **FunctionDescriptor Order**: [codeOffset, paramCount, localCount, upvalueCount, nameIndex]
4. **Buffer Sizes**: HeapAllocator = 32 bytes, EntityPoolState = 32 bytes
5. **VM_STACK_SIZE**: 32 (reduced from original design for GPU resource limits)

### GPU Constraints

| Challenge | Solution |
|-----------|----------|
| No dynamic malloc | Pre-allocated heap with atomic bump allocator |
| Per-thread limits | Small stack size (32), state in global buffers |
| Control flow divergence | Simple bytecode, avoid deep nesting |
| String handling | Interned string pool with indices |

---

## 9. TODO

- [ ] Closure and upvalue support
- [ ] String concatenation
- [ ] Table iterators (pairs/ipairs)
- [ ] Standard library (math, string, table)
- [ ] Debug mode with breakpoints

---

## Resources

- [Slang](https://shader-slang.org/)
- [SlangPy](https://github.com/shader-slang/slangpy)
- [Lua Reference](https://www.lua.org/manual/5.4/)
