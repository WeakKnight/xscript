# XScript

**XScript** is a GPU-accelerated scripting language designed for game development. It combines the flexibility of Lua with the performance of GPU compute shaders.

## Features

- **GPU-Powered VM**: Virtual machine runs on GPU via Slang/HLSL
- **Lua-Like Design**: Dynamic typing, metatables, and familiar semantics
- **C-Style Syntax**: Familiar syntax for most developers
- **Python API**: Easy integration via SlangPy
- **Reference Counting GC**: Deterministic memory management

## Quick Start

```python
import xscript as xs

# Create a context
ctx = xs.Context()

# Compile and execute script
script = ctx.compile('''
    var x = 10;
    var y = 20;
    return x + y;
''')

result = ctx.execute(script)
print(result)  # 30
```

## Language Examples

### Variables and Functions

```c
// Variable declaration
var x = 10;
var name = "player";

// Function definition
func add(a, b) {
    return a + b;
}

// Anonymous function
var multiply = func(a, b) {
    return a * b;
};
```

### Tables and Metatables

```c
// Create a table
var player = {
    name: "Hero",
    hp: 100,
    mp: 50
};

// Access fields
player.level = 10;
player["exp"] = 0;

// Metatable for operator overloading
var VectorMeta = {
    __add: func(a, b) {
        return { x: a.x + b.x, y: a.y + b.y };
    }
};
setmetatable(vector, VectorMeta);
```

### Control Flow

```c
// If/else
if (hp > 0) {
    print("Alive!");
} else {
    print("Dead!");
}

// For loop
for (var i = 0; i < 10; i += 1) {
    print(i);
}

// While loop
while (running) {
    update();
}
```

## Installation

```bash
pip install xscript
```

For GPU support:

```bash
pip install xscript[gpu]
```

## Project Structure

```
xscript/
├── compiler/           # Python compiler
│   ├── lexer.py        # Tokenization
│   ├── parser.py       # Parsing
│   └── codegen.py      # Bytecode generation
├── runtime/            # Slang VM
│   ├── vm.slang        # Virtual machine
│   ├── value.slang     # Type system
│   └── table.slang     # Table implementation
├── api/                # Python API
│   ├── context.py      # Script context
│   └── types.py        # Type wrappers
└── examples/           # Example scripts
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Python Host                         │
├─────────────────────────────────────────────────────┤
│  Context API  │  Compiler  │  Type Conversion       │
└───────────────┴────────────┴────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                GPU Runtime (Slang)                   │
├─────────────────────────────────────────────────────┤
│  Stack VM  │  Value System  │  GC  │  Tables        │
└────────────┴────────────────┴──────┴────────────────┘
```

## Host Function Registration

```python
ctx = xs.Context()

@ctx.register("spawn_enemy")
def spawn_enemy(enemy_type: str, x: float, y: float):
    # Python implementation
    return game.spawn(enemy_type, x, y)

# Now callable from script:
# spawn_enemy("goblin", 10.0, 20.0);
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines first.

## Acknowledgments

- [Slang](https://github.com/shader-slang/slang) - The shader language compiler
- [SlangPy](https://github.com/shader-slang/slangpy) - Python bindings for Slang
- Lua - Inspiration for the language design

