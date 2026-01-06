# XScript

**GPU-native scripting language with SIMT/ECS for massively parallel game logic.**

## Core Advantage

XScript runs game scripts on **GPU compute shaders** using a **SIMT (Single Instruction Multiple Threads)** model that maps directly to **ECS (Entity-Component-System)** architecture:

| ECS | XScript | GPU Execution |
|-----|---------|---------------|
| Entity | Table | 1 thread per entity |
| Component | Key | Filter by keys |
| System | Dispatch | Parallel kernel |

**10,000 entities. One GPU dispatch. All processed in parallel.**

```python
import xscript as xs

ctx = xs.Context(device="cuda")

# Compile system
systems = ctx.compile('''
    func movement(entity, dt) {
        entity.position.x += entity.velocity.x * dt;
        entity.position.y += entity.velocity.y * dt;
    }
''')

# Spawn 10,000 entities
for i in range(10000):
    ctx.spawn({"position": {"x": i, "y": 0}, "velocity": {"x": 1, "y": 0.5}})

# Execute on ALL in parallel
ctx.dispatch(systems, "movement", ctx.filter("position", "velocity"), dt=0.016)
```

**GPU-side spawning** - no CPU round-trip:

```c
func weapon_fire(entity, dt) {
    if (entity.cooldown <= 0) {
        spawn_entity({
            position: entity.position,
            velocity: {x: entity.aim.x * 50, y: entity.aim.y * 50},
            damage: 10
        });
        entity.cooldown = 0.1;
    }
}
```

## Features

- **GPU VM** - Slang/HLSL compute shader runtime
- **SIMT/ECS** - Native parallel entity processing
- **GPU Spawning** - Create entities on GPU, no sync
- **Lua-like** - Dynamic types, metatables, C-style syntax
- **Python API** - Easy host integration

## License

MIT
