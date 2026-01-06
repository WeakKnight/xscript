# XScript SIMT Programming Model

> Bridging GPU Compute Shaders with ECS Game Architecture

## 1. Introduction & Motivation

### 1.1 Beyond Traditional Script VMs on GPU

XScript's most significant advantage is that its runtime is built on **compute shaders**, which inherently operate under the **SIMT (Single Instruction Multiple Threads)** execution model. However, simply porting a traditional stack-based VM to run on the GPU provides minimal benefit — the real opportunity lies in embracing SIMT as a first-class programming paradigm.

Traditional game scripting approaches suffer from:

- **Sequential execution**: NPCs, particles, and game entities are updated one-by-one
- **Cache inefficiency**: Random access patterns when iterating heterogeneous objects
- **CPU bottlenecks**: Complex AI and physics logic saturate CPU cores

XScript addresses these by enabling **massively parallel script execution** where thousands of entities run the same logic simultaneously on GPU threads.

### 1.2 The Natural Alignment: SIMT Meets ECS

Modern game engines increasingly adopt the **Entity-Component-System (ECS)** architecture for its data-oriented design benefits. ECS naturally aligns with SIMT:

| ECS Concept | Description | SIMT Parallel Execution |
|-------------|-------------|-------------------------|
| **Entity** | A unique identifier for a game object | One GPU thread per entity |
| **Component** | Data attached to an entity (Position, Velocity, Health) | Contiguous memory access |
| **System** | Logic that operates on entities with specific components | Single GPU kernel dispatch |

This alignment is not coincidental — both paradigms favor:

1. **Data homogeneity**: Same data types grouped together
2. **Uniform control flow**: Same operations applied to many items
3. **Batch processing**: Operating on sets rather than individuals

### 1.3 XScript's ECS Mapping

XScript leverages its dynamic Table type to naturally express ECS concepts:

```
┌─────────────────────────────────────────────────────────────────┐
│                        XScript ECS Mapping                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Table  ═══════════════════════════════════►  Entity            │
│   (Hash table instance)                        (Game object ID)  │
│                                                                  │
│   Key    ═══════════════════════════════════►  Component         │
│   (Field name: "position", "velocity")         (Data type)       │
│                                                                  │
│   Dispatch ═════════════════════════════════►  System Tick       │
│   (Parallel function execution)                (Update loop)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Concepts

### 2.1 Table as Entity

In XScript, each **Table** instance represents a game entity. Unlike traditional ECS where entities are just integer IDs, XScript Tables carry their component data directly:

```c
// XScript: Creating entities with components
var enemy = {
    position: { x: 10.0, y: 0.0, z: 20.0 },
    velocity: { x: 0.0, y: 0.0, z: 1.0 },
    health: 100,
    damage: 15,
    state: "patrol"
};

var bullet = {
    position: { x: 0.0, y: 1.0, z: 0.0 },
    velocity: { x: 0.0, y: 0.0, z: 50.0 },
    damage: 25,
    lifetime: 2.0
};
```

Each Table is self-contained, carrying exactly the components it needs.

### 2.2 Key as Component

The **Key** (field name) of a Table acts as the component identifier. The presence or absence of a key determines which systems operate on that entity:

| Key (Component) | Type | Description |
|-----------------|------|-------------|
| `position` | Table `{x, y, z}` | Entity's world position |
| `velocity` | Table `{x, y, z}` | Movement vector |
| `health` | Number | Current hit points |
| `damage` | Number | Attack power |
| `ai_state` | String | Behavior state machine |
| `sprite` | Table | Rendering data |
| `collider` | Table | Physics collision shape |

Systems query for entities that possess specific combinations of keys:

```python
# Python Host: Query entities with both position and velocity
movable_entities = ctx.filter("position", "velocity")
```

### 2.3 Dispatch as System Tick

A **Dispatch** operation executes a script function across all filtered entities in parallel. Each dispatch corresponds to one "System tick" in ECS terminology:

```
┌──────────────────────────────────────────────────────────────────┐
│                     Single Dispatch Operation                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Filter Phase:  Select entities with required components         │
│        │                                                          │
│        ▼                                                          │
│   ┌─────────┬─────────┬─────────┬─────────┬─────────┐            │
│   │Entity 0 │Entity 1 │Entity 2 │Entity 3 │Entity N │  ...       │
│   └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘            │
│        │         │         │         │         │                  │
│        ▼         ▼         ▼         ▼         ▼                  │
│   ┌─────────────────────────────────────────────────┐            │
│   │              GPU Kernel Execution                │            │
│   │   Thread 0   Thread 1   Thread 2   Thread 3 ...  │            │
│   │      │          │          │          │          │            │
│   │      ▼          ▼          ▼          ▼          │            │
│   │   update()   update()   update()   update()      │            │
│   └─────────────────────────────────────────────────┘            │
│                                                                   │
│   Result: All entities updated in parallel (SIMT)                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. API Design

### 3.1 Entity Registration

XScript supports entity registration from **both Host (Python) and Device (GPU/XScript)** sides, enabling dynamic entity spawning during gameplay.

#### 3.1.1 Host-Side Registration (Python)

Register entities from Python before or between dispatches. Entities are identified by auto-generated IDs, not names — `name` is just another optional component:

```python
import xscript as xs
import numpy as np

ctx = xs.Context(device="cuda")

# Register entity - returns auto-generated entity ID
enemy1 = ctx.spawn({
    "position": {"x": 10.0, "y": 0.0, "z": 20.0},
    "velocity": {"x": 0.0, "y": 0.0, "z": 1.0},
    "health": 100,
    "ai_state": "patrol"
})

# Name is just a component like any other (optional)
enemy2 = ctx.spawn({
    "name": "Boss Enemy",  # Optional - just another component
    "position": {"x": -5.0, "y": 0.0, "z": 15.0},
    "velocity": {"x": 1.0, "y": 0.0, "z": 0.0},
    "health": 500,
    "ai_state": "idle",
    "boss_tag": {}  # Tag component
})

# Bulk spawn - returns list of entity IDs
enemy_ids = ctx.spawn_batch([
    {"position": {"x": i * 2.0, "y": 0.0, "z": 0.0}, "health": 100}
    for i in range(10000)
])

# Query by name component if needed
bosses = ctx.query(required=["name", "boss_tag"])
```

**Why no mandatory name?**
- Entity ID is the true identifier (auto-generated integer or handle)
- `name` is just another component — optional like any other
- Many entities don't need names (bullets, particles, debris)
- Consistent with ECS philosophy: entities are just IDs with attached components

#### 3.1.2 Device-Side Registration (GPU/XScript)

Entities can be spawned directly from XScript code running on GPU. This is essential for:
- Bullets fired by enemies
- Particle effects
- Enemy spawners
- Procedural generation
- Chain reactions (explosions spawning more explosions)

**Built-in spawn functions:**

```c
// XScript: Spawn a new entity on GPU
func fire_bullet(entity, dt) {
    if (entity.fire_cooldown <= 0) {
        // Create new bullet entity directly on GPU
        var bullet = spawn_entity({
            position: {
                x: entity.position.x,
                y: entity.position.y + 1.0,
                z: entity.position.z
            },
            velocity: {
                x: entity.aim_dir.x * 50.0,
                y: entity.aim_dir.y * 50.0,
                z: entity.aim_dir.z * 50.0
            },
            damage: entity.weapon_damage,
            lifetime: 3.0,
            owner: entity  // Reference to spawner
        });
        
        entity.fire_cooldown = entity.fire_rate;
    }
}

// Spawn multiple entities at once
func explode(entity) {
    var count = 20;
    for (var i = 0; i < count; i += 1) {
        var angle = (i / count) * 6.28318;  // 2*PI
        spawn_entity({
            position: entity.position,
            velocity: {
                x: cos(angle) * 10.0,
                y: 5.0,
                z: sin(angle) * 10.0
            },
            particle_type: "explosion",
            lifetime: 1.0
        });
    }
    
    // Mark self for destruction
    entity.destroyed = true;
}
```

#### 3.1.3 Spawn Buffer Architecture

GPU-side spawning uses a **Spawn Buffer** mechanism to handle parallel entity creation safely:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GPU Spawn Buffer Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   During Dispatch:                                                   │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  Thread 0    Thread 1    Thread 2    Thread 3    ...     │      │
│   │     │           │           │           │                │      │
│   │     ▼           ▼           ▼           ▼                │      │
│   │  spawn()     spawn()       (none)     spawn()            │      │
│   │     │           │                       │                │      │
│   │     ▼           ▼                       ▼                │      │
│   │  ┌─────────────────────────────────────────────────┐     │      │
│   │  │            Spawn Buffer (Atomic Append)         │     │      │
│   │  │  [Entity A] [Entity B] [Entity C] ...           │     │      │
│   │  └─────────────────────────────────────────────────┘     │      │
│   └──────────────────────────────────────────────────────────┘      │
│                              │                                       │
│                              ▼                                       │
│   After Dispatch (Commit Phase):                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  Spawn Buffer ──────► Entity Pool                        │      │
│   │  (New entities added to main entity storage)             │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key implementation details:**

```slang
// Slang: Spawn buffer structure
RWStructuredBuffer<XValue> g_spawnBuffer;      // Pending entities
RWStructuredBuffer<uint> g_spawnCount;          // Atomic counter

// Atomic spawn operation
uint spawn_entity_internal(XValue entity) {
    uint index;
    InterlockedAdd(g_spawnCount[0], 1, index);
    
    if (index < SPAWN_BUFFER_CAPACITY) {
        g_spawnBuffer[index] = entity;
        return index;
    }
    return INVALID_ENTITY;  // Buffer full
}
```

**Host-side commit:**

```python
def game_loop(dt):
    # Run systems
    ctx.dispatch(systems, "movement_update", movable, dt=dt)
    ctx.dispatch(systems, "weapon_update", armed, dt=dt)  # May spawn bullets
    ctx.dispatch(systems, "explosion_update", exploding)   # May spawn particles
    
    # Commit spawned entities to main pool
    # This integrates all GPU-spawned entities
    ctx.commit_spawns()
    
    # Cleanup destroyed entities
    ctx.remove_destroyed()
```

#### 3.1.4 Entity Destruction

Entities can be marked for destruction from both Host and Device:

```c
// XScript: Mark entity for destruction
func bullet_update(entity, dt) {
    entity.lifetime = entity.lifetime - dt;
    
    if (entity.lifetime <= 0) {
        destroy_entity(entity);  // Mark for removal
    }
}

// Or simply set a destruction flag
func on_death(entity) {
    entity.destroyed = true;
    
    // Spawn death effect before destruction
    spawn_entity({
        position: entity.position,
        particle_type: "death_effect",
        lifetime: 2.0
    });
}
```

```python
# Python: Process destroyed entities after dispatch
def cleanup_phase():
    destroyed = ctx.query(condition="destroyed == true")
    ctx.remove_entities(destroyed)
    
    # Or automatic cleanup
    ctx.remove_destroyed()  # Removes all entities with 'destroyed' component
```

### 3.2 Component Filtering

The `filter()` method selects entities based on their components:

```python
# Basic filter: entities with position AND velocity
movable = ctx.filter("position", "velocity")

# Filter with all required components
physics_entities = ctx.filter("position", "velocity", "collider", "mass")

# Advanced query with exclusions
active_enemies = ctx.query(
    required=["position", "velocity", "ai_state"],
    optional=["target"],
    excluded=["dead", "disabled"]
)
```

### 3.3 Dispatch Execution

The `dispatch()` method executes a system function on filtered entities:

```python
# Compile the movement system
movement_system = ctx.compile('''
    func update_movement(entity, dt) {
        entity.position.x = entity.position.x + entity.velocity.x * dt;
        entity.position.y = entity.position.y + entity.velocity.y * dt;
        entity.position.z = entity.position.z + entity.velocity.z * dt;
    }
''')

# Execute on all movable entities
movable = ctx.filter("position", "velocity")
ctx.dispatch(movement_system, "update_movement", movable, dt=0.016)
```

### 3.4 Complete System Loop

```python
import xscript as xs

ctx = xs.Context(device="cuda")

# Compile all systems
systems = ctx.compile_file("game_systems.xs")

def game_loop(dt):
    # Movement System
    movable = ctx.filter("position", "velocity")
    ctx.dispatch(systems, "movement_update", movable, dt=dt)
    
    # Gravity System
    falling = ctx.filter("position", "velocity", "gravity")
    ctx.dispatch(systems, "gravity_update", falling, dt=dt)
    
    # AI System
    ai_entities = ctx.filter("position", "ai_state", "target")
    ctx.dispatch(systems, "ai_update", ai_entities, dt=dt)
    
    # Health System
    damageable = ctx.filter("health", "damage_queue")
    ctx.dispatch(systems, "health_update", damageable)
    
    # Cleanup System
    dead = ctx.query(required=["health"], condition="health <= 0")
    ctx.dispatch(systems, "death_handler", dead)
```

---

## 4. Execution Model

### 4.1 GPU Thread Mapping

Each GPU thread is assigned to one entity. The thread ID maps directly to an entity index:

```slang
[shader("compute")]
[numthreads(64, 1, 1)]
void system_dispatch(
    uint3 threadId : SV_DispatchThreadID,
    uniform uint entityCount,
    uniform float dt
) {
    uint entityIndex = threadId.x;
    
    if (entityIndex >= entityCount) return;
    
    // Each thread operates on its assigned entity
    XValue entity = g_entityTables[entityIndex];
    
    // Execute the system function
    VMState state = create_vm_state(entity);
    vm_execute_system(state, entity, dt);
    
    // Write back modified entity
    g_entityTables[entityIndex] = entity;
}
```

### 4.2 Data Layout Strategy

For optimal GPU memory access, XScript supports two data layouts:

#### Array of Structures (AOS) - Default

Each entity is stored as a complete Table:

```
Memory Layout (AOS):
┌──────────────────────────────────────────────────────────┐
│ Entity 0: [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, hp] │
│ Entity 1: [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, hp] │
│ Entity 2: [pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, hp] │
│ ...                                                       │
└──────────────────────────────────────────────────────────┘

Pros: Simple entity access, good for complex per-entity logic
Cons: Poor memory coalescing when accessing single component
```

#### Structure of Arrays (SOA) - Optimized

Components are stored in separate contiguous arrays:

```
Memory Layout (SOA):
┌──────────────────────────────────────────────────────────┐
│ position.x: [e0.x, e1.x, e2.x, e3.x, ...]               │
│ position.y: [e0.y, e1.y, e2.y, e3.y, ...]               │
│ position.z: [e0.z, e1.z, e2.z, e3.z, ...]               │
│ velocity.x: [e0.x, e1.x, e2.x, e3.x, ...]               │
│ velocity.y: [e0.y, e1.y, e2.y, e3.y, ...]               │
│ velocity.z: [e0.z, e1.z, e2.z, e3.z, ...]               │
│ health:     [e0.hp, e1.hp, e2.hp, e3.hp, ...]           │
└──────────────────────────────────────────────────────────┘

Pros: Excellent memory coalescing, cache-friendly
Cons: More complex entity reconstruction
```

### 4.3 Archetype-Based Organization

Entities with identical component sets (same keys) are grouped into **Archetypes**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Archetype System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Archetype A: [position, velocity, health]                       │
│  ┌─────────┬─────────┬─────────┬─────────┐                      │
│  │Entity 0 │Entity 1 │Entity 2 │Entity 3 │  (Contiguous)        │
│  └─────────┴─────────┴─────────┴─────────┘                      │
│                                                                  │
│  Archetype B: [position, velocity, health, ai_state]            │
│  ┌─────────┬─────────┬─────────┐                                │
│  │Entity 4 │Entity 5 │Entity 6 │  (Contiguous)                  │
│  └─────────┴─────────┴─────────┘                                │
│                                                                  │
│  Archetype C: [position, sprite]                                │
│  ┌─────────┬─────────┐                                          │
│  │Entity 7 │Entity 8 │  (Contiguous)                            │
│  └─────────┴─────────┘                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Benefits:
- Entities in same archetype have identical memory layout
- Single dispatch can process entire archetype without branching
- Component addition/removal moves entity between archetypes

### 4.4 Handling Branch Divergence

SIMT execution suffers when threads take different code paths. XScript mitigates this through:

**1. Archetype Separation**: Entities with different components run in separate dispatches

```python
# Instead of one dispatch with branching:
# BAD: ctx.dispatch(systems, "update_all", all_entities)

# Use separate dispatches:
# GOOD: Minimal branch divergence within each dispatch
ctx.dispatch(systems, "update_enemies", enemies)
ctx.dispatch(systems, "update_bullets", bullets)
ctx.dispatch(systems, "update_particles", particles)
```

**2. State Machine Batching**: Group entities by state before dispatch

```python
# Group by AI state
idle_enemies = ctx.query(required=["ai_state"], condition="ai_state == 'idle'")
patrol_enemies = ctx.query(required=["ai_state"], condition="ai_state == 'patrol'")
chase_enemies = ctx.query(required=["ai_state"], condition="ai_state == 'chase'")

# Separate dispatches - uniform control flow within each
ctx.dispatch(systems, "ai_idle", idle_enemies, dt=dt)
ctx.dispatch(systems, "ai_patrol", patrol_enemies, dt=dt)
ctx.dispatch(systems, "ai_chase", chase_enemies, dt=dt)
```

---

## 5. Code Examples

### 5.1 Complete Movement System

**game_systems.xs**:
```c
// Movement System - updates position based on velocity
func movement_update(entity, dt) {
    var pos = entity.position;
    var vel = entity.velocity;
    
    pos.x = pos.x + vel.x * dt;
    pos.y = pos.y + vel.y * dt;
    pos.z = pos.z + vel.z * dt;
    
    // Boundary check
    if (pos.x < -100) { pos.x = -100; vel.x = 0; }
    if (pos.x > 100) { pos.x = 100; vel.x = 0; }
    if (pos.z < -100) { pos.z = -100; vel.z = 0; }
    if (pos.z > 100) { pos.z = 100; vel.z = 0; }
}

// Gravity System - applies gravitational acceleration
func gravity_update(entity, dt) {
    var vel = entity.velocity;
    var gravity = entity.gravity;
    
    vel.y = vel.y + gravity * dt;
    
    // Ground collision
    if (entity.position.y <= 0 and vel.y < 0) {
        entity.position.y = 0;
        vel.y = 0;
    }
}

// Health System - processes damage and healing
func health_update(entity) {
    var queue = entity.damage_queue;
    
    // Process all pending damage/healing
    for (var i = 0; i < #queue; i += 1) {
        entity.health = entity.health - queue[i];
    }
    
    // Clamp health
    if (entity.health < 0) {
        entity.health = 0;
    }
    if (entity.health > entity.max_health) {
        entity.health = entity.max_health;
    }
    
    // Clear the queue
    entity.damage_queue = {};
}
```

### 5.2 AI Behavior System

**ai_systems.xs**:
```c
// AI Patrol Behavior
func ai_patrol(entity, dt) {
    var pos = entity.position;
    var patrol = entity.patrol_data;
    var target = patrol.points[patrol.current_index];
    
    // Move towards current patrol point
    var dx = target.x - pos.x;
    var dz = target.z - pos.z;
    var dist = (dx * dx + dz * dz) ^ 0.5;
    
    if (dist > 0.5) {
        // Normalize and apply speed
        var speed = entity.move_speed * dt;
        entity.velocity.x = (dx / dist) * speed;
        entity.velocity.z = (dz / dist) * speed;
    } else {
        // Reached point, move to next
        patrol.current_index = (patrol.current_index + 1) % #patrol.points;
        entity.velocity.x = 0;
        entity.velocity.z = 0;
    }
}

// AI Chase Behavior
func ai_chase(entity, dt) {
    var pos = entity.position;
    var target_pos = entity.target.position;
    
    var dx = target_pos.x - pos.x;
    var dz = target_pos.z - pos.z;
    var dist = (dx * dx + dz * dz) ^ 0.5;
    
    if (dist > entity.attack_range) {
        // Chase target
        var speed = entity.move_speed * dt;
        entity.velocity.x = (dx / dist) * speed;
        entity.velocity.z = (dz / dist) * speed;
    } else {
        // In attack range - stop and attack
        entity.velocity.x = 0;
        entity.velocity.z = 0;
        entity.ai_state = "attack";
    }
}

// AI Attack Behavior  
func ai_attack(entity, dt) {
    entity.attack_timer = entity.attack_timer - dt;
    
    if (entity.attack_timer <= 0) {
        // Deal damage to target
        var target = entity.target;
        if (target != nil and target.health != nil) {
            // Queue damage on target
            if (target.damage_queue == nil) {
                target.damage_queue = {};
            }
            target.damage_queue[#target.damage_queue] = entity.damage;
        }
        
        // Reset attack timer
        entity.attack_timer = entity.attack_cooldown;
    }
}
```

### 5.3 Python Host Integration

**game.py**:
```python
import xscript as xs
import time

class GameWorld:
    def __init__(self):
        self.ctx = xs.Context(device="cuda")
        self.systems = self.ctx.compile_file("game_systems.xs")
        self.ai_systems = self.ctx.compile_file("ai_systems.xs")
        
    def spawn_enemy(self, x, z, patrol_points=None):
        """Spawn an enemy entity with AI components. Returns entity ID."""
        entity = {
            "position": {"x": x, "y": 0.0, "z": z},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "health": 100,
            "max_health": 100,
            "damage": 10,
            "move_speed": 5.0,
            "attack_range": 2.0,
            "attack_cooldown": 1.0,
            "attack_timer": 0.0,
            "ai_state": "patrol" if patrol_points else "idle",
            "target": None,
            "damage_queue": {},
            "enemy_tag": {}  # Tag for filtering
        }
        
        if patrol_points:
            entity["patrol_data"] = {
                "points": patrol_points,
                "current_index": 0
            }
        
        # spawn() returns auto-generated entity ID
        return self.ctx.spawn(entity)
        
    def spawn_bullet(self, x, y, z, vx, vy, vz, damage):
        """Spawn a projectile entity. Returns entity ID."""
        return self.ctx.spawn({
            "position": {"x": x, "y": y, "z": z},
            "velocity": {"x": vx, "y": vy, "z": vz},
            "damage": damage,
            "lifetime": 5.0,
            "bullet_tag": {}  # Tag for filtering
        })
        
    def update(self, dt):
        """Run all game systems for one frame."""
        
        # Physics Systems
        movable = self.ctx.filter("position", "velocity")
        self.ctx.dispatch(self.systems, "movement_update", movable, dt=dt)
        
        falling = self.ctx.filter("position", "velocity", "gravity")
        self.ctx.dispatch(self.systems, "gravity_update", falling, dt=dt)
        
        # AI Systems - grouped by state for minimal divergence
        patrol = self.ctx.query(
            required=["position", "velocity", "patrol_data"],
            condition="ai_state == 'patrol'"
        )
        self.ctx.dispatch(self.ai_systems, "ai_patrol", patrol, dt=dt)
        
        chase = self.ctx.query(
            required=["position", "velocity", "target"],
            condition="ai_state == 'chase'"
        )
        self.ctx.dispatch(self.ai_systems, "ai_chase", chase, dt=dt)
        
        attack = self.ctx.query(
            required=["target", "damage", "attack_timer"],
            condition="ai_state == 'attack'"
        )
        self.ctx.dispatch(self.ai_systems, "ai_attack", attack, dt=dt)
        
        # Health System
        damageable = self.ctx.filter("health", "damage_queue")
        self.ctx.dispatch(self.systems, "health_update", damageable)
        
        # Weapon System - may spawn bullets on GPU
        armed = self.ctx.filter("weapon", "aim_dir", "position")
        self.ctx.dispatch(self.systems, "weapon_update", armed, dt=dt)
        
        # Commit GPU-spawned entities (bullets, particles, etc.)
        self.ctx.commit_spawns()
        
        # Cleanup destroyed entities
        self.ctx.remove_destroyed()


# Usage Example
if __name__ == "__main__":
    world = GameWorld()
    
    # Spawn 10,000 enemies with patrol routes
    # No names needed - entities are identified by auto-generated IDs
    for i in range(10000):
        x = (i % 100) * 2.0 - 100.0
        z = (i // 100) * 2.0 - 100.0
        patrol = [
            {"x": x, "z": z},
            {"x": x + 10, "z": z},
            {"x": x + 10, "z": z + 10},
            {"x": x, "z": z + 10}
        ]
        entity_id = world.spawn_enemy(x, z, patrol)
        # entity_id is auto-generated, e.g., 0, 1, 2, ...
    
    # Or use batch spawn for better performance
    enemy_batch = [
        {
            "position": {"x": (i % 100) * 2.0, "y": 0.0, "z": (i // 100) * 2.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "health": 100,
            "enemy_tag": {}
        }
        for i in range(10000)
    ]
    entity_ids = world.ctx.spawn_batch(enemy_batch)
    
    # Game loop
    last_time = time.time()
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        world.update(dt)
        
        # Cap at 60 FPS
        time.sleep(max(0, 1/60 - dt))
```

---

## 6. Performance Considerations

### 6.1 Memory Coalescing

GPU memory access is most efficient when adjacent threads access adjacent memory locations:

```
Coalesced Access (GOOD):
Thread 0 reads entity[0].position.x
Thread 1 reads entity[1].position.x
Thread 2 reads entity[2].position.x
Thread 3 reads entity[3].position.x
→ Single memory transaction

Strided Access (BAD):
Thread 0 reads entity[0].position.x
Thread 1 reads entity[0].position.y
Thread 2 reads entity[0].position.z
Thread 3 reads entity[0].velocity.x
→ Multiple memory transactions
```

**Recommendation**: Use SOA layout for systems that access few components, AOS for systems accessing many components of each entity.

### 6.2 Warp Occupancy

Maximize GPU utilization by ensuring enough entities per dispatch:

| Entity Count | Warps (32 threads) | Recommendation |
|--------------|-------------------|-----------------|
| < 32 | < 1 | Consider CPU execution |
| 32 - 1024 | 1 - 32 | Good for simple systems |
| 1024 - 10000 | 32 - 312 | Optimal range |
| > 10000 | > 312 | Excellent GPU utilization |

### 6.3 System Ordering

Order systems to minimize GPU-CPU synchronization:

```python
# GOOD: Batch all GPU dispatches together
def update(dt):
    # All GPU work first
    ctx.dispatch(systems, "movement", movable, dt=dt)
    ctx.dispatch(systems, "physics", physics_ents, dt=dt)
    ctx.dispatch(systems, "ai", ai_ents, dt=dt)
    ctx.dispatch(systems, "health", health_ents)
    
    # Then read results back to CPU
    dead_entities = ctx.query(condition="health <= 0")
    for entity in dead_entities:
        spawn_death_effect(entity)  # CPU-side effect
```

### 6.4 Component Data Packing

Pack related components to improve cache locality:

```c
// Instead of separate components:
var entity = {
    position_x: 10.0,
    position_y: 0.0,
    position_z: 20.0,
    velocity_x: 1.0,
    velocity_y: 0.0,
    velocity_z: 0.0
};

// Use nested tables (packed access):
var entity = {
    position: { x: 10.0, y: 0.0, z: 20.0 },
    velocity: { x: 1.0, y: 0.0, z: 0.0 }
};
```

---

## 7. Advanced Features

### 7.1 Component Queries

Complex queries support flexible entity selection:

```python
# Entities with position AND (velocity OR acceleration)
dynamic_entities = ctx.query(
    required=["position"],
    any_of=["velocity", "acceleration"]
)

# Enemies that are alive and not stunned
active_enemies = ctx.query(
    required=["enemy_tag", "health", "ai_state"],
    excluded=["stunned", "dead"],
    condition="health > 0"
)

# Entities within a spatial region
nearby = ctx.query(
    required=["position"],
    condition="position.x > 0 and position.x < 100"
)
```

### 7.2 Entity Relationships

Handle entity references for targeting, parenting, etc.:

```c
// XScript: Entity references
func ai_chase(entity, dt) {
    var target = entity.target;  // Reference to another entity
    
    if (target == nil or target.health <= 0) {
        entity.target = nil;
        entity.ai_state = "patrol";
        return;
    }
    
    // Chase logic using target.position
    move_towards(entity, target.position, dt);
}
```

### 7.3 Event Queues

Deferred operations via component-based event queues:

```c
// XScript: Queue-based damage system
func apply_damage(entity, amount) {
    if (entity.damage_queue == nil) {
        entity.damage_queue = {};
    }
    entity.damage_queue[#entity.damage_queue] = amount;
}

func process_damage_queue(entity) {
    var queue = entity.damage_queue;
    var total = 0;
    
    for (var i = 0; i < #queue; i += 1) {
        total = total + queue[i];
    }
    
    entity.health = entity.health - total;
    entity.damage_queue = {};  // Clear queue
}
```

### 7.4 GPU-Side Entity Spawning

One of XScript's most powerful features is the ability to spawn entities directly from GPU code. This enables:

- **Zero-latency spawning**: No CPU round-trip for bullet creation
- **Massive parallel spawning**: Thousands of particles in one dispatch
- **Chain reactions**: Explosions spawning more explosions, all on GPU

#### Spawn Built-in Functions

| Function | Description |
|----------|-------------|
| `spawn_entity(table)` | Create new entity, returns entity reference |
| `spawn_entities(count, template)` | Batch spawn with template |
| `destroy_entity(entity)` | Mark entity for removal |
| `clone_entity(entity)` | Create a copy of an entity |

#### Example: Projectile System

```c
// XScript: Complete weapon system with GPU-side bullet spawning
func weapon_update(entity, dt) {
    // Cooldown
    if (entity.weapon.cooldown > 0) {
        entity.weapon.cooldown = entity.weapon.cooldown - dt;
        return;
    }
    
    // Check fire input
    if (entity.weapon.trigger_pressed) {
        // Spawn bullet directly on GPU
        var bullet = spawn_entity({
            position: {
                x: entity.position.x + entity.aim_dir.x * 0.5,
                y: entity.position.y + 1.5,
                z: entity.position.z + entity.aim_dir.z * 0.5
            },
            velocity: {
                x: entity.aim_dir.x * entity.weapon.bullet_speed,
                y: entity.aim_dir.y * entity.weapon.bullet_speed,
                z: entity.aim_dir.z * entity.weapon.bullet_speed
            },
            damage: entity.weapon.damage,
            lifetime: entity.weapon.range / entity.weapon.bullet_speed,
            owner_id: entity.id,
            bullet_tag: {}  // Tag for filtering
        });
        
        // Reset cooldown
        entity.weapon.cooldown = 1.0 / entity.weapon.fire_rate;
    }
}

// Bullet lifetime system
func bullet_update(entity, dt) {
    entity.lifetime = entity.lifetime - dt;
    
    if (entity.lifetime <= 0) {
        destroy_entity(entity);
    }
}
```

#### Example: Particle Explosion

```c
// XScript: GPU-side particle burst
func explosion_update(entity) {
    if (entity.exploded) {
        return;
    }
    
    // Spawn particle ring
    var particle_count = 32;
    for (var i = 0; i < particle_count; i += 1) {
        var angle = (i / particle_count) * 6.28318;
        var speed = 8.0 + random() * 4.0;
        
        spawn_entity({
            position: entity.position,
            velocity: {
                x: cos(angle) * speed,
                y: 2.0 + random() * 3.0,
                z: sin(angle) * speed
            },
            particle_tag: {},
            particle_type: "fire",
            lifetime: 0.5 + random() * 0.5,
            scale: 0.5 + random() * 0.5,
            color: { r: 1.0, g: 0.5, b: 0.1 }
        });
    }
    
    // Spawn smoke column
    for (var i = 0; i < 10; i += 1) {
        spawn_entity({
            position: {
                x: entity.position.x + (random() - 0.5) * 2.0,
                y: entity.position.y,
                z: entity.position.z + (random() - 0.5) * 2.0
            },
            velocity: { x: 0, y: 3.0 + random() * 2.0, z: 0 },
            particle_tag: {},
            particle_type: "smoke",
            lifetime: 1.0 + random() * 1.0,
            scale: 1.0 + random() * 1.0,
            color: { r: 0.3, g: 0.3, b: 0.3 }
        });
    }
    
    entity.exploded = true;
    destroy_entity(entity);
}
```

#### Example: Enemy Spawner

```c
// XScript: Spawner that creates enemies on GPU
func spawner_update(entity, dt) {
    entity.spawn_timer = entity.spawn_timer - dt;
    
    if (entity.spawn_timer <= 0 and entity.spawn_count < entity.max_spawns) {
        // Calculate spawn position
        var angle = random() * 6.28318;
        var radius = entity.spawn_radius;
        
        var new_enemy = spawn_entity({
            position: {
                x: entity.position.x + cos(angle) * radius,
                y: entity.position.y,
                z: entity.position.z + sin(angle) * radius
            },
            velocity: { x: 0, y: 0, z: 0 },
            health: entity.spawn_template.health,
            max_health: entity.spawn_template.health,
            damage: entity.spawn_template.damage,
            move_speed: entity.spawn_template.speed,
            ai_state: "idle",
            enemy_tag: {},
            spawner_id: entity.id  // Track parent spawner
        });
        
        entity.spawn_count = entity.spawn_count + 1;
        entity.spawn_timer = entity.spawn_interval;
    }
}
```

---

## 8. Best Practices

### 8.1 System Design Guidelines

1. **Keep systems focused**: Each system should do one thing well
2. **Minimize component access**: Access only the components you need
3. **Avoid cross-entity dependencies**: When possible, use event queues
4. **Group by behavior**: Separate dispatches for different AI states

### 8.2 Component Design Guidelines

1. **Use descriptive key names**: `position`, `velocity`, not `p`, `v`
2. **Keep components small**: Split large components into focused ones
3. **Use tags for filtering**: Empty tables as boolean markers (`enemy_tag: {}`)
4. **Prefer numbers over strings**: Faster comparison and less memory

### 8.3 GPU-Side Spawning Guidelines

1. **Batch spawns when possible**: Spawn multiple entities in one system rather than many single spawns
2. **Use spawn pools**: Pre-allocate spawn buffer capacity for expected maximum spawns
3. **Commit spawns strategically**: Call `commit_spawns()` once per frame, not after every dispatch
4. **Avoid spawn storms**: Limit maximum spawns per dispatch to prevent buffer overflow
5. **Track spawn counts**: Monitor `get_spawn_count()` to detect spawn-heavy frames

```c
// GOOD: Controlled spawning with limits
func explosion_update(entity) {
    var max_particles = 50;  // Limit per explosion
    var count = min(entity.intensity * 10, max_particles);
    
    for (var i = 0; i < count; i += 1) {
        spawn_entity({ /* particle data */ });
    }
}

// BAD: Unbounded spawning
func bad_explosion(entity) {
    for (var i = 0; i < entity.intensity * 100; i += 1) {
        spawn_entity({ /* may overflow buffer */ });
    }
}
```

### 8.4 Performance Checklist

- [ ] Are entities with same components grouped (archetype optimization)?
- [ ] Are dispatch calls batched to minimize CPU-GPU sync?
- [ ] Are AI states separated into different dispatches?
- [ ] Is component data packed for cache efficiency?
- [ ] Are there enough entities to saturate GPU (1000+)?
- [ ] Is `commit_spawns()` called only once per frame?
- [ ] Are spawn counts bounded to prevent buffer overflow?
- [ ] Are destroyed entities cleaned up each frame?

---

## 9. Future Directions

### 9.1 Planned Features

- **Spatial indexing**: Built-in spatial hash grid for efficient proximity queries
- **Automatic SOA conversion**: Compiler-optimized data layout based on access patterns
- **Cross-entity messaging**: Efficient GPU-side entity-to-entity communication
- **Hierarchical entities**: Parent-child relationships with transform propagation
- **Prefab system**: Reusable entity templates with GPU-side instantiation

### 9.2 Research Areas

- **Adaptive batching**: Dynamic grouping based on runtime patterns
- **Warp-level primitives**: Expose GPU warp operations to scripts
- **Persistent threads**: Long-running systems with minimal dispatch overhead
- **Spawn prediction**: Pre-allocate spawn buffer based on historical data
- **GPU-side archetype migration**: Change entity archetypes without host involvement

---

## Appendix A: API Reference Summary

### Context Methods (Host-Side / Python)

| Method | Description |
|--------|-------------|
| `spawn(table)` | Spawn entity, returns entity ID |
| `spawn_batch(tables)` | Bulk spawn entities, returns list of IDs |
| `destroy(entity_id)` | Mark entity for destruction |
| `filter(*components)` | Get entities with all specified components |
| `query(required, optional, excluded, condition)` | Advanced entity query |
| `dispatch(script, func, entities, **args)` | Execute system on entities |
| `commit_spawns()` | Integrate GPU-spawned entities into main pool |
| `remove_destroyed()` | Remove all entities marked for destruction |
| `get_spawn_count()` | Get number of entities spawned in last dispatch |
| `get_entity(entity_id)` | Get entity table by ID |

### Built-in Functions (Device-Side / XScript)

| Function | Description |
|----------|-------------|
| `spawn_entity(table)` | Create new entity on GPU, returns reference |
| `spawn_entities(count, template_func)` | Batch spawn with template function |
| `destroy_entity(entity)` | Mark entity for removal |
| `clone_entity(entity)` | Create a copy of existing entity |
| `has_component(entity, key)` | Check if entity has a component |
| `add_component(entity, key, value)` | Add component to entity |
| `remove_component(entity, key)` | Remove component from entity |

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `required` | List[str] | Components that must be present |
| `optional` | List[str] | Components that may be present |
| `excluded` | List[str] | Components that must not be present |
| `any_of` | List[str] | At least one of these must be present |
| `condition` | str | XScript expression filter |

---

## Appendix B: Terminology

| Term | Definition |
|------|------------|
| **SIMT** | Single Instruction Multiple Threads - GPU execution model |
| **ECS** | Entity-Component-System - data-oriented game architecture |
| **Entity** | A game object identifier (XScript Table) |
| **Component** | Data attached to entity (XScript Key) |
| **System** | Logic operating on entities (XScript Dispatch) |
| **Archetype** | Group of entities with identical component sets |
| **Warp** | Group of 32 GPU threads executing in lockstep |
| **Divergence** | When threads in a warp take different branches |
| **Coalescing** | Combining multiple memory accesses into one transaction |

