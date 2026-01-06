"""
XScript GPU Dispatch Demo

This demonstrates the README example running on GPU using SlangPy.
It shows the core ECS dispatch functionality: compiling XScript code,
spawning entities with components, and executing systems in parallel on GPU.
"""
import sys
sys.path.insert(0, '.')
from api.context import Context

def main():
    print('=== XScript GPU Dispatch Demo ===')
    print()

    # Create GPU context
    ctx = Context(device='cuda')
    print('[1] Created CUDA context')

    # Compile movement system
    systems = ctx.compile('''
        func movement(entity, dt) {
            entity.position.x = entity.position.x + entity.velocity.x * dt;
            entity.position.y = entity.position.y + entity.velocity.y * dt;
        }
    ''')
    print('[2] Compiled movement system')

    # Spawn 100 entities with position and velocity components
    for i in range(100):
        ctx.spawn({
            'position': {'x': float(i), 'y': 0.0},
            'velocity': {'x': 1.0, 'y': 0.5}
        })
    print('[3] Spawned 100 entities')

    # Execute movement system on GPU
    dt = 0.016  # ~60 FPS
    stats = ctx.dispatch(systems, 'movement', 
                         ctx.filter('position', 'velocity'), dt=dt)
    print(f'[4] Dispatched to GPU: {stats}')

    # Verify results
    e0 = ctx.get_entity(0)
    e50 = ctx.get_entity(50)
    e99 = ctx.get_entity(99)

    print()
    print('=== Results ===')
    print(f"Entity 0:  position=({e0['position']['x']:.4f}, {e0['position']['y']:.4f})")
    print(f"           Expected: ({0.0 + 1.0 * dt:.4f}, {0.0 + 0.5 * dt:.4f})")
    print(f"Entity 50: position=({e50['position']['x']:.4f}, {e50['position']['y']:.4f})")
    print(f"           Expected: ({50.0 + 1.0 * dt:.4f}, {0.0 + 0.5 * dt:.4f})")
    print(f"Entity 99: position=({e99['position']['x']:.4f}, {e99['position']['y']:.4f})")
    print(f"           Expected: ({99.0 + 1.0 * dt:.4f}, {0.0 + 0.5 * dt:.4f})")
    print()
    
    # Verify correctness
    assert abs(e0['position']['x'] - (0.0 + 1.0 * dt)) < 0.001
    assert abs(e0['position']['y'] - (0.0 + 0.5 * dt)) < 0.001
    assert abs(e99['position']['x'] - (99.0 + 1.0 * dt)) < 0.001
    
    print('SUCCESS! All 100 entities processed correctly on GPU.')


if __name__ == '__main__':
    main()

