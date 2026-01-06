"""
Integration tests for XScript.

Tests end-to-end functionality including the README example.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import xscript as xs
from api.context import Context


class TestReadmeExample(unittest.TestCase):
    """Test the README example works end-to-end."""
    
    def test_readme_movement_system(self):
        """Test the movement system from README works."""
        ctx = xs.Context(device="cpu")
        
        # Compile system
        systems = ctx.compile('''
            func movement(entity, dt) {
                entity.position.x = entity.position.x + entity.velocity.x * dt;
                entity.position.y = entity.position.y + entity.velocity.y * dt;
            }
        ''')
        
        # Spawn entities
        entity_ids = []
        for i in range(100):
            eid = ctx.spawn({
                "position": {"x": float(i), "y": 0.0},
                "velocity": {"x": 1.0, "y": 0.5}
            })
            entity_ids.append(eid)
        
        # Verify initial state
        self.assertEqual(ctx.entity_count(), 100)
        e0 = ctx.get_entity(entity_ids[0])
        self.assertEqual(e0["position"]["x"], 0.0)
        
        # Execute dispatch
        dt = 0.016
        stats = ctx.dispatch(systems, "movement", 
                            ctx.filter("position", "velocity"), dt=dt)
        
        # Verify dispatch stats
        self.assertEqual(stats.processed, 100)
        self.assertEqual(stats.skipped, 0)
        
        # Verify entities were updated
        e0_after = ctx.get_entity(entity_ids[0])
        self.assertAlmostEqual(e0_after["position"]["x"], 0.0 + 1.0 * dt, places=5)
        self.assertAlmostEqual(e0_after["position"]["y"], 0.0 + 0.5 * dt, places=5)
        
        # Check another entity
        e50_after = ctx.get_entity(entity_ids[50])
        self.assertAlmostEqual(e50_after["position"]["x"], 50.0 + 1.0 * dt, places=5)
    
    def test_readme_with_10000_entities(self):
        """Test performance with 10,000 entities as in README."""
        ctx = xs.Context(device="cpu")
        
        systems = ctx.compile('''
            func movement(entity, dt) {
                entity.position.x = entity.position.x + entity.velocity.x * dt;
                entity.position.y = entity.position.y + entity.velocity.y * dt;
            }
        ''')
        
        # Spawn 10,000 entities
        for i in range(10000):
            ctx.spawn({
                "position": {"x": float(i), "y": 0.0},
                "velocity": {"x": 1.0, "y": 0.5}
            })
        
        self.assertEqual(ctx.entity_count(), 10000)
        
        # Execute
        stats = ctx.dispatch(systems, "movement", 
                            ctx.filter("position", "velocity"), dt=0.016)
        
        self.assertEqual(stats.processed, 10000)


class TestMultipleDispatches(unittest.TestCase):
    """Test multiple dispatch calls work correctly."""
    
    def test_multiple_dispatch_accumulation(self):
        """Test that multiple dispatches accumulate correctly."""
        ctx = Context()
        
        script = ctx.compile('''
            func increment(entity, dt) {
                entity.counter = entity.counter + 1;
            }
        ''')
        
        eid = ctx.spawn({"counter": 0})
        
        # Dispatch 5 times
        for _ in range(5):
            ctx.dispatch(script, "increment", ctx.filter("counter"), dt=0)
        
        entity = ctx.get_entity(eid)
        self.assertEqual(entity["counter"], 5)
    
    def test_different_systems_on_same_entity(self):
        """Test different systems can modify the same entity."""
        ctx = Context()
        
        physics = ctx.compile('''
            func apply_velocity(entity, dt) {
                entity.x = entity.x + entity.vx * dt;
            }
        ''')
        
        health = ctx.compile('''
            func apply_damage(entity, dt) {
                entity.hp = entity.hp - entity.damage_per_sec * dt;
            }
        ''')
        
        eid = ctx.spawn({
            "x": 0, "vx": 100,
            "hp": 100, "damage_per_sec": 10
        })
        
        # Run both systems
        ctx.dispatch(physics, "apply_velocity", ctx.filter("x", "vx"), dt=1.0)
        ctx.dispatch(health, "apply_damage", ctx.filter("hp", "damage_per_sec"), dt=1.0)
        
        entity = ctx.get_entity(eid)
        self.assertEqual(entity["x"], 100)
        self.assertEqual(entity["hp"], 90)


class TestFilteredDispatch(unittest.TestCase):
    """Test that filters correctly select entities."""
    
    def test_filter_by_single_component(self):
        """Test filtering by a single component."""
        ctx = Context()
        
        script = ctx.compile('''
            func mark(entity, dt) {
                entity.marked = 1;
            }
        ''')
        
        # Entity with marker component
        e1 = ctx.spawn({"marker": True, "marked": 0})
        # Entity without marker component
        e2 = ctx.spawn({"marked": 0})
        
        ctx.dispatch(script, "mark", ctx.filter("marker"), dt=0)
        
        self.assertEqual(ctx.get_entity(e1)["marked"], 1)
        self.assertEqual(ctx.get_entity(e2)["marked"], 0)
    
    def test_filter_by_multiple_components(self):
        """Test filtering by multiple components."""
        ctx = Context()
        
        script = ctx.compile('''
            func process(entity, dt) {
                entity.result = entity.a + entity.b;
            }
        ''')
        
        e1 = ctx.spawn({"a": 10, "b": 20, "result": 0})  # Has both
        e2 = ctx.spawn({"a": 5, "result": 0})  # Missing b
        e3 = ctx.spawn({"b": 15, "result": 0})  # Missing a
        
        ctx.dispatch(script, "process", ctx.filter("a", "b"), dt=0)
        
        self.assertEqual(ctx.get_entity(e1)["result"], 30)
        self.assertEqual(ctx.get_entity(e2)["result"], 0)
        self.assertEqual(ctx.get_entity(e3)["result"], 0)


class TestNestedTableAccess(unittest.TestCase):
    """Test accessing nested table properties."""
    
    def test_nested_read_write(self):
        """Test reading and writing nested properties."""
        ctx = Context()
        
        script = ctx.compile('''
            func move(entity, dt) {
                entity.transform.position.x = entity.transform.position.x + 10;
            }
        ''')
        
        eid = ctx.spawn({
            "transform": {
                "position": {"x": 0, "y": 0}
            }
        })
        
        ctx.dispatch(script, "move", ctx.filter("transform"), dt=0)
        
        entity = ctx.get_entity(eid)
        self.assertEqual(entity["transform"]["position"]["x"], 10)


class TestComplexScript(unittest.TestCase):
    """Test more complex script behaviors."""
    
    def test_conditional_logic(self):
        """Test if/else in system functions."""
        ctx = Context()
        
        script = ctx.compile('''
            func update_health(entity, dt) {
                if (entity.hp > 0) {
                    entity.alive = 1;
                } else {
                    entity.alive = 0;
                }
            }
        ''')
        
        e1 = ctx.spawn({"hp": 100, "alive": 0})
        e2 = ctx.spawn({"hp": 0, "alive": 1})
        
        ctx.dispatch(script, "update_health", ctx.filter("hp"), dt=0)
        
        self.assertEqual(ctx.get_entity(e1)["alive"], 1)
        self.assertEqual(ctx.get_entity(e2)["alive"], 0)
    
    def test_arithmetic_operations(self):
        """Test various arithmetic operations."""
        ctx = Context()
        
        script = ctx.compile('''
            func compute(entity, dt) {
                entity.sum = entity.a + entity.b;
                entity.diff = entity.a - entity.b;
                entity.prod = entity.a * entity.b;
                entity.quot = entity.a / entity.b;
            }
        ''')
        
        eid = ctx.spawn({
            "a": 10, "b": 2,
            "sum": 0, "diff": 0, "prod": 0, "quot": 0
        })
        
        ctx.dispatch(script, "compute", ctx.filter("a", "b"), dt=0)
        
        entity = ctx.get_entity(eid)
        self.assertEqual(entity["sum"], 12)
        self.assertEqual(entity["diff"], 8)
        self.assertEqual(entity["prod"], 20)
        self.assertEqual(entity["quot"], 5)


if __name__ == "__main__":
    unittest.main()

