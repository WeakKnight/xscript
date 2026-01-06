"""
Unit tests for XScript Python API.

Tests for spawn, filter, dispatch methods.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import xscript as xs
from api.context import Context, Script
from api.types import XValue, XTable


class TestContextBasic(unittest.TestCase):
    """Test basic Context functionality."""
    
    def test_create_context_default(self):
        """Test creating context with default options."""
        ctx = Context()
        self.assertEqual(ctx.device, "cpu")
        self.assertFalse(ctx.debug)
    
    def test_create_context_cuda(self):
        """Test creating context with cuda device."""
        ctx = Context(device="cuda")
        self.assertEqual(ctx.device, "cuda")
    
    def test_compile_simple(self):
        """Test compiling simple expression."""
        ctx = Context()
        script = ctx.compile("var x = 10;")
        self.assertIsInstance(script, Script)
        self.assertIsNotNone(script.bytecode)


class TestSpawn(unittest.TestCase):
    """Test ctx.spawn() method."""
    
    def test_spawn_returns_entity_id(self):
        """Test that spawn returns an entity ID."""
        ctx = Context()
        entity_id = ctx.spawn({"x": 0, "y": 0})
        self.assertIsInstance(entity_id, int)
        self.assertGreaterEqual(entity_id, 0)
    
    def test_spawn_empty_table(self):
        """Test spawning entity with empty table."""
        ctx = Context()
        entity_id = ctx.spawn({})
        self.assertIsInstance(entity_id, int)
    
    def test_spawn_with_components(self):
        """Test spawning entity with component data."""
        ctx = Context()
        entity_id = ctx.spawn({
            "position": {"x": 10, "y": 20},
            "velocity": {"x": 1, "y": 2},
            "health": 100
        })
        self.assertIsInstance(entity_id, int)
    
    def test_spawn_multiple_entities(self):
        """Test spawning multiple entities returns unique IDs."""
        ctx = Context()
        ids = []
        for i in range(10):
            entity_id = ctx.spawn({"index": i})
            ids.append(entity_id)
        
        # All IDs should be unique
        self.assertEqual(len(ids), len(set(ids)))
    
    def test_spawn_stores_entity(self):
        """Test that spawned entity can be retrieved."""
        ctx = Context()
        entity_id = ctx.spawn({"name": "test", "value": 42})
        
        # Entity should exist in context
        entity = ctx.get_entity(entity_id)
        self.assertIsNotNone(entity)
    
    def test_spawn_nested_tables(self):
        """Test spawning entity with nested table structure."""
        ctx = Context()
        entity_id = ctx.spawn({
            "transform": {
                "position": {"x": 0, "y": 0, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}
            }
        })
        self.assertIsInstance(entity_id, int)


class TestFilter(unittest.TestCase):
    """Test ctx.filter() method."""
    
    def test_filter_returns_filter_object(self):
        """Test that filter returns a Filter object."""
        ctx = Context()
        f = ctx.filter("position", "velocity")
        self.assertIsNotNone(f)
        self.assertTrue(hasattr(f, 'keys'))
    
    def test_filter_single_key(self):
        """Test filter with single component key."""
        ctx = Context()
        f = ctx.filter("health")
        self.assertEqual(f.keys, ("health",))
    
    def test_filter_multiple_keys(self):
        """Test filter with multiple component keys."""
        ctx = Context()
        f = ctx.filter("position", "velocity", "health")
        self.assertEqual(f.keys, ("position", "velocity", "health"))
    
    def test_filter_empty_keys(self):
        """Test filter with no keys (matches all entities)."""
        ctx = Context()
        f = ctx.filter()
        self.assertEqual(f.keys, ())
    
    def test_filter_matches_entity(self):
        """Test that filter correctly matches entities."""
        ctx = Context()
        
        # Spawn entities with different components
        e1 = ctx.spawn({"position": {"x": 0}, "velocity": {"x": 1}})
        e2 = ctx.spawn({"position": {"x": 0}})  # No velocity
        e3 = ctx.spawn({"health": 100})  # No position or velocity
        
        # Create filter for position + velocity
        f = ctx.filter("position", "velocity")
        
        # Check which entities match
        self.assertTrue(f.matches(ctx.get_entity(e1)))
        self.assertFalse(f.matches(ctx.get_entity(e2)))
        self.assertFalse(f.matches(ctx.get_entity(e3)))


class TestDispatch(unittest.TestCase):
    """Test ctx.dispatch() method."""
    
    def test_dispatch_basic(self):
        """Test basic dispatch call."""
        ctx = Context()
        script = ctx.compile('''
            func update(entity, dt) {
                entity.x = entity.x + 1;
            }
        ''')
        
        ctx.spawn({"x": 0})
        ctx.spawn({"x": 10})
        
        # Should not raise
        ctx.dispatch(script, "update", ctx.filter("x"), dt=0.016)
    
    def test_dispatch_updates_entities(self):
        """Test that dispatch actually updates entity data."""
        ctx = Context()
        script = ctx.compile('''
            func move(entity, dt) {
                entity.x = entity.x + entity.vx * dt;
            }
        ''')
        
        e1 = ctx.spawn({"x": 0, "vx": 100})
        
        ctx.dispatch(script, "move", ctx.filter("x", "vx"), dt=1.0)
        
        # Entity x should now be 100
        entity = ctx.get_entity(e1)
        self.assertEqual(entity["x"], 100)
    
    def test_dispatch_respects_filter(self):
        """Test that dispatch only affects matching entities."""
        ctx = Context()
        script = ctx.compile('''
            func increment(entity, dt) {
                entity.value = entity.value + 1;
            }
        ''')
        
        e1 = ctx.spawn({"value": 0, "marker": True})
        e2 = ctx.spawn({"value": 0})  # No marker
        
        # Only dispatch to entities with marker
        ctx.dispatch(script, "increment", ctx.filter("value", "marker"), dt=0)
        
        self.assertEqual(ctx.get_entity(e1)["value"], 1)
        self.assertEqual(ctx.get_entity(e2)["value"], 0)
    
    def test_dispatch_with_dt(self):
        """Test that dt parameter is passed correctly."""
        ctx = Context()
        script = ctx.compile('''
            func scale(entity, dt) {
                entity.time = dt;
            }
        ''')
        
        e1 = ctx.spawn({"time": 0})
        ctx.dispatch(script, "scale", ctx.filter("time"), dt=0.5)
        
        self.assertEqual(ctx.get_entity(e1)["time"], 0.5)
    
    def test_dispatch_nonexistent_function(self):
        """Test dispatch with non-existent function name."""
        ctx = Context()
        script = ctx.compile("var x = 1;")
        ctx.spawn({"x": 0})
        
        with self.assertRaises(NameError):
            ctx.dispatch(script, "nonexistent", ctx.filter(), dt=0)
    
    def test_dispatch_returns_stats(self):
        """Test that dispatch returns execution statistics."""
        ctx = Context()
        script = ctx.compile('''
            func noop(entity, dt) { }
        ''')
        
        for i in range(5):
            ctx.spawn({"id": i})
        
        stats = ctx.dispatch(script, "noop", ctx.filter("id"), dt=0)
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats.processed, 5)
        self.assertEqual(stats.skipped, 0)


class TestGetEntity(unittest.TestCase):
    """Test ctx.get_entity() method."""
    
    def test_get_entity_returns_table(self):
        """Test that get_entity returns the entity table."""
        ctx = Context()
        entity_id = ctx.spawn({"x": 10, "y": 20})
        
        entity = ctx.get_entity(entity_id)
        self.assertIsNotNone(entity)
        self.assertEqual(entity["x"], 10)
        self.assertEqual(entity["y"], 20)
    
    def test_get_entity_invalid_id(self):
        """Test get_entity with invalid ID returns None."""
        ctx = Context()
        entity = ctx.get_entity(999999)
        self.assertIsNone(entity)
    
    def test_get_entity_after_modification(self):
        """Test that entity changes are reflected."""
        ctx = Context()
        entity_id = ctx.spawn({"counter": 0})
        
        # Modify via dispatch
        script = ctx.compile('''
            func inc(entity, dt) {
                entity.counter = entity.counter + 1;
            }
        ''')
        ctx.dispatch(script, "inc", ctx.filter("counter"), dt=0)
        
        entity = ctx.get_entity(entity_id)
        self.assertEqual(entity["counter"], 1)


class TestDestroyEntity(unittest.TestCase):
    """Test ctx.destroy() method."""
    
    def test_destroy_entity(self):
        """Test destroying an entity."""
        ctx = Context()
        entity_id = ctx.spawn({"x": 0})
        
        ctx.destroy(entity_id)
        
        # Entity should no longer be valid
        entity = ctx.get_entity(entity_id)
        self.assertIsNone(entity)
    
    def test_destroy_invalid_entity(self):
        """Test destroying non-existent entity doesn't crash."""
        ctx = Context()
        # Should not raise
        ctx.destroy(999999)


class TestEntityCount(unittest.TestCase):
    """Test ctx.entity_count() method."""
    
    def test_entity_count_empty(self):
        """Test entity count with no entities."""
        ctx = Context()
        self.assertEqual(ctx.entity_count(), 0)
    
    def test_entity_count_after_spawn(self):
        """Test entity count after spawning."""
        ctx = Context()
        ctx.spawn({})
        ctx.spawn({})
        ctx.spawn({})
        self.assertEqual(ctx.entity_count(), 3)
    
    def test_entity_count_after_destroy(self):
        """Test entity count after destroying."""
        ctx = Context()
        e1 = ctx.spawn({})
        e2 = ctx.spawn({})
        ctx.destroy(e1)
        self.assertEqual(ctx.entity_count(), 1)


if __name__ == "__main__":
    unittest.main()

