"""
GPU Dispatch Integration Tests

Tests the GPU dispatch path for XScript, comparing results with CPU execution.
Requires: slangpy (GPU support)
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import slangpy - skip all tests if not available
try:
    import slangpy
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

import xscript as xs
from api.context import Context


@pytest.mark.skipif(not HAS_GPU, reason="SlangPy not available")
class TestGPUDispatchBasic:
    """Basic GPU dispatch tests."""
    
    def test_gpu_context_creation(self):
        """Test creating a context with GPU device."""
        ctx = Context(device="cuda")
        assert ctx.device == "cuda"
    
    def test_gpu_init_on_dispatch(self):
        """Test that GPU is initialized on first dispatch."""
        ctx = Context(device="cuda", debug=True)
        
        script = ctx.compile('''
            func noop(entity, dt) { }
        ''')
        
        ctx.spawn({"x": 0})
        
        # This should trigger GPU initialization
        stats = ctx.dispatch(script, "noop", ctx.filter("x"), dt=0)
        
        assert ctx._initialized
        assert ctx._device is not None


@pytest.mark.skipif(not HAS_GPU, reason="SlangPy not available")
class TestGPUDispatchVsCPU:
    """Compare GPU and CPU dispatch results."""
    
    def test_simple_increment(self):
        """Test simple increment produces same result on CPU and GPU."""
        script_src = '''
            func increment(entity, dt) {
                entity.counter = entity.counter + 1;
            }
        '''
        
        # CPU execution
        ctx_cpu = Context(device="cpu")
        script_cpu = ctx_cpu.compile(script_src)
        ctx_cpu.spawn({"counter": 0})
        ctx_cpu.dispatch(script_cpu, "increment", ctx_cpu.filter("counter"), dt=0)
        cpu_result = ctx_cpu.get_entity(0)["counter"]
        
        # GPU execution
        ctx_gpu = Context(device="cuda")
        script_gpu = ctx_gpu.compile(script_src)
        ctx_gpu.spawn({"counter": 0})
        ctx_gpu.dispatch(script_gpu, "increment", ctx_gpu.filter("counter"), dt=0)
        gpu_result = ctx_gpu.get_entity(0)["counter"]
        
        assert cpu_result == gpu_result == 1
    
    def test_dt_parameter(self):
        """Test that dt parameter is correctly passed."""
        script_src = '''
            func set_dt(entity, dt) {
                entity.time = dt;
            }
        '''
        
        # CPU
        ctx_cpu = Context(device="cpu")
        script_cpu = ctx_cpu.compile(script_src)
        ctx_cpu.spawn({"time": 0})
        ctx_cpu.dispatch(script_cpu, "set_dt", ctx_cpu.filter("time"), dt=0.5)
        cpu_result = ctx_cpu.get_entity(0)["time"]
        
        # GPU
        ctx_gpu = Context(device="cuda")
        script_gpu = ctx_gpu.compile(script_src)
        ctx_gpu.spawn({"time": 0})
        ctx_gpu.dispatch(script_gpu, "set_dt", ctx_gpu.filter("time"), dt=0.5)
        gpu_result = ctx_gpu.get_entity(0)["time"]
        
        assert abs(cpu_result - 0.5) < 0.001
        assert abs(gpu_result - 0.5) < 0.001
    
    def test_arithmetic(self):
        """Test arithmetic operations on GPU."""
        script_src = '''
            func compute(entity, dt) {
                entity.result = entity.a + entity.b * 2;
            }
        '''
        
        # CPU
        ctx_cpu = Context(device="cpu")
        script_cpu = ctx_cpu.compile(script_src)
        ctx_cpu.spawn({"a": 10, "b": 5, "result": 0})
        ctx_cpu.dispatch(script_cpu, "compute", ctx_cpu.filter("a", "b"), dt=0)
        cpu_result = ctx_cpu.get_entity(0)["result"]
        
        # GPU
        ctx_gpu = Context(device="cuda")
        script_gpu = ctx_gpu.compile(script_src)
        ctx_gpu.spawn({"a": 10, "b": 5, "result": 0})
        ctx_gpu.dispatch(script_gpu, "compute", ctx_gpu.filter("a", "b"), dt=0)
        gpu_result = ctx_gpu.get_entity(0)["result"]
        
        # Expected: 10 + 5 * 2 = 20
        assert cpu_result == 20
        assert gpu_result == 20


@pytest.mark.skipif(not HAS_GPU, reason="SlangPy not available")
class TestGPUDispatchMultipleEntities:
    """Test GPU dispatch with multiple entities."""
    
    def test_10_entities(self):
        """Test dispatch with 10 entities."""
        ctx = Context(device="cuda")
        
        script = ctx.compile('''
            func increment(entity, dt) {
                entity.value = entity.value + 1;
            }
        ''')
        
        # Spawn 10 entities
        for i in range(10):
            ctx.spawn({"value": i})
        
        assert ctx.entity_count() == 10
        
        # Dispatch
        stats = ctx.dispatch(script, "increment", ctx.filter("value"), dt=0)
        
        assert stats.processed == 10
        
        # Verify each entity was incremented
        for i in range(10):
            entity = ctx.get_entity(i)
            assert entity["value"] == i + 1
    
    def test_100_entities(self):
        """Test dispatch with 100 entities."""
        ctx = Context(device="cuda")
        
        script = ctx.compile('''
            func double(entity, dt) {
                entity.x = entity.x * 2;
            }
        ''')
        
        # Spawn 100 entities
        for i in range(100):
            ctx.spawn({"x": float(i)})
        
        # Dispatch
        stats = ctx.dispatch(script, "double", ctx.filter("x"), dt=0)
        
        assert stats.processed == 100
        
        # Verify
        for i in range(100):
            entity = ctx.get_entity(i)
            assert entity["x"] == i * 2
    
    def test_1000_entities(self):
        """Test dispatch with 1000 entities."""
        ctx = Context(device="cuda")
        
        script = ctx.compile('''
            func add_velocity(entity, dt) {
                entity.x = entity.x + entity.vx * dt;
            }
        ''')
        
        # Spawn 1000 entities
        for i in range(1000):
            ctx.spawn({"x": 0.0, "vx": 1.0})
        
        # Dispatch with dt=1.0
        stats = ctx.dispatch(script, "add_velocity", ctx.filter("x", "vx"), dt=1.0)
        
        assert stats.processed == 1000
        
        # All entities should have x=1.0 now
        for i in range(0, 1000, 100):  # Sample every 100th
            entity = ctx.get_entity(i)
            assert abs(entity["x"] - 1.0) < 0.001


@pytest.mark.skipif(not HAS_GPU, reason="SlangPy not available")
class TestGPUDispatchFilter:
    """Test entity filtering in GPU dispatch."""
    
    def test_filter_by_component(self):
        """Test that only matching entities are processed."""
        ctx = Context(device="cuda")
        
        script = ctx.compile('''
            func mark(entity, dt) {
                entity.marked = 1;
            }
        ''')
        
        # Spawn entities with and without marker component
        ctx.spawn({"marker": True, "marked": 0})  # Should be processed
        ctx.spawn({"marked": 0})                   # Should be skipped
        ctx.spawn({"marker": True, "marked": 0})  # Should be processed
        
        stats = ctx.dispatch(script, "mark", ctx.filter("marker"), dt=0)
        
        assert stats.processed == 2
        assert stats.skipped == 1
        
        # Verify
        assert ctx.get_entity(0)["marked"] == 1
        assert ctx.get_entity(1)["marked"] == 0
        assert ctx.get_entity(2)["marked"] == 1


@pytest.mark.skipif(not HAS_GPU, reason="SlangPy not available")
class TestGPUDispatchMovementSystem:
    """Test the movement system from README."""
    
    def test_readme_example(self):
        """Test the exact README example works on GPU."""
        ctx = xs.Context(device="cuda")
        
        # Compile system
        systems = ctx.compile('''
            func movement(entity, dt) {
                entity.position.x = entity.position.x + entity.velocity.x * dt;
                entity.position.y = entity.position.y + entity.velocity.y * dt;
            }
        ''')
        
        # Spawn 100 entities
        for i in range(100):
            ctx.spawn({
                "position": {"x": float(i), "y": 0.0},
                "velocity": {"x": 1.0, "y": 0.5}
            })
        
        # Execute
        dt = 0.016
        stats = ctx.dispatch(systems, "movement", 
                            ctx.filter("position", "velocity"), dt=dt)
        
        # Verify
        assert stats.processed == 100
        
        # Check first entity
        e0 = ctx.get_entity(0)
        assert abs(e0["position"]["x"] - (0.0 + 1.0 * dt)) < 0.001
        assert abs(e0["position"]["y"] - (0.0 + 0.5 * dt)) < 0.001


@pytest.mark.skipif(not HAS_GPU, reason="SlangPy not available") 
class TestGPUDispatchStats:
    """Test dispatch statistics."""
    
    def test_stats_accuracy(self):
        """Test that stats accurately reflect processing."""
        ctx = Context(device="cuda")
        
        script = ctx.compile('''
            func process(entity, dt) {
                entity.processed = 1;
            }
        ''')
        
        # 5 matching, 3 not matching
        for i in range(5):
            ctx.spawn({"a": 1, "b": 2, "processed": 0})
        for i in range(3):
            ctx.spawn({"a": 1, "processed": 0})  # Missing b
        
        stats = ctx.dispatch(script, "process", ctx.filter("a", "b"), dt=0)
        
        assert stats.processed == 5
        assert stats.skipped == 3
        assert stats.errors == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

