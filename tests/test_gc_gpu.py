"""
XScript GC GPU Tests

This module runs the GC tests from runtime/tests/test_gc.slang on GPU.
Uses a simple passed/failed count format.

Requires: slangpy (GPU support)
"""

import struct
import pytest
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

# SlangPy is required for GPU tests
import slangpy

# =============================================================================
# Path Configuration
# =============================================================================

RUNTIME_TESTS_DIR = Path(__file__).parent.parent / "runtime" / "tests"


# =============================================================================
# GC Test Runner
# =============================================================================

class GCTestRunner:
    """Runner for test_gc.slang tests.
    
    This runner creates buffers for heap, GC state, and simple pass/fail results.
    Uses g_testResults[0] = passed count, g_testResults[1] = failed count format.
    """
    
    # Buffer sizes
    HEAP_SIZE = 16384           # Heap memory (uints)
    HEAP_STATE_SIZE = 8         # HeapAllocator struct (8 uints)
    GC_STATE_SIZE = 4           # GCState struct (4 uints)
    TEST_RESULTS_SIZE = 64      # Test results buffer (uints)
    
    def __init__(self):
        """Initialize the GC test runner."""
        self.slang_path = RUNTIME_TESTS_DIR / "test_gc.slang"
        if not self.slang_path.exists():
            raise FileNotFoundError(f"Slang test file not found: {self.slang_path}")
        
        # Create device
        self.device = slangpy.Device()
        
        # Create a session with include paths
        opts = slangpy.SlangCompilerOptions()
        opts.include_paths = [str(RUNTIME_TESTS_DIR)]
        self.session = self.device.create_slang_session(compiler_options=opts)
        
        # Module path without extension
        self.module_path = str(self.slang_path).replace('.slang', '')
        
        # Cache for compiled kernels
        self._kernel_cache: Dict[str, Any] = {}
        
        # Create buffers
        self._create_buffers()
    
    def _create_buffers(self):
        """Create GPU buffers."""
        # Heap memory buffer
        self._heap_memory_buffer = self.device.create_buffer(
            size=self.HEAP_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Heap state buffer
        self._heap_state_buffer = self.device.create_buffer(
            size=self.HEAP_STATE_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # GC state buffer
        self._gc_state_buffer = self.device.create_buffer(
            size=self.GC_STATE_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Test results buffer (simple: [0]=passed, [1]=failed)
        self._test_results_buffer = self.device.create_buffer(
            size=self.TEST_RESULTS_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Initialize buffers to zero
        self._reset_buffers()
    
    def _reset_buffers(self):
        """Reset all buffers before a test."""
        self._heap_memory_buffer.copy_from_numpy(np.zeros(self.HEAP_SIZE, dtype=np.uint32))
        self._heap_state_buffer.copy_from_numpy(np.zeros(self.HEAP_STATE_SIZE, dtype=np.uint32))
        self._gc_state_buffer.copy_from_numpy(np.zeros(self.GC_STATE_SIZE, dtype=np.uint32))
        self._test_results_buffer.copy_from_numpy(np.zeros(self.TEST_RESULTS_SIZE, dtype=np.uint32))
    
    def _get_kernel(self, kernel_name: str):
        """Get or create a compute kernel for the given entry point."""
        if kernel_name not in self._kernel_cache:
            # Load program with specific entry point
            program = self.session.load_program(
                self.module_path,
                [kernel_name]
            )
            
            # Create compute kernel
            desc = slangpy.ComputeKernelDesc()
            desc.program = program
            self._kernel_cache[kernel_name] = self.device.create_compute_kernel(desc)
        
        return self._kernel_cache[kernel_name]
    
    def run_kernel(self, kernel_name: str) -> Tuple[int, int]:
        """Run a compute kernel and return (passed, failed) counts."""
        kernel = self._get_kernel(kernel_name)
        
        # Dispatch with 1x1x1 threads (single invocation)
        kernel.dispatch(
            thread_count=slangpy.uint3(1, 1, 1),
            vars={
                "g_heapMemory": self._heap_memory_buffer,
                "g_heapState": self._heap_state_buffer,
                "g_gcState": self._gc_state_buffer,
                "g_testResults": self._test_results_buffer,
            }
        )
        
        # Read back results
        result_data = self._test_results_buffer.to_numpy()
        passed = int(result_data[0])
        failed = int(result_data[1])
        
        return passed, failed
    
    def run_test(self, kernel_name: str) -> None:
        """Run a single test kernel and assert it passes."""
        self._reset_buffers()
        passed, failed = self.run_kernel(kernel_name)
        
        if failed > 0:
            pytest.fail(f"GPU test '{kernel_name}' failed: {passed} passed, {failed} failed")
        elif passed == 0:
            pytest.fail(f"GPU test '{kernel_name}' reported no results")


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def gc_runner():
    """Create a GC test runner for the test session."""
    try:
        runner = GCTestRunner()
        return runner
    except Exception as e:
        pytest.skip(f"GPU not available: {e}")


# =============================================================================
# GC Tests
# =============================================================================

class TestGCReferenceCountingBasics:
    """Test reference counting basics."""
    
    def test_incref_decref(self, gc_runner):
        """Test incref and decref operations."""
        gc_runner.run_test("test_incref_decref")
    
    def test_decref_frees(self, gc_runner):
        """Test that decref frees objects when refcount reaches 0."""
        gc_runner.run_test("test_decref_frees")
    
    def test_multiple_refs(self, gc_runner):
        """Test objects survive with multiple references."""
        gc_runner.run_test("test_multiple_refs")


class TestGCValueAssignment:
    """Test value assignment with reference counting."""
    
    def test_xvalue_assign(self, gc_runner):
        """Test proper refcount on assignment."""
        gc_runner.run_test("test_xvalue_assign")
    
    def test_assign_same_value(self, gc_runner):
        """Test no-op when assigning same value."""
        gc_runner.run_test("test_assign_same_value")
    
    def test_assign_chain(self, gc_runner):
        """Test A=B, B=C refcount correctness."""
        gc_runner.run_test("test_assign_chain")


class TestGCTableCleanup:
    """Test table GC cleanup."""
    
    def test_metatable_cleanup(self, gc_runner):
        """Test metatable freed with table."""
        gc_runner.run_test("test_metatable_cleanup")


class TestGCWeakReferences:
    """Test weak references."""
    
    def test_weak_ref_no_incref(self, gc_runner):
        """Test weak refs don't increment count."""
        gc_runner.run_test("test_weak_ref_no_incref")
    
    def test_weak_strengthen(self, gc_runner):
        """Test strengthening weak ref."""
        gc_runner.run_test("test_weak_strengthen")
    
    def test_weak_after_free(self, gc_runner):
        """Test weak ref returns nil after target freed."""
        gc_runner.run_test("test_weak_after_free")


class TestGCWriteBarrier:
    """Test write barrier."""
    
    def test_write_barrier(self, gc_runner):
        """Test old/new value refcounts updated."""
        gc_runner.run_test("test_write_barrier")


class TestGCControl:
    """Test GC control functions."""
    
    def test_gc_enable_disable(self, gc_runner):
        """Test enable/disable works."""
        gc_runner.run_test("test_gc_enable_disable")
    
    def test_gc_stats(self, gc_runner):
        """Test statistics tracking."""
        gc_runner.run_test("test_gc_stats")


# =============================================================================
# Standalone runner
# =============================================================================

def main():
    """Run all GC tests and print results."""
    print("=" * 60)
    print("XScript GC GPU Tests")
    print("=" * 60)
    
    try:
        runner = GCTestRunner()
    except Exception as e:
        print(f"ERROR: Failed to initialize GPU: {e}")
        return 1
    
    tests = [
        "test_incref_decref",
        "test_decref_frees",
        "test_multiple_refs",
        "test_xvalue_assign",
        "test_assign_same_value",
        "test_assign_chain",
        "test_metatable_cleanup",
        "test_weak_ref_no_incref",
        "test_weak_strengthen",
        "test_weak_after_free",
        "test_write_barrier",
        "test_gc_enable_disable",
        "test_gc_stats",
    ]
    
    total_passed = 0
    total_failed = 0
    test_results = []
    
    for test_name in tests:
        try:
            runner._reset_buffers()
            passed, failed = runner.run_kernel(test_name)
            total_passed += passed
            total_failed += failed
            
            status = "PASS" if failed == 0 and passed > 0 else "FAIL"
            test_results.append((test_name, status, passed, failed))
            
            print(f"  {test_name}: {status} ({passed} passed, {failed} failed)")
            
        except Exception as e:
            test_results.append((test_name, "ERROR", 0, 0))
            print(f"  {test_name}: ERROR - {e}")
    
    print("-" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())

