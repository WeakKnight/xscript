"""
XScript Entity GPU Tests

This module runs the entity pool tests from runtime/tests/test_entity.slang on GPU.
Uses a simple passed/failed count format.

Requires: slangpy (GPU support)
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

import slangpy

# =============================================================================
# Path Configuration
# =============================================================================

RUNTIME_TESTS_DIR = Path(__file__).parent.parent / "runtime" / "tests"


# =============================================================================
# Entity Test Runner
# =============================================================================

class EntityTestRunner:
    """Runner for test_entity.slang tests.
    
    Creates buffers for entity pool, heap, and test results.
    Uses g_testResults[0] = passed count, g_testResults[1] = failed count format.
    """
    
    # Buffer sizes
    ENTITY_MAX_COUNT = 1024         # Entity pool slots
    FREE_LIST_SIZE = 1024           # Free list capacity
    DESTROY_LIST_SIZE = 256         # Destroy list capacity
    HEAP_SIZE = 16384               # Heap memory (uints)
    HEAP_STATE_SIZE = 8             # HeapAllocator struct (8 uints)
    TEST_RESULTS_SIZE = 64          # Test results buffer
    
    # EntitySlot struct: tablePtr, generation, flags, reserved = 4 uints = 16 bytes
    ENTITY_SLOT_SIZE = 4
    
    # EntityPoolState struct: 8 uints
    ENTITY_POOL_STATE_SIZE = 8
    
    def __init__(self):
        """Initialize the Entity test runner."""
        self.slang_path = RUNTIME_TESTS_DIR / "test_entity.slang"
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
        # Entity pool buffer (array of EntitySlot)
        self._entity_pool_buffer = self.device.create_buffer(
            size=self.ENTITY_MAX_COUNT * self.ENTITY_SLOT_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Entity pool state buffer
        self._entity_pool_state_buffer = self.device.create_buffer(
            size=self.ENTITY_POOL_STATE_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Free list buffer
        self._entity_free_list_buffer = self.device.create_buffer(
            size=self.FREE_LIST_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Destroy list buffer
        self._entity_destroy_list_buffer = self.device.create_buffer(
            size=self.DESTROY_LIST_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Destroy count buffer
        self._entity_destroy_count_buffer = self.device.create_buffer(
            size=4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
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
        
        # Test results buffer
        self._test_results_buffer = self.device.create_buffer(
            size=self.TEST_RESULTS_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        self._reset_buffers()
    
    def _reset_buffers(self):
        """Reset all buffers before a test."""
        self._entity_pool_buffer.copy_from_numpy(
            np.zeros(self.ENTITY_MAX_COUNT * self.ENTITY_SLOT_SIZE, dtype=np.uint32))
        self._entity_pool_state_buffer.copy_from_numpy(
            np.zeros(self.ENTITY_POOL_STATE_SIZE, dtype=np.uint32))
        self._entity_free_list_buffer.copy_from_numpy(
            np.zeros(self.FREE_LIST_SIZE, dtype=np.uint32))
        self._entity_destroy_list_buffer.copy_from_numpy(
            np.zeros(self.DESTROY_LIST_SIZE, dtype=np.uint32))
        self._entity_destroy_count_buffer.copy_from_numpy(
            np.zeros(1, dtype=np.uint32))
        self._heap_memory_buffer.copy_from_numpy(
            np.zeros(self.HEAP_SIZE, dtype=np.uint32))
        self._heap_state_buffer.copy_from_numpy(
            np.zeros(self.HEAP_STATE_SIZE, dtype=np.uint32))
        self._test_results_buffer.copy_from_numpy(
            np.zeros(self.TEST_RESULTS_SIZE, dtype=np.uint32))
    
    def _get_kernel(self, kernel_name: str):
        """Get or create a compute kernel for the given entry point."""
        if kernel_name not in self._kernel_cache:
            program = self.session.load_program(
                self.module_path,
                [kernel_name]
            )
            desc = slangpy.ComputeKernelDesc()
            desc.program = program
            self._kernel_cache[kernel_name] = self.device.create_compute_kernel(desc)
        
        return self._kernel_cache[kernel_name]
    
    def run_kernel(self, kernel_name: str) -> Tuple[int, int]:
        """Run a compute kernel and return (passed, failed) counts."""
        kernel = self._get_kernel(kernel_name)
        
        kernel.dispatch(
            thread_count=slangpy.uint3(1, 1, 1),
            vars={
                "g_entityPool": self._entity_pool_buffer,
                "g_entityPoolState": self._entity_pool_state_buffer,
                "g_entityFreeList": self._entity_free_list_buffer,
                "g_entityDestroyList": self._entity_destroy_list_buffer,
                "g_entityDestroyCount": self._entity_destroy_count_buffer,
                "g_heapMemory": self._heap_memory_buffer,
                "g_heapState": self._heap_state_buffer,
                "g_testResults": self._test_results_buffer,
            }
        )
        
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
def entity_runner():
    """Create an Entity test runner for the test session."""
    try:
        runner = EntityTestRunner()
        return runner
    except Exception as e:
        pytest.skip(f"GPU not available: {e}")


# =============================================================================
# Entity Tests
# =============================================================================

class TestEntityIDEncoding:
    """Test entity ID encoding/decoding."""
    
    def test_entity_id_encoding(self, entity_runner):
        """Test entity ID make/get index/generation."""
        entity_runner.run_test("test_entity_id_encoding")


class TestEntityPoolInit:
    """Test entity pool initialization."""
    
    def test_entity_pool_init(self, entity_runner):
        """Test pool initialization."""
        entity_runner.run_test("test_entity_pool_init")


class TestEntityCreate:
    """Test entity creation."""
    
    def test_entity_create(self, entity_runner):
        """Test creating entities."""
        entity_runner.run_test("test_entity_create")


class TestEntityValidation:
    """Test entity validation."""
    
    def test_entity_validation(self, entity_runner):
        """Test entity ID validation."""
        entity_runner.run_test("test_entity_validation")


class TestEntityDestroy:
    """Test entity destruction."""
    
    def test_entity_destroy(self, entity_runner):
        """Test destroying entities."""
        entity_runner.run_test("test_entity_destroy")


class TestEntityReuse:
    """Test entity slot reuse."""
    
    def test_entity_reuse(self, entity_runner):
        """Test entity reuse with generation."""
        entity_runner.run_test("test_entity_reuse")


class TestEntityMultipleOps:
    """Test multiple entity operations."""
    
    def test_entity_multiple_ops(self, entity_runner):
        """Test multiple entity operations."""
        entity_runner.run_test("test_entity_multiple_ops")


# =============================================================================
# Standalone runner
# =============================================================================

def main():
    """Run all entity tests and print results."""
    print("=" * 60)
    print("XScript Entity GPU Tests")
    print("=" * 60)
    
    try:
        runner = EntityTestRunner()
    except Exception as e:
        print(f"ERROR: Failed to initialize GPU: {e}")
        return 1
    
    tests = [
        "test_entity_id_encoding",
        "test_entity_pool_init",
        "test_entity_create",
        "test_entity_validation",
        "test_entity_destroy",
        "test_entity_reuse",
        "test_entity_multiple_ops",
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_name in tests:
        try:
            runner._reset_buffers()
            passed, failed = runner.run_kernel(test_name)
            total_passed += passed
            total_failed += failed
            
            status = "PASS" if failed == 0 and passed > 0 else "FAIL"
            print(f"  {test_name}: {status} ({passed} passed, {failed} failed)")
            
        except Exception as e:
            print(f"  {test_name}: ERROR - {e}")
    
    print("-" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())


