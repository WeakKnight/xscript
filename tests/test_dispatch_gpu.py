"""
XScript Dispatch GPU Tests

This module runs the dispatch tests from runtime/tests/test_dispatch.slang on GPU.
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
# Dispatch Test Runner
# =============================================================================

class DispatchTestRunner:
    """Runner for test_dispatch.slang tests.
    
    Creates buffers for dispatch config/state, entity pool, and test results.
    Uses g_testResults[0] = passed count, g_testResults[1] = failed count format.
    """
    
    # Buffer sizes
    DISPATCH_MAX_REQUIRED_KEYS = 8
    ENTITY_MAX_COUNT = 1024
    DISPATCH_ENTITY_LIST_SIZE = 1024
    DISPATCH_RESULTS_SIZE = 256
    TEST_RESULTS_SIZE = 256
    
    # DispatchConfig struct: functionIndex, entityCount, requiredKeyCount, 
    #   requiredKeys[8], dt, flags, padding0, padding1 = 15 uints
    DISPATCH_CONFIG_SIZE = 15
    
    # DispatchState struct: 8 uints
    DISPATCH_STATE_SIZE = 8
    
    # EntitySlot struct: 4 uints
    ENTITY_SLOT_SIZE = 4
    
    # EntityPoolState struct: 8 uints
    ENTITY_POOL_STATE_SIZE = 8
    
    def __init__(self):
        """Initialize the Dispatch test runner."""
        self.slang_path = RUNTIME_TESTS_DIR / "test_dispatch.slang"
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
        # Dispatch config buffer
        self._dispatch_config_buffer = self.device.create_buffer(
            size=self.DISPATCH_CONFIG_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Dispatch state buffer
        self._dispatch_state_buffer = self.device.create_buffer(
            size=self.DISPATCH_STATE_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Dispatch entity list
        self._dispatch_entity_list_buffer = self.device.create_buffer(
            size=self.DISPATCH_ENTITY_LIST_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Dispatch results buffer
        self._dispatch_results_buffer = self.device.create_buffer(
            size=self.DISPATCH_RESULTS_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Filtered entity count
        self._filtered_entity_count_buffer = self.device.create_buffer(
            size=4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Entity pool buffer
        self._entity_pool_buffer = self.device.create_buffer(
            size=self.ENTITY_MAX_COUNT * self.ENTITY_SLOT_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Entity pool state buffer
        self._entity_pool_state_buffer = self.device.create_buffer(
            size=self.ENTITY_POOL_STATE_SIZE * 4,
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
        self._dispatch_config_buffer.copy_from_numpy(
            np.zeros(self.DISPATCH_CONFIG_SIZE, dtype=np.uint32))
        self._dispatch_state_buffer.copy_from_numpy(
            np.zeros(self.DISPATCH_STATE_SIZE, dtype=np.uint32))
        self._dispatch_entity_list_buffer.copy_from_numpy(
            np.zeros(self.DISPATCH_ENTITY_LIST_SIZE, dtype=np.uint32))
        self._dispatch_results_buffer.copy_from_numpy(
            np.zeros(self.DISPATCH_RESULTS_SIZE, dtype=np.uint32))
        self._filtered_entity_count_buffer.copy_from_numpy(
            np.zeros(1, dtype=np.uint32))
        self._entity_pool_buffer.copy_from_numpy(
            np.zeros(self.ENTITY_MAX_COUNT * self.ENTITY_SLOT_SIZE, dtype=np.uint32))
        self._entity_pool_state_buffer.copy_from_numpy(
            np.zeros(self.ENTITY_POOL_STATE_SIZE, dtype=np.uint32))
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
    
    def run_kernel(self, kernel_name: str, thread_count: Tuple[int, int, int] = (1, 1, 1)) -> Tuple[int, int]:
        """Run a compute kernel and return (passed, failed) counts."""
        kernel = self._get_kernel(kernel_name)
        
        kernel.dispatch(
            thread_count=slangpy.uint3(thread_count[0], thread_count[1], thread_count[2]),
            vars={
                "g_dispatchConfig": self._dispatch_config_buffer,
                "g_dispatchState": self._dispatch_state_buffer,
                "g_dispatchEntityList": self._dispatch_entity_list_buffer,
                "g_dispatchResults": self._dispatch_results_buffer,
                "g_filteredEntityCount": self._filtered_entity_count_buffer,
                "g_entityPool": self._entity_pool_buffer,
                "g_entityPoolState": self._entity_pool_state_buffer,
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
    
    def run_parallel_test(self) -> None:
        """Run parallel dispatch test then verify."""
        self._reset_buffers()
        
        # Initialize dispatch state first
        self.run_kernel("test_dispatch_init")
        
        # Run parallel dispatch (64 threads)
        self.run_kernel("test_dispatch_parallel", (64, 1, 1))
        
        # Run verify
        passed, failed = self.run_kernel("test_dispatch_parallel_verify")
        
        if failed > 0:
            pytest.fail(f"GPU parallel dispatch test failed: {passed} passed, {failed} failed")
        elif passed == 0:
            pytest.fail(f"GPU parallel dispatch test reported no results")


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def dispatch_runner():
    """Create a Dispatch test runner for the test session."""
    try:
        runner = DispatchTestRunner()
        return runner
    except Exception as e:
        pytest.skip(f"GPU not available: {e}")


# =============================================================================
# Dispatch Tests
# =============================================================================

class TestDispatchInit:
    """Test dispatch initialization."""
    
    def test_dispatch_init(self, dispatch_runner):
        """Test dispatch state initialization."""
        dispatch_runner.run_test("test_dispatch_init")


class TestDispatchConfig:
    """Test dispatch configuration."""
    
    def test_dispatch_config(self, dispatch_runner):
        """Test dispatch configuration."""
        dispatch_runner.run_test("test_dispatch_config")


class TestDispatchNoFilter:
    """Test dispatch without filter."""
    
    def test_dispatch_no_filter(self, dispatch_runner):
        """Test entity matching without filter."""
        dispatch_runner.run_test("test_dispatch_no_filter")


class TestDispatchEntityList:
    """Test dispatch entity list."""
    
    def test_dispatch_entity_list(self, dispatch_runner):
        """Test building entity list."""
        dispatch_runner.run_test("test_dispatch_entity_list")


class TestDispatchCurrentEntity:
    """Test current entity management."""
    
    def test_dispatch_current_entity(self, dispatch_runner):
        """Test current entity get/set."""
        dispatch_runner.run_test("test_dispatch_current_entity")


class TestDispatchStatus:
    """Test dispatch status."""
    
    def test_dispatch_status(self, dispatch_runner):
        """Test dispatch status transitions."""
        dispatch_runner.run_test("test_dispatch_status")


class TestDispatchParallel:
    """Test parallel dispatch."""
    
    def test_dispatch_parallel(self, dispatch_runner):
        """Test parallel dispatch from 64 threads."""
        dispatch_runner.run_parallel_test()


# =============================================================================
# Standalone runner
# =============================================================================

def main():
    """Run all dispatch tests and print results."""
    print("=" * 60)
    print("XScript Dispatch GPU Tests")
    print("=" * 60)
    
    try:
        runner = DispatchTestRunner()
    except Exception as e:
        print(f"ERROR: Failed to initialize GPU: {e}")
        return 1
    
    tests = [
        "test_dispatch_init",
        "test_dispatch_config",
        "test_dispatch_no_filter",
        "test_dispatch_entity_list",
        "test_dispatch_current_entity",
        "test_dispatch_status",
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
    
    # Parallel test
    try:
        runner._reset_buffers()
        runner.run_kernel("test_dispatch_init")
        runner.run_kernel("test_dispatch_parallel", (64, 1, 1))
        passed, failed = runner.run_kernel("test_dispatch_parallel_verify")
        total_passed += passed
        total_failed += failed
        
        status = "PASS" if failed == 0 and passed > 0 else "FAIL"
        print(f"  test_dispatch_parallel: {status} ({passed} passed, {failed} failed)")
        
    except Exception as e:
        print(f"  test_dispatch_parallel: ERROR - {e}")
    
    print("-" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())

