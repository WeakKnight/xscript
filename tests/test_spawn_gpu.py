"""
XScript Spawn Buffer GPU Tests

This module runs the spawn buffer tests from runtime/tests/test_spawn.slang on GPU.
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
# Spawn Test Runner
# =============================================================================

class SpawnTestRunner:
    """Runner for test_spawn.slang tests.
    
    Creates buffers for spawn buffer, state, and test results.
    Uses g_testResults[0] = passed count, g_testResults[1] = failed count format.
    """
    
    # Buffer sizes
    SPAWN_BUFFER_CAPACITY = 1024    # Spawn requests
    SPAWNED_ENTITY_IDS_SIZE = 1024  # Spawned entity IDs
    TEST_RESULTS_SIZE = 256         # Test results buffer (larger for parallel tests)
    
    # SpawnRequest struct: tablePtr, sourceEntityId, sourceThreadId, status = 4 uints
    SPAWN_REQUEST_SIZE = 4
    
    # SpawnBufferState struct: 4 uints
    SPAWN_BUFFER_STATE_SIZE = 4
    
    def __init__(self):
        """Initialize the Spawn test runner."""
        self.slang_path = RUNTIME_TESTS_DIR / "test_spawn.slang"
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
        # Spawn buffer (array of SpawnRequest)
        self._spawn_buffer = self.device.create_buffer(
            size=self.SPAWN_BUFFER_CAPACITY * self.SPAWN_REQUEST_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Spawn buffer state
        self._spawn_buffer_state = self.device.create_buffer(
            size=self.SPAWN_BUFFER_STATE_SIZE * 4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Spawn count (atomic counter)
        self._spawn_count = self.device.create_buffer(
            size=4,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Spawned entity IDs
        self._spawned_entity_ids = self.device.create_buffer(
            size=self.SPAWNED_ENTITY_IDS_SIZE * 4,
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
        self._spawn_buffer.copy_from_numpy(
            np.zeros(self.SPAWN_BUFFER_CAPACITY * self.SPAWN_REQUEST_SIZE, dtype=np.uint32))
        self._spawn_buffer_state.copy_from_numpy(
            np.zeros(self.SPAWN_BUFFER_STATE_SIZE, dtype=np.uint32))
        self._spawn_count.copy_from_numpy(
            np.zeros(1, dtype=np.uint32))
        self._spawned_entity_ids.copy_from_numpy(
            np.zeros(self.SPAWNED_ENTITY_IDS_SIZE, dtype=np.uint32))
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
                "g_spawnBuffer": self._spawn_buffer,
                "g_spawnBufferState": self._spawn_buffer_state,
                "g_spawnCount": self._spawn_count,
                "g_spawnedEntityIds": self._spawned_entity_ids,
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
        """Run parallel spawn test (64 threads) then verify."""
        self._reset_buffers()
        
        # Run parallel spawn (64 threads)
        self.run_kernel("test_spawn_parallel", (64, 1, 1))
        
        # Run verify
        passed, failed = self.run_kernel("test_spawn_parallel_verify")
        
        if failed > 0:
            pytest.fail(f"GPU parallel spawn test failed: {passed} passed, {failed} failed")
        elif passed == 0:
            pytest.fail(f"GPU parallel spawn test reported no results")


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def spawn_runner():
    """Create a Spawn test runner for the test session."""
    try:
        runner = SpawnTestRunner()
        return runner
    except Exception as e:
        pytest.skip(f"GPU not available: {e}")


# =============================================================================
# Spawn Tests
# =============================================================================

class TestSpawnBufferInit:
    """Test spawn buffer initialization."""
    
    def test_spawn_buffer_init(self, spawn_runner):
        """Test buffer initialization."""
        spawn_runner.run_test("test_spawn_buffer_init")


class TestSpawnSingle:
    """Test single spawn."""
    
    def test_spawn_single(self, spawn_runner):
        """Test spawning a single entity."""
        spawn_runner.run_test("test_spawn_single")


class TestSpawnMultiple:
    """Test multiple spawns."""
    
    def test_spawn_multiple(self, spawn_runner):
        """Test spawning multiple entities."""
        spawn_runner.run_test("test_spawn_multiple")


class TestSpawnWithSource:
    """Test spawn with source tracking."""
    
    def test_spawn_with_source(self, spawn_runner):
        """Test spawning with source entity tracking."""
        spawn_runner.run_test("test_spawn_with_source")


class TestSpawnCommit:
    """Test spawn commit."""
    
    def test_spawn_commit(self, spawn_runner):
        """Test committing spawns."""
        spawn_runner.run_test("test_spawn_commit")


class TestSpawnBufferReset:
    """Test spawn buffer reset."""
    
    def test_spawn_buffer_reset(self, spawn_runner):
        """Test resetting spawn buffer."""
        spawn_runner.run_test("test_spawn_buffer_reset")


class TestSpawnParallel:
    """Test parallel spawns."""
    
    def test_spawn_parallel(self, spawn_runner):
        """Test parallel spawning from 64 threads."""
        spawn_runner.run_parallel_test()


# =============================================================================
# Standalone runner
# =============================================================================

def main():
    """Run all spawn tests and print results."""
    print("=" * 60)
    print("XScript Spawn Buffer GPU Tests")
    print("=" * 60)
    
    try:
        runner = SpawnTestRunner()
    except Exception as e:
        print(f"ERROR: Failed to initialize GPU: {e}")
        return 1
    
    tests = [
        ("test_spawn_buffer_init", (1, 1, 1)),
        ("test_spawn_single", (1, 1, 1)),
        ("test_spawn_multiple", (1, 1, 1)),
        ("test_spawn_with_source", (1, 1, 1)),
        ("test_spawn_commit", (1, 1, 1)),
        ("test_spawn_buffer_reset", (1, 1, 1)),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_name, threads in tests:
        try:
            runner._reset_buffers()
            passed, failed = runner.run_kernel(test_name, threads)
            total_passed += passed
            total_failed += failed
            
            status = "PASS" if failed == 0 and passed > 0 else "FAIL"
            print(f"  {test_name}: {status} ({passed} passed, {failed} failed)")
            
        except Exception as e:
            print(f"  {test_name}: ERROR - {e}")
    
    # Parallel test
    try:
        runner._reset_buffers()
        runner.run_kernel("test_spawn_parallel", (64, 1, 1))
        passed, failed = runner.run_kernel("test_spawn_parallel_verify")
        total_passed += passed
        total_failed += failed
        
        status = "PASS" if failed == 0 and passed > 0 else "FAIL"
        print(f"  test_spawn_parallel: {status} ({passed} passed, {failed} failed)")
        
    except Exception as e:
        print(f"  test_spawn_parallel: ERROR - {e}")
    
    print("-" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit(main())


