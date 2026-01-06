"""
XScript GPU VM Tests

This module runs the Slang-based VM tests on actual GPU hardware using SlangPy.
Each test kernel from runtime/tests/*.slang is executed individually for
granular error reporting.

Requires: slangpy (GPU support)
"""

import struct
import pytest
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# SlangPy is required for GPU tests
import slangpy

# =============================================================================
# Path Configuration
# =============================================================================

RUNTIME_TESTS_DIR = Path(__file__).parent.parent / "runtime" / "tests"


# =============================================================================
# Test Result Structures (matching Slang structs)
# =============================================================================

@dataclass
class ValueTestResult:
    """Result structure for test_value.slang tests."""
    test_id: int
    passed: bool
    expected_type: int
    actual_type: int
    expected_value: float
    actual_value: float


@dataclass
class StackTestResult:
    """Result structure for test_stack.slang tests."""
    test_id: int
    passed: bool
    stack_pointer: int
    expected_value: float
    actual_value: float


@dataclass
class ArithmeticTestResult:
    """Result structure for test_arithmetic.slang tests."""
    test_id: int
    passed: bool
    expected: float
    actual: float


@dataclass
class VMTestResult:
    """Result structure for test_vm.slang tests."""
    test_id: int
    passed: bool
    vm_status: int
    stack_pointer: int
    expected_value: float
    actual_value: float


# =============================================================================
# GPU Test Runner Base Class
# =============================================================================

class GPUTestRunner:
    """Base class for running Slang GPU tests."""
    
    # Maximum number of test results per run
    MAX_RESULTS = 64
    
    def __init__(self, slang_file: str):
        """Initialize the GPU test runner with a Slang file."""
        self.slang_path = RUNTIME_TESTS_DIR / slang_file
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
        
        # Allocate result buffers (will be configured per test type)
        self._result_buffer = None
        self._count_buffer = None
    
    def _create_buffers(self, result_struct_size: int):
        """Create GPU buffers for test results."""
        # Buffer for test results
        self._result_buffer = self.device.create_buffer(
            size=result_struct_size * self.MAX_RESULTS,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Buffer for test count (single uint)
        self._count_buffer = self.device.create_buffer(
            size=4,  # sizeof(uint)
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Initialize count to 0
        self._count_buffer.copy_from_numpy(np.array([0], dtype=np.uint32))
    
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
    
    def run_kernel(self, kernel_name: str) -> Tuple[bool, str]:
        """Run a compute kernel and return whether it passed."""
        # Get the kernel
        kernel = self._get_kernel(kernel_name)
        
        # Dispatch with 1x1x1 threads (single invocation)
        kernel.dispatch(
            thread_count=slangpy.uint3(1, 1, 1),
            vars={
                "g_testResults": self._result_buffer,
                "g_testCount": self._count_buffer
            }
        )
        
        # Read back results
        return self._check_results()
    
    def _check_results(self) -> Tuple[bool, str]:
        """Check test results from GPU buffers. Override in subclasses."""
        raise NotImplementedError


# =============================================================================
# Value Test Runner
# =============================================================================

class ValueTestRunner(GPUTestRunner):
    """Runner for test_value.slang tests."""
    
    # TestResult struct: uint testId, uint passed, uint expectedType, uint actualType, float expectedValue, float actualValue
    RESULT_STRUCT_SIZE = 24  # 4 + 4 + 4 + 4 + 4 + 4 bytes
    
    def __init__(self):
        super().__init__("test_value.slang")
        self._create_buffers(self.RESULT_STRUCT_SIZE)
    
    def _check_results(self) -> Tuple[bool, str]:
        """Check value test results."""
        # Read count
        count_data = self._count_buffer.to_numpy()
        count = int(count_data[0])
        
        if count == 0:
            return False, "No test results reported"
        
        # Read results
        result_data = self._result_buffer.to_numpy()
        result_bytes = result_data.tobytes()
        
        all_passed = True
        messages = []
        
        for i in range(min(count, self.MAX_RESULTS)):
            offset = i * self.RESULT_STRUCT_SIZE
            test_id, passed, exp_type, act_type, exp_val, act_val = struct.unpack_from(
                'IIIIff', result_bytes, offset
            )
            
            if passed == 0:
                all_passed = False
                messages.append(
                    f"Test {test_id} FAILED: expected type={exp_type} value={exp_val}, "
                    f"got type={act_type} value={act_val}"
                )
        
        return all_passed, "; ".join(messages) if messages else "All tests passed"
    
    def run_test(self, kernel_name: str) -> None:
        """Run a single test kernel and assert it passes."""
        # Reset count buffer
        self._count_buffer.copy_from_numpy(np.array([0], dtype=np.uint32))
        
        passed, message = self.run_kernel(kernel_name)
        assert passed, f"GPU test '{kernel_name}' failed: {message}"


# =============================================================================
# Stack Test Runner
# =============================================================================

class StackTestRunner(GPUTestRunner):
    """Runner for test_stack.slang tests."""
    
    # TestResult struct: uint testId, uint passed, uint stackPointer, float expectedValue, float actualValue
    RESULT_STRUCT_SIZE = 20  # 4 + 4 + 4 + 4 + 4 bytes
    
    def __init__(self):
        super().__init__("test_stack.slang")
        self._create_buffers(self.RESULT_STRUCT_SIZE)
    
    def _check_results(self) -> Tuple[bool, str]:
        """Check stack test results."""
        count_data = self._count_buffer.to_numpy()
        count = int(count_data[0])
        
        if count == 0:
            return False, "No test results reported"
        
        result_data = self._result_buffer.to_numpy()
        result_bytes = result_data.tobytes()
        
        all_passed = True
        messages = []
        
        for i in range(min(count, self.MAX_RESULTS)):
            offset = i * self.RESULT_STRUCT_SIZE
            test_id, passed, sp, exp_val, act_val = struct.unpack_from(
                'IIIff', result_bytes, offset
            )
            
            if passed == 0:
                all_passed = False
                messages.append(
                    f"Test {test_id} FAILED: sp={sp}, expected={exp_val}, actual={act_val}"
                )
        
        return all_passed, "; ".join(messages) if messages else "All tests passed"
    
    def run_test(self, kernel_name: str) -> None:
        """Run a single test kernel and assert it passes."""
        self._count_buffer.copy_from_numpy(np.array([0], dtype=np.uint32))
        passed, message = self.run_kernel(kernel_name)
        assert passed, f"GPU test '{kernel_name}' failed: {message}"


# =============================================================================
# Arithmetic Test Runner
# =============================================================================

class ArithmeticTestRunner(GPUTestRunner):
    """Runner for test_arithmetic.slang tests."""
    
    # TestResult struct: uint testId, uint passed, float expected, float actual
    RESULT_STRUCT_SIZE = 16  # 4 + 4 + 4 + 4 bytes
    
    def __init__(self):
        super().__init__("test_arithmetic.slang")
        self._create_buffers(self.RESULT_STRUCT_SIZE)
    
    def _check_results(self) -> Tuple[bool, str]:
        """Check arithmetic test results."""
        count_data = self._count_buffer.to_numpy()
        count = int(count_data[0])
        
        if count == 0:
            return False, "No test results reported"
        
        result_data = self._result_buffer.to_numpy()
        result_bytes = result_data.tobytes()
        
        all_passed = True
        messages = []
        
        for i in range(min(count, self.MAX_RESULTS)):
            offset = i * self.RESULT_STRUCT_SIZE
            test_id, passed, expected, actual = struct.unpack_from(
                'IIff', result_bytes, offset
            )
            
            if passed == 0:
                all_passed = False
                messages.append(
                    f"Test {test_id} FAILED: expected={expected}, actual={actual}"
                )
        
        return all_passed, "; ".join(messages) if messages else "All tests passed"
    
    def run_test(self, kernel_name: str) -> None:
        """Run a single test kernel and assert it passes."""
        self._count_buffer.copy_from_numpy(np.array([0], dtype=np.uint32))
        passed, message = self.run_kernel(kernel_name)
        assert passed, f"GPU test '{kernel_name}' failed: {message}"


# =============================================================================
# VM Test Runner
# =============================================================================

class VMTestRunner(GPUTestRunner):
    """Runner for test_vm.slang tests."""
    
    # TestResult struct: uint testId, uint passed, uint vmStatus, uint stackPointer, float expectedValue, float actualValue
    RESULT_STRUCT_SIZE = 24  # 4 + 4 + 4 + 4 + 4 + 4 bytes
    
    def __init__(self):
        super().__init__("test_vm.slang")
        self._create_buffers(self.RESULT_STRUCT_SIZE)
    
    def _check_results(self) -> Tuple[bool, str]:
        """Check VM test results."""
        count_data = self._count_buffer.to_numpy()
        count = int(count_data[0])
        
        if count == 0:
            return False, "No test results reported"
        
        result_data = self._result_buffer.to_numpy()
        result_bytes = result_data.tobytes()
        
        all_passed = True
        messages = []
        
        for i in range(min(count, self.MAX_RESULTS)):
            offset = i * self.RESULT_STRUCT_SIZE
            test_id, passed, status, sp, exp_val, act_val = struct.unpack_from(
                'IIIIff', result_bytes, offset
            )
            
            if passed == 0:
                all_passed = False
                messages.append(
                    f"Test {test_id} FAILED: status={status}, sp={sp}, "
                    f"expected={exp_val}, actual={act_val}"
                )
        
        return all_passed, "; ".join(messages) if messages else "All tests passed"
    
    def run_test(self, kernel_name: str) -> None:
        """Run a single test kernel and assert it passes."""
        self._count_buffer.copy_from_numpy(np.array([0], dtype=np.uint32))
        passed, message = self.run_kernel(kernel_name)
        assert passed, f"GPU test '{kernel_name}' failed: {message}"


# =============================================================================
# Ops Test Runner
# =============================================================================

class OpsTestRunner(GPUTestRunner):
    """Runner for test_ops.slang tests."""
    
    # TestResult struct: uint testId, uint passed, float expected, float actual
    RESULT_STRUCT_SIZE = 16  # 4 + 4 + 4 + 4 bytes
    
    def __init__(self):
        super().__init__("test_ops.slang")
        self._create_buffers(self.RESULT_STRUCT_SIZE)
    
    def _check_results(self) -> Tuple[bool, str]:
        """Check ops test results."""
        count_data = self._count_buffer.to_numpy()
        count = int(count_data[0])
        
        if count == 0:
            return False, "No test results reported"
        
        result_data = self._result_buffer.to_numpy()
        result_bytes = result_data.tobytes()
        
        all_passed = True
        messages = []
        
        for i in range(min(count, self.MAX_RESULTS)):
            offset = i * self.RESULT_STRUCT_SIZE
            test_id, passed, expected, actual = struct.unpack_from(
                'IIff', result_bytes, offset
            )
            
            if passed == 0:
                all_passed = False
                messages.append(
                    f"Test {test_id} FAILED: expected={expected}, actual={actual}"
                )
        
        return all_passed, "; ".join(messages) if messages else "All tests passed"
    
    def run_test(self, kernel_name: str) -> None:
        """Run a single test kernel and assert it passes."""
        self._count_buffer.copy_from_numpy(np.array([0], dtype=np.uint32))
        passed, message = self.run_kernel(kernel_name)
        assert passed, f"GPU test '{kernel_name}' failed: {message}"


# =============================================================================
# Heap Test Runner
# =============================================================================

class HeapTestRunner(GPUTestRunner):
    """Runner for test_heap.slang tests.
    
    This runner creates additional buffers for heap memory and heap state.
    """
    
    # TestResult struct: uint testId, uint passed, uint expected, uint actual
    RESULT_STRUCT_SIZE = 16  # 4 + 4 + 4 + 4 bytes
    
    # Heap configuration
    HEAP_SIZE = 16384  # 16384 uints = 64KB heap memory (larger for concurrent tests)
    HEAP_STATE_SIZE = 32  # HeapAllocator struct = 8 uints = 32 bytes
    ALLOC_RESULTS_SIZE = 128  # Buffer for storing concurrent allocation results
    
    def __init__(self):
        super().__init__("test_heap.slang")
        self._create_buffers(self.RESULT_STRUCT_SIZE)
        self._create_heap_buffers()
    
    def _create_heap_buffers(self):
        """Create GPU buffers for heap memory and state."""
        # Heap memory buffer (array of uints)
        self._heap_memory_buffer = self.device.create_buffer(
            size=self.HEAP_SIZE * 4,  # 4 bytes per uint
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Heap state buffer (HeapAllocator struct)
        self._heap_state_buffer = self.device.create_buffer(
            size=self.HEAP_STATE_SIZE,
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Alloc results buffer (for concurrent tests)
        self._alloc_results_buffer = self.device.create_buffer(
            size=self.ALLOC_RESULTS_SIZE * 4,  # 4 bytes per uint
            usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
        )
        
        # Initialize buffers to zero
        self._heap_memory_buffer.copy_from_numpy(np.zeros(self.HEAP_SIZE, dtype=np.uint32))
        self._heap_state_buffer.copy_from_numpy(np.zeros(8, dtype=np.uint32))
        self._alloc_results_buffer.copy_from_numpy(np.zeros(self.ALLOC_RESULTS_SIZE, dtype=np.uint32))
    
    def run_kernel(self, kernel_name: str, thread_count: Tuple[int, int, int] = (1, 1, 1)) -> Tuple[bool, str]:
        """Run a compute kernel with heap buffers."""
        kernel = self._get_kernel(kernel_name)
        
        # Dispatch with all 5 buffers
        kernel.dispatch(
            thread_count=slangpy.uint3(thread_count[0], thread_count[1], thread_count[2]),
            vars={
                "g_testResults": self._result_buffer,
                "g_testCount": self._count_buffer,
                "g_heapMemory": self._heap_memory_buffer,
                "g_heapState": self._heap_state_buffer,
                "g_allocResults": self._alloc_results_buffer
            }
        )
        
        return self._check_results()
    
    def _check_results(self) -> Tuple[bool, str]:
        """Check heap test results."""
        count_data = self._count_buffer.to_numpy()
        count = int(count_data[0])
        
        if count == 0:
            return False, "No test results reported"
        
        result_data = self._result_buffer.to_numpy()
        result_bytes = result_data.tobytes()
        
        all_passed = True
        messages = []
        
        for i in range(min(count, self.MAX_RESULTS)):
            offset = i * self.RESULT_STRUCT_SIZE
            test_id, passed, expected, actual = struct.unpack_from(
                'IIII', result_bytes, offset
            )
            
            if passed == 0:
                all_passed = False
                messages.append(
                    f"Test {test_id} FAILED: expected={expected}, actual={actual}"
                )
        
        return all_passed, "; ".join(messages) if messages else "All tests passed"
    
    def _reset_buffers(self):
        """Reset all buffers before a test."""
        self._count_buffer.copy_from_numpy(np.array([0], dtype=np.uint32))
        self._heap_memory_buffer.copy_from_numpy(np.zeros(self.HEAP_SIZE, dtype=np.uint32))
        self._heap_state_buffer.copy_from_numpy(np.zeros(8, dtype=np.uint32))
        self._alloc_results_buffer.copy_from_numpy(np.zeros(self.ALLOC_RESULTS_SIZE, dtype=np.uint32))
    
    def run_test(self, kernel_name: str) -> None:
        """Run a single test kernel and assert it passes."""
        self._reset_buffers()
        passed, message = self.run_kernel(kernel_name)
        assert passed, f"GPU test '{kernel_name}' failed: {message}"
    
    def run_concurrent_test(self, kernel_name: str, thread_count: Tuple[int, int, int]) -> None:
        """Run a concurrent test kernel with specified thread count."""
        self._reset_buffers()
        passed, message = self.run_kernel(kernel_name, thread_count)
        assert passed, f"GPU concurrent test '{kernel_name}' failed: {message}"


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def value_runner():
    """Create a ValueTestRunner for the test module."""
    return ValueTestRunner()


@pytest.fixture(scope="module")
def stack_runner():
    """Create a StackTestRunner for the test module."""
    return StackTestRunner()


@pytest.fixture(scope="module")
def arithmetic_runner():
    """Create an ArithmeticTestRunner for the test module."""
    return ArithmeticTestRunner()


@pytest.fixture(scope="module")
def vm_runner():
    """Create a VMTestRunner for the test module."""
    return VMTestRunner()


@pytest.fixture(scope="module")
def ops_runner():
    """Create an OpsTestRunner for the test module."""
    return OpsTestRunner()


@pytest.fixture(scope="module")
def heap_runner():
    """Create a HeapTestRunner for the test module."""
    return HeapTestRunner()


# =============================================================================
# Value Tests (test_value.slang)
# =============================================================================

class TestValueGPU:
    """GPU tests for XValue operations from test_value.slang."""
    
    # Creation tests
    def test_nil_creation(self, value_runner):
        value_runner.run_test("test_nil_creation")
    
    def test_bool_true(self, value_runner):
        value_runner.run_test("test_bool_true")
    
    def test_bool_false(self, value_runner):
        value_runner.run_test("test_bool_false")
    
    def test_number_integer(self, value_runner):
        value_runner.run_test("test_number_integer")
    
    def test_number_float(self, value_runner):
        value_runner.run_test("test_number_float")
    
    def test_number_negative(self, value_runner):
        value_runner.run_test("test_number_negative")
    
    def test_number_zero(self, value_runner):
        value_runner.run_test("test_number_zero")
    
    # Equality tests
    def test_nil_equals_nil(self, value_runner):
        value_runner.run_test("test_nil_equals_nil")
    
    def test_number_equals(self, value_runner):
        value_runner.run_test("test_number_equals")
    
    def test_number_not_equals(self, value_runner):
        value_runner.run_test("test_number_not_equals")
    
    def test_bool_equals(self, value_runner):
        value_runner.run_test("test_bool_equals")
    
    def test_different_types_not_equal(self, value_runner):
        value_runner.run_test("test_different_types_not_equal")
    
    # Truthiness tests
    def test_nil_is_falsy(self, value_runner):
        value_runner.run_test("test_nil_is_falsy")
    
    def test_false_is_falsy(self, value_runner):
        value_runner.run_test("test_false_is_falsy")
    
    def test_true_is_truthy(self, value_runner):
        value_runner.run_test("test_true_is_truthy")
    
    def test_zero_is_truthy(self, value_runner):
        value_runner.run_test("test_zero_is_truthy")


# =============================================================================
# Stack Tests (test_stack.slang)
# =============================================================================

class TestStackGPU:
    """GPU tests for VM stack operations from test_stack.slang."""
    
    # Push/Pop tests
    def test_push_single(self, stack_runner):
        stack_runner.run_test("test_push_single")
    
    def test_push_pop(self, stack_runner):
        stack_runner.run_test("test_push_pop")
    
    def test_multiple_push_pop(self, stack_runner):
        stack_runner.run_test("test_multiple_push_pop")
    
    def test_pop_empty_stack(self, stack_runner):
        stack_runner.run_test("test_pop_empty_stack")
    
    # Peek tests
    def test_peek_top(self, stack_runner):
        stack_runner.run_test("test_peek_top")
    
    def test_peek_offset(self, stack_runner):
        stack_runner.run_test("test_peek_offset")
    
    def test_peek_out_of_bounds(self, stack_runner):
        stack_runner.run_test("test_peek_out_of_bounds")
    
    # DUP tests
    def test_dup(self, stack_runner):
        stack_runner.run_test("test_dup")
    
    def test_dup_empty(self, stack_runner):
        stack_runner.run_test("test_dup_empty")
    
    # SWAP tests
    def test_swap(self, stack_runner):
        stack_runner.run_test("test_swap")
    
    def test_swap_insufficient(self, stack_runner):
        stack_runner.run_test("test_swap_insufficient")


# =============================================================================
# Arithmetic Tests (test_arithmetic.slang)
# =============================================================================

class TestArithmeticGPU:
    """GPU tests for arithmetic operations from test_arithmetic.slang."""
    
    # Addition tests
    def test_add_integers(self, arithmetic_runner):
        arithmetic_runner.run_test("test_add_integers")
    
    def test_add_floats(self, arithmetic_runner):
        arithmetic_runner.run_test("test_add_floats")
    
    def test_add_negative(self, arithmetic_runner):
        arithmetic_runner.run_test("test_add_negative")
    
    # Subtraction tests
    def test_sub_integers(self, arithmetic_runner):
        arithmetic_runner.run_test("test_sub_integers")
    
    def test_sub_to_negative(self, arithmetic_runner):
        arithmetic_runner.run_test("test_sub_to_negative")
    
    # Multiplication tests
    def test_mul_integers(self, arithmetic_runner):
        arithmetic_runner.run_test("test_mul_integers")
    
    def test_mul_by_zero(self, arithmetic_runner):
        arithmetic_runner.run_test("test_mul_by_zero")
    
    def test_mul_negatives(self, arithmetic_runner):
        arithmetic_runner.run_test("test_mul_negatives")
    
    # Division tests
    def test_div_integers(self, arithmetic_runner):
        arithmetic_runner.run_test("test_div_integers")
    
    def test_div_fraction(self, arithmetic_runner):
        arithmetic_runner.run_test("test_div_fraction")
    
    # Modulo tests
    def test_mod_basic(self, arithmetic_runner):
        arithmetic_runner.run_test("test_mod_basic")
    
    def test_mod_no_remainder(self, arithmetic_runner):
        arithmetic_runner.run_test("test_mod_no_remainder")
    
    # Power tests
    def test_pow_square(self, arithmetic_runner):
        arithmetic_runner.run_test("test_pow_square")
    
    def test_pow_cube(self, arithmetic_runner):
        arithmetic_runner.run_test("test_pow_cube")
    
    def test_pow_zero(self, arithmetic_runner):
        arithmetic_runner.run_test("test_pow_zero")
    
    # Negation tests
    def test_neg_positive(self, arithmetic_runner):
        arithmetic_runner.run_test("test_neg_positive")
    
    def test_neg_negative(self, arithmetic_runner):
        arithmetic_runner.run_test("test_neg_negative")
    
    # Comparison tests
    def test_lt_true(self, arithmetic_runner):
        arithmetic_runner.run_test("test_lt_true")
    
    def test_lt_false(self, arithmetic_runner):
        arithmetic_runner.run_test("test_lt_false")
    
    def test_le_equal(self, arithmetic_runner):
        arithmetic_runner.run_test("test_le_equal")


# =============================================================================
# VM Bytecode Tests (test_vm.slang)
# =============================================================================

class TestVMGPU:
    """GPU tests for VM bytecode execution from test_vm.slang."""
    
    # Basic push tests
    def test_push_nil_halt(self, vm_runner):
        vm_runner.run_test("test_push_nil_halt")
    
    def test_push_true_false(self, vm_runner):
        vm_runner.run_test("test_push_true_false")
    
    def test_push_num_constant(self, vm_runner):
        vm_runner.run_test("test_push_num_constant")
    
    # Arithmetic tests
    def test_add_two_numbers(self, vm_runner):
        vm_runner.run_test("test_add_two_numbers")
    
    def test_arithmetic_expression(self, vm_runner):
        vm_runner.run_test("test_arithmetic_expression")
    
    # Global tests
    def test_set_get_global(self, vm_runner):
        vm_runner.run_test("test_set_get_global")
    
    # Comparison tests
    def test_comparison_eq(self, vm_runner):
        vm_runner.run_test("test_comparison_eq")
    
    def test_comparison_lt(self, vm_runner):
        vm_runner.run_test("test_comparison_lt")
    
    # Negation test
    def test_negation(self, vm_runner):
        vm_runner.run_test("test_negation")


# =============================================================================
# Ops Tests (test_ops.slang)
# =============================================================================

class TestOpsGPU:
    """GPU tests for extended VM operations from test_ops.slang."""
    
    # Local variable tests (400-409)
    def test_get_local_basic(self, ops_runner):
        ops_runner.run_test("test_get_local_basic")
    
    def test_get_local_with_fp(self, ops_runner):
        ops_runner.run_test("test_get_local_with_fp")
    
    def test_get_local_out_of_bounds(self, ops_runner):
        ops_runner.run_test("test_get_local_out_of_bounds")
    
    def test_set_local_basic(self, ops_runner):
        ops_runner.run_test("test_set_local_basic")
    
    def test_set_local_with_fp(self, ops_runner):
        ops_runner.run_test("test_set_local_with_fp")
    
    # Binary operations - numbers (410-419)
    def test_binop_add_numbers(self, ops_runner):
        ops_runner.run_test("test_binop_add_numbers")
    
    def test_binop_sub_numbers(self, ops_runner):
        ops_runner.run_test("test_binop_sub_numbers")
    
    def test_binop_mul_numbers(self, ops_runner):
        ops_runner.run_test("test_binop_mul_numbers")
    
    def test_binop_div_numbers(self, ops_runner):
        ops_runner.run_test("test_binop_div_numbers")
    
    def test_binop_div_by_zero(self, ops_runner):
        ops_runner.run_test("test_binop_div_by_zero")
    
    def test_binop_mod_numbers(self, ops_runner):
        ops_runner.run_test("test_binop_mod_numbers")
    
    def test_binop_pow_numbers(self, ops_runner):
        ops_runner.run_test("test_binop_pow_numbers")
    
    # Binary operations - type errors (420-429)
    def test_binop_add_nil_number(self, ops_runner):
        ops_runner.run_test("test_binop_add_nil_number")
    
    def test_binop_add_bool_number(self, ops_runner):
        ops_runner.run_test("test_binop_add_bool_number")
    
    def test_binop_sub_nil_number(self, ops_runner):
        ops_runner.run_test("test_binop_sub_nil_number")
    
    def test_binop_mul_number_nil(self, ops_runner):
        ops_runner.run_test("test_binop_mul_number_nil")
    
    def test_binop_div_bool_bool(self, ops_runner):
        ops_runner.run_test("test_binop_div_bool_bool")
    
    # Function call tests (430-439)
    def test_call_frame_setup(self, ops_runner):
        ops_runner.run_test("test_call_frame_setup")
    
    def test_function_descriptor(self, ops_runner):
        ops_runner.run_test("test_function_descriptor")
    
    def test_vm_return_simple(self, ops_runner):
        ops_runner.run_test("test_vm_return_simple")
    
    def test_frame_pointer_mechanics(self, ops_runner):
        ops_runner.run_test("test_frame_pointer_mechanics")
    
    def test_nested_frame_restore(self, ops_runner):
        ops_runner.run_test("test_nested_frame_restore")


# =============================================================================
# Heap Tests (test_heap.slang)
# =============================================================================

class TestHeapGPU:
    """GPU tests for heap memory operations from test_heap.slang."""
    
    # Helper function tests (500-509)
    def test_align_up_basic(self, heap_runner):
        heap_runner.run_test("test_align_up_basic")
    
    def test_align_up_already_aligned(self, heap_runner):
        heap_runner.run_test("test_align_up_already_aligned")
    
    def test_align_up_zero(self, heap_runner):
        heap_runner.run_test("test_align_up_zero")
    
    def test_size_class_small(self, heap_runner):
        heap_runner.run_test("test_size_class_small")
    
    def test_size_class_medium(self, heap_runner):
        heap_runner.run_test("test_size_class_medium")
    
    def test_size_class_large(self, heap_runner):
        heap_runner.run_test("test_size_class_large")
    
    def test_size_class_huge(self, heap_runner):
        heap_runner.run_test("test_size_class_huge")
    
    # Heap initialization tests (510-519)
    def test_heap_init(self, heap_runner):
        heap_runner.run_test("test_heap_init")
    
    def test_heap_get_stats_after_init(self, heap_runner):
        heap_runner.run_test("test_heap_get_stats_after_init")
    
    # Basic allocation tests (520-529)
    def test_heap_alloc_single(self, heap_runner):
        heap_runner.run_test("test_heap_alloc_single")
    
    def test_heap_alloc_updates_stats(self, heap_runner):
        heap_runner.run_test("test_heap_alloc_updates_stats")
    
    def test_heap_alloc_multiple(self, heap_runner):
        heap_runner.run_test("test_heap_alloc_multiple")
    
    def test_heap_alloc_sequential(self, heap_runner):
        heap_runner.run_test("test_heap_alloc_sequential")
    
    def test_heap_alloc_fails_when_full(self, heap_runner):
        heap_runner.run_test("test_heap_alloc_fails_when_full")
    
    # Memory access tests (530-539)
    def test_heap_write_read_uint(self, heap_runner):
        heap_runner.run_test("test_heap_write_read_uint")
    
    def test_heap_write_read_multiple_uint(self, heap_runner):
        heap_runner.run_test("test_heap_write_read_multiple_uint")
    
    def test_heap_write_read_float(self, heap_runner):
        heap_runner.run_test("test_heap_write_read_float")
    
    def test_heap_write_read_negative_float(self, heap_runner):
        heap_runner.run_test("test_heap_write_read_negative_float")
    
    def test_heap_separate_allocations(self, heap_runner):
        heap_runner.run_test("test_heap_separate_allocations")
    
    # Reference counting tests (540-549)
    def test_heap_initial_refcount(self, heap_runner):
        heap_runner.run_test("test_heap_initial_refcount")
    
    def test_heap_incref(self, heap_runner):
        heap_runner.run_test("test_heap_incref")
    
    def test_heap_incref_multiple(self, heap_runner):
        heap_runner.run_test("test_heap_incref_multiple")
    
    def test_heap_decref_no_free(self, heap_runner):
        heap_runner.run_test("test_heap_decref_no_free")
    
    def test_heap_decref_with_free(self, heap_runner):
        heap_runner.run_test("test_heap_decref_with_free")
    
    def test_heap_get_refcount_null(self, heap_runner):
        heap_runner.run_test("test_heap_get_refcount_null")
    
    # Free and reuse tests (550-559)
    def test_heap_free(self, heap_runner):
        heap_runner.run_test("test_heap_free")
    
    def test_heap_free_null(self, heap_runner):
        heap_runner.run_test("test_heap_free_null")
    
    def test_heap_block_marked_free(self, heap_runner):
        heap_runner.run_test("test_heap_block_marked_free")
    
    def test_heap_multiple_alloc_free(self, heap_runner):
        heap_runner.run_test("test_heap_multiple_alloc_free")
    
    # Concurrent allocation tests (560-569)
    def test_concurrent_alloc_32(self, heap_runner):
        """32 threads concurrently allocating memory."""
        heap_runner.run_concurrent_test("test_concurrent_alloc_32", (1, 1, 1))
    
    def test_concurrent_alloc_64(self, heap_runner):
        """64 threads concurrently allocating memory."""
        heap_runner.run_concurrent_test("test_concurrent_alloc_64", (1, 1, 1))
    
    def test_concurrent_incref_32(self, heap_runner):
        """32 threads concurrently incrementing refcount."""
        heap_runner.run_concurrent_test("test_concurrent_incref_32", (1, 1, 1))
    
    def test_concurrent_decref_16(self, heap_runner):
        """16 threads concurrently decrementing refcount."""
        heap_runner.run_concurrent_test("test_concurrent_decref_16", (1, 1, 1))
    
    def test_concurrent_alloc_multigroup(self, heap_runner):
        """Multiple thread groups allocating concurrently."""
        heap_runner.run_concurrent_test("test_concurrent_alloc_multigroup", (4, 1, 1))
    
    def test_concurrent_alloc_varying_sizes(self, heap_runner):
        """16 threads allocating different sizes concurrently."""
        heap_runner.run_concurrent_test("test_concurrent_alloc_varying_sizes", (1, 1, 1))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
