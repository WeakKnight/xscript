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
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
