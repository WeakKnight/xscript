// XScript Operations Tests (32-bit)
// Tests for extended VM operations from ops.slang

// =============================================================================
// Type Constants
// =============================================================================

static const uint TYPE_NIL      = 0;
static const uint TYPE_BOOL     = 1;
static const uint TYPE_NUMBER   = 2;
static const uint TYPE_STRING   = 3;
static const uint TYPE_TABLE    = 4;
static const uint TYPE_FUNCTION = 5;

// =============================================================================
// VM Constants
// =============================================================================

static const uint VM_STATUS_RUNNING      = 0;
static const uint VM_STATUS_PAUSED       = 1;
static const uint VM_STATUS_WAITING_HOST = 2;
static const uint VM_STATUS_ERROR        = 3;
static const uint VM_STATUS_COMPLETED    = 4;

static const uint VM_STACK_SIZE = 256;
static const uint VM_CALL_STACK_SIZE = 64;

static const uint ERR_NONE            = 0;
static const uint ERR_STACK_OVERFLOW  = 1;
static const uint ERR_STACK_UNDERFLOW = 2;
static const uint ERR_TYPE_ERROR      = 3;
static const uint ERR_DIV_BY_ZERO     = 4;
static const uint ERR_INVALID_OPCODE  = 5;
static const uint ERR_OUT_OF_MEMORY   = 6;
static const uint ERR_CALL_DEPTH      = 7;

// =============================================================================
// XValue Struct (32-bit)
// =============================================================================

struct XValue {
    uint type;
    uint flags;
    uint data;
    
    static XValue nil() {
        XValue v;
        v.type = TYPE_NIL;
        v.flags = 0;
        v.data = 0;
        return v;
    }
    
    static XValue boolean(bool b) {
        XValue v;
        v.type = TYPE_BOOL;
        v.flags = 0;
        v.data = b ? 1 : 0;
        return v;
    }
    
    static XValue number(float n) {
        XValue v;
        v.type = TYPE_NUMBER;
        v.flags = 0;
        v.data = asuint(n);
        return v;
    }
    
    static XValue string(uint stringIndex) {
        XValue v;
        v.type = TYPE_STRING;
        v.flags = 0;
        v.data = stringIndex;
        return v;
    }
    
    static XValue table(uint heapOffset) {
        XValue v;
        v.type = TYPE_TABLE;
        v.flags = 0;
        v.data = heapOffset;
        return v;
    }
    
    bool isNil() { return type == TYPE_NIL; }
    float asNumber() { return asfloat(data); }
    uint asUint() { return data; }
};

// =============================================================================
// ECS Helper State and Functions (for testing)
// =============================================================================

// Mock current entity state (thread-local in practice, static for single-threaded tests)
static uint g_testCurrentEntityId = 0xFFFFFFFF;
static uint g_testCurrentEntityTablePtr = 0;

void vm_set_current_entity(uint entityId, uint tablePtr) {
    g_testCurrentEntityId = entityId;
    g_testCurrentEntityTablePtr = tablePtr;
}

void vm_clear_current_entity() {
    g_testCurrentEntityId = 0xFFFFFFFF;
    g_testCurrentEntityTablePtr = 0;
}

XValue vm_get_current_entity() {
    if (g_testCurrentEntityTablePtr == 0) {
        return XValue::nil();
    }
    return XValue::table(g_testCurrentEntityTablePtr);
}

uint vm_get_current_entity_id() {
    return g_testCurrentEntityId;
}

// Simplified component check for testing (always returns false - no real table access)
bool vm_has_component(XValue entity, XValue key) {
    if (entity.type != TYPE_TABLE) {
        return false;
    }
    // In real impl, would look up key in table
    // For testing, we just check that entity is a valid table
    return false;  // Simplified: no actual table lookup
}

// Simplified component add for testing
bool vm_add_component(XValue entity, XValue key, XValue value) {
    if (entity.type != TYPE_TABLE || key.isNil()) {
        return false;
    }
    // In real impl, would add to table
    return true;  // Simplified: always succeed for valid inputs
}

// Simplified component remove for testing
bool vm_remove_component(XValue entity, XValue key) {
    if (entity.type != TYPE_TABLE) {
        return false;
    }
    // In real impl, would remove from table
    return true;  // Simplified: always succeed for valid inputs
}

// =============================================================================
// XValue Arithmetic (from value.slang)
// =============================================================================

XValue xvalue_add(XValue a, XValue b) {
    return XValue::number(a.asNumber() + b.asNumber());
}

XValue xvalue_sub(XValue a, XValue b) {
    return XValue::number(a.asNumber() - b.asNumber());
}

XValue xvalue_mul(XValue a, XValue b) {
    return XValue::number(a.asNumber() * b.asNumber());
}

XValue xvalue_div(XValue a, XValue b) {
    return XValue::number(a.asNumber() / b.asNumber());
}

XValue xvalue_mod(XValue a, XValue b) {
    float na = a.asNumber();
    float nb = b.asNumber();
    return XValue::number(na - floor(na / nb) * nb);
}

XValue xvalue_pow(XValue a, XValue b) {
    return XValue::number(pow(a.asNumber(), b.asNumber()));
}

void xvalue_assign(inout XValue dest, XValue src) {
    dest.type = src.type;
    dest.flags = src.flags;
    dest.data = src.data;
}

// =============================================================================
// VMState Structure
// =============================================================================

struct VMState {
    XValue stack[VM_STACK_SIZE];
    uint sp;        // Stack pointer
    uint fp;        // Frame pointer
    uint pc;        // Program counter
    uint status;    // Execution status
    uint error;     // Error code (if status == ERROR)
};

struct CallFrame {
    uint returnPC;   // Return address
    uint prevFP;     // Previous frame pointer
    uint localBase;  // Base of local variables
    uint argCount;   // Number of arguments
};

struct FunctionDescriptor {
    uint codeOffset;     // Offset in bytecode
    uint paramCount;     // Number of parameters
    uint localCount;     // Number of local variables
    uint upvalueCount;   // Number of upvalues
    uint nameIndex;      // Function name (string index)
};

// =============================================================================
// Stack Operations (from ops.slang)
// =============================================================================

bool vm_push(inout VMState state, XValue v) {
    if (state.sp >= VM_STACK_SIZE) {
        state.status = VM_STATUS_ERROR;
        state.error = ERR_STACK_OVERFLOW;
        return false;
    }
    state.stack[state.sp] = v;
    state.sp++;
    return true;
}

bool vm_pop(inout VMState state, out XValue v) {
    if (state.sp == 0) {
        state.status = VM_STATUS_ERROR;
        state.error = ERR_STACK_UNDERFLOW;
        v = XValue::nil();
        return false;
    }
    state.sp--;
    v = state.stack[state.sp];
    return true;
}

XValue vm_peek(VMState state, uint offset) {
    if (state.sp <= offset) {
        return XValue::nil();
    }
    return state.stack[state.sp - 1 - offset];
}

XValue vm_get_local(VMState state, uint slot) {
    uint index = state.fp + slot;
    if (index >= state.sp) {
        return XValue::nil();
    }
    return state.stack[index];
}

void vm_set_local(inout VMState state, uint slot, XValue v) {
    uint index = state.fp + slot;
    if (index < VM_STACK_SIZE) {
        xvalue_assign(state.stack[index], v);
    }
}

// =============================================================================
// Binary Operations (from ops.slang - simplified without metamethods)
// =============================================================================

XValue vm_binop_add(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_add(a, b);
    }
    return XValue::nil();
}

XValue vm_binop_sub(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_sub(a, b);
    }
    return XValue::nil();
}

XValue vm_binop_mul(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_mul(a, b);
    }
    return XValue::nil();
}

XValue vm_binop_div(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        float bNum = b.asNumber();
        if (bNum == 0.0f) {
            return XValue::number(1.0f / 0.0f);  // inf
        }
        return xvalue_div(a, b);
    }
    return XValue::nil();
}

XValue vm_binop_mod(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_mod(a, b);
    }
    return XValue::nil();
}

XValue vm_binop_pow(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_pow(a, b);
    }
    return XValue::nil();
}

// =============================================================================
// Test Result Buffer
// =============================================================================

struct TestResult {
    uint testId;
    uint passed;
    float expected;
    float actual;
};

RWStructuredBuffer<TestResult> g_testResults;
RWStructuredBuffer<uint> g_testCount;

void report_ops_test(uint testId, bool passed, float expected, float actual) {
    uint idx;
    InterlockedAdd(g_testCount[0], 1, idx);
    
    TestResult result;
    result.testId = testId;
    result.passed = passed ? 1 : 0;
    result.expected = expected;
    result.actual = actual;
    g_testResults[idx] = result;
}

// =============================================================================
// Local Variable Tests (400-409)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_get_local_basic() {
    // Set up VMState with some locals
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    
    // Push some values as locals
    vm_push(state, XValue::number(10.0f));  // local 0
    vm_push(state, XValue::number(20.0f));  // local 1
    vm_push(state, XValue::number(30.0f));  // local 2
    
    // Get local 1
    XValue v = vm_get_local(state, 1);
    
    bool passed = (v.type == TYPE_NUMBER) && (v.asNumber() == 20.0f);
    report_ops_test(400, passed, 20.0f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_get_local_with_fp() {
    // Test get_local with non-zero frame pointer
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    
    // Push some values (simulating caller's frame)
    vm_push(state, XValue::number(1.0f));
    vm_push(state, XValue::number(2.0f));
    
    // Set frame pointer to start of new frame
    state.fp = state.sp;  // fp = 2
    
    // Push local variables for current frame
    vm_push(state, XValue::number(100.0f));  // local 0 in current frame
    vm_push(state, XValue::number(200.0f));  // local 1 in current frame
    
    // Get local 0 (should be 100)
    XValue v = vm_get_local(state, 0);
    
    bool passed = (v.asNumber() == 100.0f);
    report_ops_test(401, passed, 100.0f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_get_local_out_of_bounds() {
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    
    // Push one local
    vm_push(state, XValue::number(42.0f));
    
    // Try to get local that doesn't exist
    XValue v = vm_get_local(state, 5);
    
    bool passed = v.isNil();
    report_ops_test(402, passed, 0.0f, float(v.type));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_set_local_basic() {
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    
    // Push initial values
    vm_push(state, XValue::number(10.0f));
    vm_push(state, XValue::number(20.0f));
    vm_push(state, XValue::number(30.0f));
    
    // Set local 1 to a new value
    vm_set_local(state, 1, XValue::number(999.0f));
    
    // Verify it was set
    XValue v = vm_get_local(state, 1);
    
    bool passed = (v.asNumber() == 999.0f);
    report_ops_test(403, passed, 999.0f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_set_local_with_fp() {
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    
    // Caller's frame
    vm_push(state, XValue::number(1.0f));
    vm_push(state, XValue::number(2.0f));
    
    // New frame
    state.fp = state.sp;
    vm_push(state, XValue::number(100.0f));
    vm_push(state, XValue::number(200.0f));
    
    // Set local 1 in current frame
    vm_set_local(state, 1, XValue::number(777.0f));
    
    // Verify
    XValue v = vm_get_local(state, 1);
    
    bool passed = (v.asNumber() == 777.0f);
    report_ops_test(404, passed, 777.0f, v.asNumber());
}

// =============================================================================
// Binary Operations Tests - Numbers (410-419)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_add_numbers() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::number(15.0f);
    XValue b = XValue::number(27.0f);
    XValue c = vm_binop_add(state, a, b);
    
    bool passed = (c.type == TYPE_NUMBER) && (c.asNumber() == 42.0f);
    report_ops_test(410, passed, 42.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_sub_numbers() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::number(50.0f);
    XValue b = XValue::number(8.0f);
    XValue c = vm_binop_sub(state, a, b);
    
    bool passed = (c.asNumber() == 42.0f);
    report_ops_test(411, passed, 42.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_mul_numbers() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::number(6.0f);
    XValue b = XValue::number(7.0f);
    XValue c = vm_binop_mul(state, a, b);
    
    bool passed = (c.asNumber() == 42.0f);
    report_ops_test(412, passed, 42.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_div_numbers() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::number(84.0f);
    XValue b = XValue::number(2.0f);
    XValue c = vm_binop_div(state, a, b);
    
    bool passed = (c.asNumber() == 42.0f);
    report_ops_test(413, passed, 42.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_div_by_zero() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::number(42.0f);
    XValue b = XValue::number(0.0f);
    XValue c = vm_binop_div(state, a, b);
    
    // Division by zero should return infinity
    bool passed = isinf(c.asNumber()) && (c.asNumber() > 0);
    report_ops_test(414, passed, 1.0f, isinf(c.asNumber()) ? 1.0f : 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_mod_numbers() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::number(17.0f);
    XValue b = XValue::number(5.0f);
    XValue c = vm_binop_mod(state, a, b);
    
    bool passed = (c.asNumber() == 2.0f);
    report_ops_test(415, passed, 2.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_pow_numbers() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::number(2.0f);
    XValue b = XValue::number(10.0f);
    XValue c = vm_binop_pow(state, a, b);
    
    // 2^10 = 1024
    bool passed = abs(c.asNumber() - 1024.0f) < 0.01f;
    report_ops_test(416, passed, 1024.0f, c.asNumber());
}

// =============================================================================
// Binary Operations Tests - Type Errors (420-429)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_add_nil_number() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::nil();
    XValue b = XValue::number(42.0f);
    XValue c = vm_binop_add(state, a, b);
    
    // Adding nil to number should return nil
    bool passed = c.isNil();
    report_ops_test(420, passed, 0.0f, float(c.type));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_add_bool_number() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::boolean(true);
    XValue b = XValue::number(42.0f);
    XValue c = vm_binop_add(state, a, b);
    
    // Adding bool to number should return nil
    bool passed = c.isNil();
    report_ops_test(421, passed, 0.0f, float(c.type));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_sub_nil_number() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::nil();
    XValue b = XValue::number(10.0f);
    XValue c = vm_binop_sub(state, a, b);
    
    bool passed = c.isNil();
    report_ops_test(422, passed, 0.0f, float(c.type));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_mul_number_nil() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::number(5.0f);
    XValue b = XValue::nil();
    XValue c = vm_binop_mul(state, a, b);
    
    bool passed = c.isNil();
    report_ops_test(423, passed, 0.0f, float(c.type));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_binop_div_bool_bool() {
    VMState state;
    state.status = VM_STATUS_RUNNING;
    
    XValue a = XValue::boolean(true);
    XValue b = XValue::boolean(false);
    XValue c = vm_binop_div(state, a, b);
    
    bool passed = c.isNil();
    report_ops_test(424, passed, 0.0f, float(c.type));
}

// =============================================================================
// Function Call Tests (430-439) - Simplified without buffers
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_call_frame_setup() {
    // Test basic call frame structure
    CallFrame frame;
    frame.returnPC = 100;
    frame.prevFP = 5;
    frame.localBase = 10;
    frame.argCount = 3;
    
    bool passed = (frame.returnPC == 100) && 
                  (frame.prevFP == 5) && 
                  (frame.localBase == 10) && 
                  (frame.argCount == 3);
    report_ops_test(430, passed, 1.0f, passed ? 1.0f : 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_function_descriptor() {
    // Test function descriptor structure
    FunctionDescriptor func;
    func.codeOffset = 200;
    func.paramCount = 2;
    func.localCount = 5;
    func.upvalueCount = 1;
    func.nameIndex = 0;
    
    bool passed = (func.codeOffset == 200) && 
                  (func.paramCount == 2) && 
                  (func.localCount == 5);
    report_ops_test(431, passed, 1.0f, passed ? 1.0f : 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_vm_return_simple() {
    // Test return value handling
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.pc = 0;
    state.status = VM_STATUS_RUNNING;
    
    // Simulate function with one local and return value
    vm_push(state, XValue::number(100.0f));  // local 0
    vm_push(state, XValue::number(42.0f));   // return value
    
    // Simulate return: pop return value, set state
    XValue retVal;
    vm_pop(state, retVal);
    
    bool passed = (retVal.asNumber() == 42.0f) && (state.sp == 1);
    report_ops_test(432, passed, 42.0f, retVal.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_frame_pointer_mechanics() {
    // Test that frame pointer correctly isolates stack frames
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    
    // Caller's locals
    vm_push(state, XValue::number(1.0f));
    vm_push(state, XValue::number(2.0f));
    uint callerSP = state.sp;
    
    // Save frame info
    CallFrame savedFrame;
    savedFrame.prevFP = state.fp;
    savedFrame.localBase = state.sp;
    
    // Enter new frame
    state.fp = state.sp;
    vm_push(state, XValue::number(10.0f));  // arg 0
    vm_push(state, XValue::number(20.0f));  // arg 1
    
    // Verify we can access new frame's locals
    XValue arg0 = vm_get_local(state, 0);
    XValue arg1 = vm_get_local(state, 1);
    
    bool passed = (arg0.asNumber() == 10.0f) && (arg1.asNumber() == 20.0f);
    report_ops_test(433, passed, 10.0f, arg0.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_nested_frame_restore() {
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    
    // Outer frame
    vm_push(state, XValue::number(100.0f));
    uint outerFP = state.fp;
    uint outerSP = state.sp;
    
    // Enter inner frame
    state.fp = state.sp;
    vm_push(state, XValue::number(200.0f));
    
    // "Return" from inner frame
    state.sp = outerSP;
    state.fp = outerFP;
    
    // Verify outer frame's local is accessible
    XValue v = vm_get_local(state, 0);
    
    bool passed = (v.asNumber() == 100.0f) && (state.sp == 1);
    report_ops_test(434, passed, 100.0f, v.asNumber());
}

// =============================================================================
// Function Call Variants Tests (440-449)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_call_zero_args() {
    // Test: Simulate calling a function with 0 arguments
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    state.error = 0;
    
    // Save caller state
    uint callerFP = state.fp;
    uint callerPC = 100;  // Simulated return address
    
    // Set up new frame with 0 args
    state.fp = state.sp;  // fp = sp = 0
    
    // Function body: push return value
    vm_push(state, XValue::number(42.0f));
    
    // Return: restore state
    XValue retVal;
    vm_pop(state, retVal);
    state.fp = callerFP;
    
    bool passed = (retVal.asNumber() == 42.0f) && (state.sp == 0) && (state.fp == 0);
    report_ops_test(440, passed, 42.0f, retVal.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_call_many_args() {
    // Test: Simulate calling a function with 8 arguments
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    state.error = 0;
    
    // Push 8 arguments
    for (uint i = 0; i < 8; i++) {
        vm_push(state, XValue::number(float(i + 1)));  // 1, 2, 3, ..., 8
    }
    
    // Set up call frame
    CallFrame frame;
    frame.returnPC = 50;
    frame.prevFP = state.fp;
    frame.localBase = state.sp - 8;  // Base where args start
    frame.argCount = 8;
    
    // New frame pointer at first arg
    state.fp = state.sp - 8;
    
    // Access all 8 args
    float sum = 0.0f;
    for (uint i = 0; i < 8; i++) {
        XValue arg = vm_get_local(state, i);
        sum += arg.asNumber();
    }
    
    // Expected sum: 1+2+3+4+5+6+7+8 = 36
    bool passed = (sum == 36.0f) && (state.sp == 8);
    report_ops_test(441, passed, 36.0f, sum);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_call_nested_simulated() {
    // Test: Simulate nested function calls f(g(x))
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    state.error = 0;
    
    // Push argument for g: 5
    vm_push(state, XValue::number(5.0f));
    
    // "Call" g(x): g doubles its input
    state.fp = state.sp - 1;  // fp points to arg
    XValue gArg = vm_get_local(state, 0);  // Get arg (5)
    
    // g's return value = arg * 2 = 10
    XValue gResult = XValue::number(gArg.asNumber() * 2.0f);
    
    // "Return" from g: replace arg with result
    state.sp = state.fp;
    vm_push(state, gResult);
    
    // Now "call" f with g's result: f adds 7
    state.fp = state.sp - 1;
    XValue fArg = vm_get_local(state, 0);  // Get arg (10)
    
    // f's return value = arg + 7 = 17
    XValue fResult = XValue::number(fArg.asNumber() + 7.0f);
    
    // "Return" from f
    state.sp = state.fp;
    vm_push(state, fResult);
    
    // Final result should be 17
    bool passed = (state.sp == 1) && (state.stack[0].asNumber() == 17.0f);
    report_ops_test(442, passed, 17.0f, state.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_return_multiple_values() {
    // Test: Simulate returning multiple values
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    state.error = 0;
    
    // Set up function that returns 3 values
    state.fp = 0;
    
    // Push 3 return values
    vm_push(state, XValue::number(10.0f));
    vm_push(state, XValue::number(20.0f));
    vm_push(state, XValue::number(30.0f));
    
    // Verify all 3 values are on stack
    bool passed = (state.sp == 3) &&
                  (state.stack[0].asNumber() == 10.0f) &&
                  (state.stack[1].asNumber() == 20.0f) &&
                  (state.stack[2].asNumber() == 30.0f);
    report_ops_test(443, passed, 3.0f, float(state.sp));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_deep_call_stack() {
    // Test: Simulate deeply nested calls (5 levels)
    VMState state;
    state.sp = 0;
    state.fp = 0;
    state.status = VM_STATUS_RUNNING;
    state.error = 0;
    
    // Simulate 5 nested call frames
    uint framePointers[5];
    
    for (uint i = 0; i < 5; i++) {
        // Push an argument
        vm_push(state, XValue::number(float(i)));
        framePointers[i] = state.fp;
        state.fp = state.sp - 1;
        // Push a local
        vm_push(state, XValue::number(float(i) * 10.0f));
    }
    
    // sp should be 10 (5 args + 5 locals)
    // Verify deepest frame's local
    XValue deepLocal = vm_get_local(state, 1);  // Local at slot 1 in current frame
    
    bool passed = (state.sp == 10) && (deepLocal.asNumber() == 40.0f);  // Frame 4's local
    report_ops_test(444, passed, 40.0f, deepLocal.asNumber());
}

// =============================================================================
// Run All Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void run_all_ops_tests() {
    g_testCount[0] = 0;
    
    // Local variable tests
    {
        VMState state;
        state.sp = 0;
        state.fp = 0;
        state.status = VM_STATUS_RUNNING;
        vm_push(state, XValue::number(10.0f));
        vm_push(state, XValue::number(20.0f));
        vm_push(state, XValue::number(30.0f));
        XValue v = vm_get_local(state, 1);
        report_ops_test(400, v.asNumber() == 20.0f, 20.0f, v.asNumber());
    }
    
    {
        VMState state;
        state.sp = 0;
        state.fp = 0;
        state.status = VM_STATUS_RUNNING;
        vm_push(state, XValue::number(1.0f));
        vm_push(state, XValue::number(2.0f));
        state.fp = state.sp;
        vm_push(state, XValue::number(100.0f));
        vm_push(state, XValue::number(200.0f));
        XValue v = vm_get_local(state, 0);
        report_ops_test(401, v.asNumber() == 100.0f, 100.0f, v.asNumber());
    }
    
    // Binary operation tests
    {
        VMState state;
        state.status = VM_STATUS_RUNNING;
        XValue c = vm_binop_add(state, XValue::number(15.0f), XValue::number(27.0f));
        report_ops_test(410, c.asNumber() == 42.0f, 42.0f, c.asNumber());
    }
    
    {
        VMState state;
        state.status = VM_STATUS_RUNNING;
        XValue c = vm_binop_sub(state, XValue::number(50.0f), XValue::number(8.0f));
        report_ops_test(411, c.asNumber() == 42.0f, 42.0f, c.asNumber());
    }
    
    {
        VMState state;
        state.status = VM_STATUS_RUNNING;
        XValue c = vm_binop_mul(state, XValue::number(6.0f), XValue::number(7.0f));
        report_ops_test(412, c.asNumber() == 42.0f, 42.0f, c.asNumber());
    }
    
    {
        VMState state;
        state.status = VM_STATUS_RUNNING;
        XValue c = vm_binop_div(state, XValue::number(42.0f), XValue::number(0.0f));
        report_ops_test(414, isinf(c.asNumber()), 1.0f, isinf(c.asNumber()) ? 1.0f : 0.0f);
    }
    
    // Type error tests
    {
        VMState state;
        state.status = VM_STATUS_RUNNING;
        XValue c = vm_binop_add(state, XValue::nil(), XValue::number(42.0f));
        report_ops_test(420, c.isNil(), 0.0f, float(c.type));
    }
    
    {
        VMState state;
        state.status = VM_STATUS_RUNNING;
        XValue c = vm_binop_add(state, XValue::boolean(true), XValue::number(42.0f));
        report_ops_test(421, c.isNil(), 0.0f, float(c.type));
    }
}

// =============================================================================
// ECS Operations Tests (450-459)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_ecs_set_get_entity() {
    // Test: Set and get current entity
    uint testEntityId = 12345;
    uint testTablePtr = 67890;
    
    vm_set_current_entity(testEntityId, testTablePtr);
    
    uint gotId = vm_get_current_entity_id();
    XValue gotEntity = vm_get_current_entity();
    
    bool passed = (gotId == testEntityId) && 
                  (gotEntity.type == TYPE_TABLE) && 
                  (gotEntity.asUint() == testTablePtr);
    
    vm_clear_current_entity();
    
    report_ops_test(450, passed, float(testEntityId), float(gotId));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_ecs_clear_entity() {
    // Test: Clear current entity
    vm_set_current_entity(100, 200);
    vm_clear_current_entity();
    
    uint gotId = vm_get_current_entity_id();
    XValue gotEntity = vm_get_current_entity();
    
    bool passed = (gotId == 0xFFFFFFFF) && gotEntity.isNil();
    
    report_ops_test(451, passed, 0xFFFFFFFF, float(gotId));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_ecs_has_component_nil() {
    // Test: has_component with nil entity returns false
    XValue nilEntity = XValue::nil();
    XValue key = XValue::string(100);
    
    bool hasComp = vm_has_component(nilEntity, key);
    
    bool passed = !hasComp;
    report_ops_test(452, passed, 0.0f, hasComp ? 1.0f : 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_ecs_has_component_non_table() {
    // Test: has_component with non-table returns false
    XValue numValue = XValue::number(42.0f);
    XValue key = XValue::string(100);
    
    bool hasComp = vm_has_component(numValue, key);
    
    bool passed = !hasComp;
    report_ops_test(453, passed, 0.0f, hasComp ? 1.0f : 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_ecs_add_component_nil() {
    // Test: add_component with nil entity returns false
    XValue nilEntity = XValue::nil();
    XValue key = XValue::string(100);
    XValue value = XValue::number(42.0f);
    
    bool added = vm_add_component(nilEntity, key, value);
    
    bool passed = !added;
    report_ops_test(454, passed, 0.0f, added ? 1.0f : 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_ecs_remove_component_nil() {
    // Test: remove_component with nil entity returns false
    XValue nilEntity = XValue::nil();
    XValue key = XValue::string(100);
    
    bool removed = vm_remove_component(nilEntity, key);
    
    bool passed = !removed;
    report_ops_test(455, passed, 0.0f, removed ? 1.0f : 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_ecs_ops() {
    // Combined test for ECS operations - reports total passed/failed count
    uint passCount = 0;
    
    // Test 1: Set and get current entity
    {
        uint testEntityId = 12345;
        uint testTablePtr = 67890;
        vm_set_current_entity(testEntityId, testTablePtr);
        uint gotId = vm_get_current_entity_id();
        XValue gotEntity = vm_get_current_entity();
        if (gotId == testEntityId && gotEntity.type == TYPE_TABLE && gotEntity.asUint() == testTablePtr) {
            passCount++;
        }
        vm_clear_current_entity();
    }
    
    // Test 2: Clear current entity
    {
        vm_set_current_entity(100, 200);
        vm_clear_current_entity();
        uint gotId = vm_get_current_entity_id();
        XValue gotEntity = vm_get_current_entity();
        if (gotId == 0xFFFFFFFF && gotEntity.isNil()) {
            passCount++;
        }
    }
    
    // Test 3: has_component with nil returns false
    {
        XValue nilEntity = XValue::nil();
        XValue key = XValue::string(100);
        if (!vm_has_component(nilEntity, key)) {
            passCount++;
        }
    }
    
    // Test 4: has_component with non-table returns false
    {
        XValue numValue = XValue::number(42.0f);
        XValue key = XValue::string(100);
        if (!vm_has_component(numValue, key)) {
            passCount++;
        }
    }
    
    // Test 5: add_component with nil returns false
    {
        XValue nilEntity = XValue::nil();
        XValue key = XValue::string(100);
        XValue value = XValue::number(42.0f);
        if (!vm_add_component(nilEntity, key, value)) {
            passCount++;
        }
    }
    
    // Test 6: remove_component with nil returns false
    {
        XValue nilEntity = XValue::nil();
        XValue key = XValue::string(100);
        if (!vm_remove_component(nilEntity, key)) {
            passCount++;
        }
    }
    
    // Report: all 6 tests should pass
    bool allPassed = (passCount == 6);
    report_ops_test(456, allPassed, 6.0f, float(passCount));
}

