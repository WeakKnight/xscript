// XScript Stack Tests (32-bit)
// Tests for VM stack operations

// =============================================================================
// Type Constants
// =============================================================================

static const uint TYPE_NIL      = 0;
static const uint TYPE_BOOL     = 1;
static const uint TYPE_NUMBER   = 2;

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
    
    static XValue number(float n) {
        XValue v;
        v.type = TYPE_NUMBER;
        v.flags = 0;
        v.data = asuint(n);
        return v;
    }
    
    float asNumber() { return asfloat(data); }
    bool isNil() { return type == TYPE_NIL; }
};

// =============================================================================
// VM Constants and Structures
// =============================================================================

static const uint VM_STACK_SIZE = 256;

static const uint VM_STATUS_RUNNING      = 0;
static const uint VM_STATUS_ERROR        = 3;

static const uint ERR_NONE            = 0;
static const uint ERR_STACK_OVERFLOW  = 1;
static const uint ERR_STACK_UNDERFLOW = 2;

struct VMState {
    XValue stack[VM_STACK_SIZE];
    uint sp;
    uint fp;
    uint pc;
    uint status;
    uint error;
};

// =============================================================================
// Stack Operations
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

// DUP: duplicate top of stack
bool vm_dup(inout VMState state) {
    if (state.sp == 0) {
        state.status = VM_STATUS_ERROR;
        state.error = ERR_STACK_UNDERFLOW;
        return false;
    }
    if (state.sp >= VM_STACK_SIZE) {
        state.status = VM_STATUS_ERROR;
        state.error = ERR_STACK_OVERFLOW;
        return false;
    }
    state.stack[state.sp] = state.stack[state.sp - 1];
    state.sp++;
    return true;
}

// SWAP: swap top two elements
bool vm_swap(inout VMState state) {
    if (state.sp < 2) {
        state.status = VM_STATUS_ERROR;
        state.error = ERR_STACK_UNDERFLOW;
        return false;
    }
    XValue temp = state.stack[state.sp - 1];
    state.stack[state.sp - 1] = state.stack[state.sp - 2];
    state.stack[state.sp - 2] = temp;
    return true;
}

// =============================================================================
// Test Result Buffer
// =============================================================================

struct TestResult {
    uint testId;
    uint passed;
    uint stackPointer;
    float expectedValue;
    float actualValue;
};

RWStructuredBuffer<TestResult> g_testResults;
RWStructuredBuffer<uint> g_testCount;

void report_stack_test(uint testId, bool passed, uint sp, float expected, float actual) {
    uint idx;
    InterlockedAdd(g_testCount[0], 1, idx);
    
    TestResult result;
    result.testId = testId;
    result.passed = passed ? 1 : 0;
    result.stackPointer = sp;
    result.expectedValue = expected;
    result.actualValue = actual;
    g_testResults[idx] = result;
}

// =============================================================================
// Stack Push/Pop Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_push_single() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    
    bool ok = vm_push(state, XValue::number(42.0f));
    
    bool passed = ok && (state.sp == 1) && (state.stack[0].asNumber() == 42.0f);
    report_stack_test(200, passed, state.sp, 42.0f, state.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_push_pop() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    
    vm_push(state, XValue::number(100.0f));
    
    XValue v;
    bool ok = vm_pop(state, v);
    
    bool passed = ok && (state.sp == 0) && (v.asNumber() == 100.0f);
    report_stack_test(201, passed, state.sp, 100.0f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_multiple_push_pop() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    
    vm_push(state, XValue::number(1.0f));
    vm_push(state, XValue::number(2.0f));
    vm_push(state, XValue::number(3.0f));
    
    XValue v;
    vm_pop(state, v);  // Should be 3
    
    bool passed = (state.sp == 2) && (v.asNumber() == 3.0f);
    report_stack_test(202, passed, state.sp, 3.0f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_pop_empty_stack() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    state.error = ERR_NONE;
    
    XValue v;
    bool ok = vm_pop(state, v);
    
    bool passed = !ok && (state.error == ERR_STACK_UNDERFLOW);
    report_stack_test(203, passed, state.sp, 0.0f, float(state.error));
}

// =============================================================================
// Peek Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_peek_top() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    
    vm_push(state, XValue::number(10.0f));
    vm_push(state, XValue::number(20.0f));
    vm_push(state, XValue::number(30.0f));
    
    XValue v = vm_peek(state, 0);  // Top should be 30
    
    bool passed = (v.asNumber() == 30.0f);
    report_stack_test(210, passed, state.sp, 30.0f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_peek_offset() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    
    vm_push(state, XValue::number(10.0f));
    vm_push(state, XValue::number(20.0f));
    vm_push(state, XValue::number(30.0f));
    
    XValue v = vm_peek(state, 2);  // Bottom should be 10
    
    bool passed = (v.asNumber() == 10.0f);
    report_stack_test(211, passed, state.sp, 10.0f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_peek_out_of_bounds() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    
    vm_push(state, XValue::number(10.0f));
    
    XValue v = vm_peek(state, 5);  // Out of bounds
    
    bool passed = v.isNil();
    report_stack_test(212, passed, state.sp, 0.0f, float(v.type));
}

// =============================================================================
// DUP Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_dup() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    
    vm_push(state, XValue::number(42.0f));
    vm_dup(state);
    
    bool passed = (state.sp == 2) && 
                  (state.stack[0].asNumber() == 42.0f) && 
                  (state.stack[1].asNumber() == 42.0f);
    report_stack_test(220, passed, state.sp, 42.0f, state.stack[1].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_dup_empty() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    state.error = ERR_NONE;
    
    bool ok = vm_dup(state);
    
    bool passed = !ok && (state.error == ERR_STACK_UNDERFLOW);
    report_stack_test(221, passed, state.sp, 0.0f, float(state.error));
}

// =============================================================================
// SWAP Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_swap() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    
    vm_push(state, XValue::number(1.0f));
    vm_push(state, XValue::number(2.0f));
    vm_swap(state);
    
    bool passed = (state.stack[0].asNumber() == 2.0f) && 
                  (state.stack[1].asNumber() == 1.0f);
    report_stack_test(230, passed, state.sp, 2.0f, state.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_swap_insufficient() {
    VMState state;
    state.sp = 0;
    state.status = VM_STATUS_RUNNING;
    state.error = ERR_NONE;
    
    vm_push(state, XValue::number(1.0f));
    bool ok = vm_swap(state);
    
    bool passed = !ok && (state.error == ERR_STACK_UNDERFLOW);
    report_stack_test(231, passed, state.sp, 0.0f, float(state.error));
}

// =============================================================================
// Run All Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void run_all_stack_tests() {
    g_testCount[0] = 0;
    
    // Push/Pop tests
    {
        VMState state;
        state.sp = 0;
        state.status = VM_STATUS_RUNNING;
        
        vm_push(state, XValue::number(42.0f));
        bool passed = (state.sp == 1) && (state.stack[0].asNumber() == 42.0f);
        report_stack_test(200, passed, state.sp, 42.0f, state.stack[0].asNumber());
    }
    
    {
        VMState state;
        state.sp = 0;
        state.status = VM_STATUS_RUNNING;
        
        vm_push(state, XValue::number(100.0f));
        XValue v;
        vm_pop(state, v);
        bool passed = (state.sp == 0) && (v.asNumber() == 100.0f);
        report_stack_test(201, passed, state.sp, 100.0f, v.asNumber());
    }
    
    {
        VMState state;
        state.sp = 0;
        state.status = VM_STATUS_RUNNING;
        
        vm_push(state, XValue::number(1.0f));
        vm_push(state, XValue::number(2.0f));
        vm_push(state, XValue::number(3.0f));
        
        XValue v;
        vm_pop(state, v);
        bool passed = (state.sp == 2) && (v.asNumber() == 3.0f);
        report_stack_test(202, passed, state.sp, 3.0f, v.asNumber());
    }
    
    {
        VMState state;
        state.sp = 0;
        state.status = VM_STATUS_RUNNING;
        state.error = ERR_NONE;
        
        XValue v;
        vm_pop(state, v);
        bool passed = (state.error == ERR_STACK_UNDERFLOW);
        report_stack_test(203, passed, state.sp, float(ERR_STACK_UNDERFLOW), float(state.error));
    }
    
    // Peek tests
    {
        VMState state;
        state.sp = 0;
        state.status = VM_STATUS_RUNNING;
        
        vm_push(state, XValue::number(10.0f));
        vm_push(state, XValue::number(20.0f));
        vm_push(state, XValue::number(30.0f));
        
        XValue v = vm_peek(state, 0);
        bool passed = (v.asNumber() == 30.0f);
        report_stack_test(210, passed, state.sp, 30.0f, v.asNumber());
    }
    
    {
        VMState state;
        state.sp = 0;
        state.status = VM_STATUS_RUNNING;
        
        vm_push(state, XValue::number(10.0f));
        vm_push(state, XValue::number(20.0f));
        vm_push(state, XValue::number(30.0f));
        
        XValue v = vm_peek(state, 2);
        bool passed = (v.asNumber() == 10.0f);
        report_stack_test(211, passed, state.sp, 10.0f, v.asNumber());
    }
    
    // DUP test
    {
        VMState state;
        state.sp = 0;
        state.status = VM_STATUS_RUNNING;
        
        vm_push(state, XValue::number(42.0f));
        vm_dup(state);
        
        bool passed = (state.sp == 2) && 
                      (state.stack[0].asNumber() == 42.0f) && 
                      (state.stack[1].asNumber() == 42.0f);
        report_stack_test(220, passed, state.sp, 42.0f, state.stack[1].asNumber());
    }
    
    // SWAP test
    {
        VMState state;
        state.sp = 0;
        state.status = VM_STATUS_RUNNING;
        
        vm_push(state, XValue::number(1.0f));
        vm_push(state, XValue::number(2.0f));
        vm_swap(state);
        
        bool passed = (state.stack[0].asNumber() == 2.0f) && 
                      (state.stack[1].asNumber() == 1.0f);
        report_stack_test(230, passed, state.sp, 2.0f, state.stack[0].asNumber());
    }
}

