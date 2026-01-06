// XScript VM Bytecode Execution Tests (32-bit)
// Tests for complete VM instruction execution with hand-crafted bytecode

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
// Opcodes (subset for testing)
// =============================================================================

static const uint OP_NOP        = 0x00;
static const uint OP_PUSH_NIL   = 0x01;
static const uint OP_PUSH_TRUE  = 0x02;
static const uint OP_PUSH_FALSE = 0x03;
static const uint OP_PUSH_NUM   = 0x04;
static const uint OP_POP        = 0x06;
static const uint OP_DUP        = 0x07;
static const uint OP_SWAP       = 0x08;

static const uint OP_GET_LOCAL  = 0x10;
static const uint OP_SET_LOCAL  = 0x11;
static const uint OP_GET_GLOBAL = 0x12;
static const uint OP_SET_GLOBAL = 0x13;

static const uint OP_ADD = 0x20;
static const uint OP_SUB = 0x21;
static const uint OP_MUL = 0x22;
static const uint OP_DIV = 0x23;
static const uint OP_MOD = 0x24;
static const uint OP_NEG = 0x25;

static const uint OP_EQ  = 0x30;
static const uint OP_NE  = 0x31;
static const uint OP_LT  = 0x32;
static const uint OP_LE  = 0x33;
static const uint OP_GT  = 0x34;
static const uint OP_GE  = 0x35;
static const uint OP_NOT = 0x36;

static const uint OP_JMP        = 0x40;
static const uint OP_JMP_IF     = 0x41;
static const uint OP_JMP_IF_NOT = 0x42;

static const uint OP_HALT = 0xFF;

// =============================================================================
// VM Status and Error Codes
// =============================================================================

static const uint VM_STATUS_RUNNING   = 0;
static const uint VM_STATUS_ERROR     = 3;
static const uint VM_STATUS_COMPLETED = 4;

static const uint ERR_NONE            = 0;
static const uint ERR_STACK_OVERFLOW  = 1;
static const uint ERR_STACK_UNDERFLOW = 2;
static const uint ERR_TYPE_ERROR      = 3;
static const uint ERR_INVALID_OPCODE  = 5;

static const uint VM_STACK_SIZE = 64;

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
    
    bool isNil() { return type == TYPE_NIL; }
    bool isTruthy() {
        if (type == TYPE_NIL) return false;
        if (type == TYPE_BOOL) return data != 0;
        return true;
    }
    bool isFalsy() { return !isTruthy(); }
    float asNumber() { return asfloat(data); }
};

// =============================================================================
// Simple VM State (for isolated testing)
// =============================================================================

struct SimpleVM {
    XValue stack[VM_STACK_SIZE];
    uint sp;
    uint pc;
    uint status;
    uint error;
    
    // Bytecode stored inline (max 64 bytes for testing)
    uint bytecode[16];
    uint bytecodeLen;
    
    // Constants stored inline (max 16)
    XValue constants[16];
    uint constantCount;
    
    // Globals stored inline (max 16)
    XValue globals[16];
};

// =============================================================================
// Simple VM Operations
// =============================================================================

uint read_byte(SimpleVM vm) {
    uint wordIdx = vm.pc / 4;
    uint byteIdx = vm.pc % 4;
    return (vm.bytecode[wordIdx] >> (byteIdx * 8)) & 0xFF;
}

uint read_uint16_at(SimpleVM vm, uint pc) {
    uint wordIdx = pc / 4;
    uint byteIdx = pc % 4;
    uint lo = (vm.bytecode[wordIdx] >> (byteIdx * 8)) & 0xFF;
    
    uint nextPc = pc + 1;
    uint nextWordIdx = nextPc / 4;
    uint nextByteIdx = nextPc % 4;
    uint hi = (vm.bytecode[nextWordIdx] >> (nextByteIdx * 8)) & 0xFF;
    
    return lo | (hi << 8);
}

bool vm_push(inout SimpleVM vm, XValue v) {
    if (vm.sp >= VM_STACK_SIZE) {
        vm.status = VM_STATUS_ERROR;
        vm.error = ERR_STACK_OVERFLOW;
        return false;
    }
    vm.stack[vm.sp] = v;
    vm.sp++;
    return true;
}

bool vm_pop(inout SimpleVM vm, out XValue v) {
    if (vm.sp == 0) {
        vm.status = VM_STATUS_ERROR;
        vm.error = ERR_STACK_UNDERFLOW;
        v = XValue::nil();
        return false;
    }
    vm.sp--;
    v = vm.stack[vm.sp];
    return true;
}

XValue vm_peek(SimpleVM vm, uint offset) {
    if (vm.sp <= offset) return XValue::nil();
    return vm.stack[vm.sp - 1 - offset];
}

// =============================================================================
// VM Step Function
// =============================================================================

void vm_step(inout SimpleVM vm) {
    if (vm.status != VM_STATUS_RUNNING || vm.pc >= vm.bytecodeLen * 4) {
        return;
    }
    
    uint opcode = read_byte(vm);
    vm.pc++;
    
    XValue a, b, result;
    uint operand8, operand16;
    
    switch (opcode) {
        case OP_NOP:
            break;
            
        case OP_PUSH_NIL:
            vm_push(vm, XValue::nil());
            break;
            
        case OP_PUSH_TRUE:
            vm_push(vm, XValue::boolean(true));
            break;
            
        case OP_PUSH_FALSE:
            vm_push(vm, XValue::boolean(false));
            break;
            
        case OP_PUSH_NUM:
            operand16 = read_uint16_at(vm, vm.pc);
            vm.pc += 2;
            if (operand16 < 16) {
                vm_push(vm, vm.constants[operand16]);
            } else {
                vm_push(vm, XValue::nil());
            }
            break;
            
        case OP_POP:
            vm_pop(vm, a);
            break;
            
        case OP_DUP:
            a = vm_peek(vm, 0);
            vm_push(vm, a);
            break;
            
        case OP_SWAP:
            if (vm.sp >= 2) {
                a = vm.stack[vm.sp - 1];
                vm.stack[vm.sp - 1] = vm.stack[vm.sp - 2];
                vm.stack[vm.sp - 2] = a;
            }
            break;
            
        case OP_GET_GLOBAL:
            operand16 = read_uint16_at(vm, vm.pc);
            vm.pc += 2;
            if (operand16 < 16) {
                vm_push(vm, vm.globals[operand16]);
            }
            break;
            
        case OP_SET_GLOBAL:
            operand16 = read_uint16_at(vm, vm.pc);
            vm.pc += 2;
            vm_pop(vm, a);
            if (operand16 < 16) {
                vm.globals[operand16] = a;
            }
            break;
            
        case OP_ADD:
            vm_pop(vm, b);
            vm_pop(vm, a);
            if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
                result = XValue::number(a.asNumber() + b.asNumber());
            } else {
                result = XValue::nil();
            }
            vm_push(vm, result);
            break;
            
        case OP_SUB:
            vm_pop(vm, b);
            vm_pop(vm, a);
            if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
                result = XValue::number(a.asNumber() - b.asNumber());
            } else {
                result = XValue::nil();
            }
            vm_push(vm, result);
            break;
            
        case OP_MUL:
            vm_pop(vm, b);
            vm_pop(vm, a);
            if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
                result = XValue::number(a.asNumber() * b.asNumber());
            } else {
                result = XValue::nil();
            }
            vm_push(vm, result);
            break;
            
        case OP_DIV:
            vm_pop(vm, b);
            vm_pop(vm, a);
            if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
                result = XValue::number(a.asNumber() / b.asNumber());
            } else {
                result = XValue::nil();
            }
            vm_push(vm, result);
            break;
            
        case OP_NEG:
            vm_pop(vm, a);
            if (a.type == TYPE_NUMBER) {
                result = XValue::number(-a.asNumber());
            } else {
                result = XValue::nil();
            }
            vm_push(vm, result);
            break;
            
        case OP_EQ:
            vm_pop(vm, b);
            vm_pop(vm, a);
            if (a.type != b.type) {
                result = XValue::boolean(false);
            } else if (a.type == TYPE_NUMBER) {
                result = XValue::boolean(a.asNumber() == b.asNumber());
            } else {
                result = XValue::boolean(a.data == b.data);
            }
            vm_push(vm, result);
            break;
            
        case OP_LT:
            vm_pop(vm, b);
            vm_pop(vm, a);
            if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
                result = XValue::boolean(a.asNumber() < b.asNumber());
            } else {
                result = XValue::boolean(false);
            }
            vm_push(vm, result);
            break;
            
        case OP_NOT:
            vm_pop(vm, a);
            result = XValue::boolean(a.isFalsy());
            vm_push(vm, result);
            break;
            
        case OP_JMP:
            {
                int offset = int(read_uint16_at(vm, vm.pc));
                if (offset >= 0x8000) offset -= 0x10000;
                vm.pc += 2;
                vm.pc = uint(int(vm.pc) + offset);
            }
            break;
            
        case OP_JMP_IF:
            {
                int offset = int(read_uint16_at(vm, vm.pc));
                if (offset >= 0x8000) offset -= 0x10000;
                vm.pc += 2;
                vm_pop(vm, a);
                if (a.isTruthy()) {
                    vm.pc = uint(int(vm.pc) + offset);
                }
            }
            break;
            
        case OP_JMP_IF_NOT:
            {
                int offset = int(read_uint16_at(vm, vm.pc));
                if (offset >= 0x8000) offset -= 0x10000;
                vm.pc += 2;
                vm_pop(vm, a);
                if (a.isFalsy()) {
                    vm.pc = uint(int(vm.pc) + offset);
                }
            }
            break;
            
        case OP_HALT:
            vm.status = VM_STATUS_COMPLETED;
            break;
            
        default:
            vm.status = VM_STATUS_ERROR;
            vm.error = ERR_INVALID_OPCODE;
            break;
    }
}

// Run VM until completion or error
void vm_run(inout SimpleVM vm, uint maxSteps) {
    for (uint i = 0; i < maxSteps && vm.status == VM_STATUS_RUNNING; i++) {
        vm_step(vm);
    }
}

// =============================================================================
// Test Result Buffer
// =============================================================================

struct TestResult {
    uint testId;
    uint passed;
    uint vmStatus;
    uint stackPointer;
    float expectedValue;
    float actualValue;
};

RWStructuredBuffer<TestResult> g_testResults;
RWStructuredBuffer<uint> g_testCount;

void report_vm_test(uint testId, bool passed, uint status, uint sp, float expected, float actual) {
    uint idx;
    InterlockedAdd(g_testCount[0], 1, idx);
    
    TestResult result;
    result.testId = testId;
    result.passed = passed ? 1 : 0;
    result.vmStatus = status;
    result.stackPointer = sp;
    result.expectedValue = expected;
    result.actualValue = actual;
    g_testResults[idx] = result;
}

// =============================================================================
// Helper to pack bytecode
// =============================================================================

uint pack_bytes(uint b0, uint b1, uint b2, uint b3) {
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

// =============================================================================
// VM Execution Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_push_nil_halt() {
    // Bytecode: PUSH_NIL, HALT
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.bytecode[0] = pack_bytes(OP_PUSH_NIL, OP_HALT, 0, 0);
    vm.bytecodeLen = 1;
    
    vm_run(vm, 10);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].type == TYPE_NIL);
    report_vm_test(300, passed, vm.status, vm.sp, 0.0f, float(vm.stack[0].type));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_push_true_false() {
    // Bytecode: PUSH_TRUE, PUSH_FALSE, HALT
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.bytecode[0] = pack_bytes(OP_PUSH_TRUE, OP_PUSH_FALSE, OP_HALT, 0);
    vm.bytecodeLen = 1;
    
    vm_run(vm, 10);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 2) && 
                  (vm.stack[0].data == 1) &&  // true
                  (vm.stack[1].data == 0);     // false
    report_vm_test(301, passed, vm.status, vm.sp, 1.0f, float(vm.stack[0].data));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_push_num_constant() {
    // Bytecode: PUSH_NUM 0x0000, HALT  (push constant[0])
    // constant[0] = 42.0
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_HALT);
    vm.bytecodeLen = 1;
    vm.constants[0] = XValue::number(42.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 10);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].asNumber() == 42.0f);
    report_vm_test(302, passed, vm.status, vm.sp, 42.0f, vm.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_add_two_numbers() {
    // Bytecode: PUSH_NUM 0, PUSH_NUM 1, ADD, HALT
    // constant[0] = 10.0, constant[1] = 20.0
    // Expected result: 30.0
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_PUSH_NUM);
    vm.bytecode[1] = pack_bytes(0x01, 0x00, OP_ADD, OP_HALT);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(10.0f);
    vm.constants[1] = XValue::number(20.0f);
    vm.constantCount = 2;
    
    vm_run(vm, 20);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].asNumber() == 30.0f);
    report_vm_test(303, passed, vm.status, vm.sp, 30.0f, vm.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_arithmetic_expression() {
    // Test: (5 + 3) * 2 = 16
    // Bytecode: PUSH_NUM 0, PUSH_NUM 1, ADD, PUSH_NUM 2, MUL, HALT
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_PUSH_NUM);
    vm.bytecode[1] = pack_bytes(0x01, 0x00, OP_ADD, OP_PUSH_NUM);
    vm.bytecode[2] = pack_bytes(0x02, 0x00, OP_MUL, OP_HALT);
    vm.bytecodeLen = 3;
    vm.constants[0] = XValue::number(5.0f);
    vm.constants[1] = XValue::number(3.0f);
    vm.constants[2] = XValue::number(2.0f);
    vm.constantCount = 3;
    
    vm_run(vm, 30);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].asNumber() == 16.0f);
    report_vm_test(304, passed, vm.status, vm.sp, 16.0f, vm.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_set_get_global() {
    // Test: global[0] = 100; push global[0]
    // Bytecode: PUSH_NUM 0, SET_GLOBAL 0, GET_GLOBAL 0, HALT
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_SET_GLOBAL);
    vm.bytecode[1] = pack_bytes(0x00, 0x00, OP_GET_GLOBAL, 0x00);
    vm.bytecode[2] = pack_bytes(0x00, OP_HALT, 0, 0);
    vm.bytecodeLen = 3;
    vm.constants[0] = XValue::number(100.0f);
    vm.constantCount = 1;
    
    for (uint i = 0; i < 16; i++) {
        vm.globals[i] = XValue::nil();
    }
    
    vm_run(vm, 30);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].asNumber() == 100.0f) &&
                  (vm.globals[0].asNumber() == 100.0f);
    report_vm_test(305, passed, vm.status, vm.sp, 100.0f, vm.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_comparison_eq() {
    // Test: 10 == 10 should be true
    // Bytecode: PUSH_NUM 0, PUSH_NUM 0, EQ, HALT
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_PUSH_NUM);
    vm.bytecode[1] = pack_bytes(0x00, 0x00, OP_EQ, OP_HALT);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(10.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 20);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].type == TYPE_BOOL) &&
                  (vm.stack[0].data == 1);  // true
    report_vm_test(306, passed, vm.status, vm.sp, 1.0f, float(vm.stack[0].data));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_comparison_lt() {
    // Test: 5 < 10 should be true
    // Bytecode: PUSH_NUM 0, PUSH_NUM 1, LT, HALT
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_PUSH_NUM);
    vm.bytecode[1] = pack_bytes(0x01, 0x00, OP_LT, OP_HALT);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(5.0f);
    vm.constants[1] = XValue::number(10.0f);
    vm.constantCount = 2;
    
    vm_run(vm, 20);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].type == TYPE_BOOL) &&
                  (vm.stack[0].data == 1);  // true
    report_vm_test(307, passed, vm.status, vm.sp, 1.0f, float(vm.stack[0].data));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_negation() {
    // Test: -42 = -42
    // Bytecode: PUSH_NUM 0, NEG, HALT
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_NEG);
    vm.bytecode[1] = pack_bytes(OP_HALT, 0, 0, 0);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(42.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 20);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].asNumber() == -42.0f);
    report_vm_test(308, passed, vm.status, vm.sp, -42.0f, vm.stack[0].asNumber());
}

// =============================================================================
// Run All Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void run_all_vm_tests() {
    g_testCount[0] = 0;
    
    // Test 300: PUSH_NIL, HALT
    {
        SimpleVM vm;
        vm.sp = 0;
        vm.pc = 0;
        vm.status = VM_STATUS_RUNNING;
        vm.bytecode[0] = pack_bytes(OP_PUSH_NIL, OP_HALT, 0, 0);
        vm.bytecodeLen = 1;
        
        vm_run(vm, 10);
        
        bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                      (vm.sp == 1) && 
                      (vm.stack[0].type == TYPE_NIL);
        report_vm_test(300, passed, vm.status, vm.sp, 0.0f, float(vm.stack[0].type));
    }
    
    // Test 301: PUSH_TRUE, PUSH_FALSE
    {
        SimpleVM vm;
        vm.sp = 0;
        vm.pc = 0;
        vm.status = VM_STATUS_RUNNING;
        vm.bytecode[0] = pack_bytes(OP_PUSH_TRUE, OP_PUSH_FALSE, OP_HALT, 0);
        vm.bytecodeLen = 1;
        
        vm_run(vm, 10);
        
        bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                      (vm.sp == 2) && 
                      (vm.stack[0].data == 1) &&
                      (vm.stack[1].data == 0);
        report_vm_test(301, passed, vm.status, vm.sp, 1.0f, float(vm.stack[0].data));
    }
    
    // Test 302: PUSH_NUM constant
    {
        SimpleVM vm;
        vm.sp = 0;
        vm.pc = 0;
        vm.status = VM_STATUS_RUNNING;
        vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_HALT);
        vm.bytecodeLen = 1;
        vm.constants[0] = XValue::number(42.0f);
        vm.constantCount = 1;
        
        vm_run(vm, 10);
        
        bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                      (vm.sp == 1) && 
                      (vm.stack[0].asNumber() == 42.0f);
        report_vm_test(302, passed, vm.status, vm.sp, 42.0f, vm.stack[0].asNumber());
    }
    
    // Test 303: ADD two numbers
    {
        SimpleVM vm;
        vm.sp = 0;
        vm.pc = 0;
        vm.status = VM_STATUS_RUNNING;
        vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_PUSH_NUM);
        vm.bytecode[1] = pack_bytes(0x01, 0x00, OP_ADD, OP_HALT);
        vm.bytecodeLen = 2;
        vm.constants[0] = XValue::number(10.0f);
        vm.constants[1] = XValue::number(20.0f);
        vm.constantCount = 2;
        
        vm_run(vm, 20);
        
        bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                      (vm.sp == 1) && 
                      (vm.stack[0].asNumber() == 30.0f);
        report_vm_test(303, passed, vm.status, vm.sp, 30.0f, vm.stack[0].asNumber());
    }
    
    // Test 304: (5 + 3) * 2 = 16
    {
        SimpleVM vm;
        vm.sp = 0;
        vm.pc = 0;
        vm.status = VM_STATUS_RUNNING;
        vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_PUSH_NUM);
        vm.bytecode[1] = pack_bytes(0x01, 0x00, OP_ADD, OP_PUSH_NUM);
        vm.bytecode[2] = pack_bytes(0x02, 0x00, OP_MUL, OP_HALT);
        vm.bytecodeLen = 3;
        vm.constants[0] = XValue::number(5.0f);
        vm.constants[1] = XValue::number(3.0f);
        vm.constants[2] = XValue::number(2.0f);
        vm.constantCount = 3;
        
        vm_run(vm, 30);
        
        bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                      (vm.sp == 1) && 
                      (vm.stack[0].asNumber() == 16.0f);
        report_vm_test(304, passed, vm.status, vm.sp, 16.0f, vm.stack[0].asNumber());
    }
    
    // Test 306: 10 == 10
    {
        SimpleVM vm;
        vm.sp = 0;
        vm.pc = 0;
        vm.status = VM_STATUS_RUNNING;
        vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_PUSH_NUM);
        vm.bytecode[1] = pack_bytes(0x00, 0x00, OP_EQ, OP_HALT);
        vm.bytecodeLen = 2;
        vm.constants[0] = XValue::number(10.0f);
        vm.constantCount = 1;
        
        vm_run(vm, 20);
        
        bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                      (vm.sp == 1) && 
                      (vm.stack[0].type == TYPE_BOOL) &&
                      (vm.stack[0].data == 1);
        report_vm_test(306, passed, vm.status, vm.sp, 1.0f, float(vm.stack[0].data));
    }
    
    // Test 308: NEG 42 = -42
    {
        SimpleVM vm;
        vm.sp = 0;
        vm.pc = 0;
        vm.status = VM_STATUS_RUNNING;
        vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_NEG);
        vm.bytecode[1] = pack_bytes(OP_HALT, 0, 0, 0);
        vm.bytecodeLen = 2;
        vm.constants[0] = XValue::number(42.0f);
        vm.constantCount = 1;
        
        vm_run(vm, 20);
        
        bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                      (vm.sp == 1) && 
                      (vm.stack[0].asNumber() == -42.0f);
        report_vm_test(308, passed, vm.status, vm.sp, -42.0f, vm.stack[0].asNumber());
    }
}

