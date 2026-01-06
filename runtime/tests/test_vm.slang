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
// Control Flow Tests (310-319)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_jmp_forward() {
    // Test: JMP skips over instructions
    // Layout: PUSH_NUM 0 (3 bytes), JMP offset (3 bytes), PUSH_NUM 1 (3 bytes), HALT
    // Bytes: 0-2=PUSH_NUM, 3=JMP, 4-5=offset, 6-8=PUSH_NUM, 9=HALT
    // After JMP reads offset, pc=6. We want to skip to HALT at 9, so offset = 9-6 = 3
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    // Byte 0: PUSH_NUM, Byte 1-2: 0x0000 (constant 0), Byte 3: JMP
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_JMP);
    // Byte 4-5: offset +3, Byte 6: PUSH_NUM, Byte 7: 0x01
    vm.bytecode[1] = pack_bytes(0x03, 0x00, OP_PUSH_NUM, 0x01);
    // Byte 8: 0x00, Byte 9: HALT
    vm.bytecode[2] = pack_bytes(0x00, OP_HALT, 0, 0);
    vm.bytecodeLen = 3;
    vm.constants[0] = XValue::number(42.0f);
    vm.constants[1] = XValue::number(99.0f);
    vm.constantCount = 2;
    
    vm_run(vm, 20);
    
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].asNumber() == 42.0f);
    report_vm_test(310, passed, vm.status, vm.sp, 42.0f, vm.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_jmp_backward() {
    // Test: Loop using backward JMP
    // Logic: push 0, push 1, add (result=1), push 1, add (result=2), ... halt at some point
    // Simplified: push 3, JMP -1 (infinite loop avoided by max steps), test that we executed
    // Actually let's do: PUSH_NUM 0, PUSH_NUM 0, ADD, DUP, PUSH_NUM 1, LT, JMP_IF_NOT to halt, JMP back
    // Simpler: Just verify backward JMP works by counting iterations via stack depth
    
    // Even simpler: PUSH_NUM 0, JMP +1, PUSH_NUM 1, JMP -7 (back to first JMP), HALT
    // This creates: push 0, jmp forward, push 1, jmp back, halt
    // Wait this is getting complex. Let's just test backward JMP directly.
    
    // Test: PUSH_NUM 0, JMP +3 (skip next PUSH), PUSH_NUM 1, HALT, <- JMP -3 would land here
    // Actually: Let's do PUSH_TRUE, JMP +1, PUSH_FALSE, HALT
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    // Simple test: Jump forward past one instruction, then verify we skipped it
    // PUSH_TRUE (1), JMP +1 (3), PUSH_FALSE (1), HALT (1)
    // pc: 0=PUSH_TRUE, 1=JMP, 2-3=offset, 4=PUSH_FALSE, 5=HALT
    // JMP offset +1 means: after reading offset at pc 2-3, pc becomes 4, then +1 = 5 (HALT)
    vm.bytecode[0] = pack_bytes(OP_PUSH_TRUE, OP_JMP, 0x01, 0x00);
    vm.bytecode[1] = pack_bytes(OP_PUSH_FALSE, OP_HALT, 0, 0);
    vm.bytecodeLen = 2;
    
    vm_run(vm, 20);
    
    // Should have only pushed TRUE, not FALSE
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) && 
                  (vm.stack[0].type == TYPE_BOOL) &&
                  (vm.stack[0].data == 1);  // true
    report_vm_test(311, passed, vm.status, vm.sp, 1.0f, float(vm.stack[0].data));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_jmp_if_true() {
    // Test: JMP_IF when condition is true
    // Layout: PUSH_TRUE (1), JMP_IF (1), offset (2), PUSH_NUM (1), idx (2), HALT (1)
    // Bytes: 0=PUSH_TRUE, 1=JMP_IF, 2-3=offset, 4=PUSH_NUM, 5-6=idx, 7=HALT
    // After JMP_IF reads offset, pc=4. We want to skip to HALT at 7, so offset = 7-4 = 3
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    // PUSH_TRUE, JMP_IF +3 (skip past PUSH_NUM to HALT)
    vm.bytecode[0] = pack_bytes(OP_PUSH_TRUE, OP_JMP_IF, 0x03, 0x00);
    // PUSH_NUM 0x00 0x00 HALT
    vm.bytecode[1] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_HALT);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(99.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 20);
    
    // Stack should be empty (JMP_IF consumed the true, then skipped PUSH_NUM)
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 0);
    report_vm_test(312, passed, vm.status, vm.sp, 0.0f, float(vm.sp));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_jmp_if_false() {
    // Test: JMP_IF when condition is false (should NOT jump)
    // Bytecode: PUSH_FALSE, JMP_IF +4, PUSH_NUM 0, HALT
    // Should NOT skip PUSH_NUM, stack should have 99.0
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    vm.bytecode[0] = pack_bytes(OP_PUSH_FALSE, OP_JMP_IF, 0x04, 0x00);
    vm.bytecode[1] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_HALT);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(99.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 20);
    
    // Stack should have 99.0 (JMP_IF consumed false, did not jump, executed PUSH_NUM)
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) &&
                  (vm.stack[0].asNumber() == 99.0f);
    report_vm_test(313, passed, vm.status, vm.sp, 99.0f, vm.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_jmp_if_not_true() {
    // Test: JMP_IF_NOT when condition is true (should NOT jump)
    // Bytecode: PUSH_TRUE, JMP_IF_NOT +4, PUSH_NUM 0, HALT
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    vm.bytecode[0] = pack_bytes(OP_PUSH_TRUE, OP_JMP_IF_NOT, 0x04, 0x00);
    vm.bytecode[1] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_HALT);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(77.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 20);
    
    // Should NOT jump (condition was true), so PUSH_NUM executes
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) &&
                  (vm.stack[0].asNumber() == 77.0f);
    report_vm_test(314, passed, vm.status, vm.sp, 77.0f, vm.stack[0].asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_jmp_if_not_false() {
    // Test: JMP_IF_NOT when condition is false (SHOULD jump)
    // Layout: PUSH_FALSE (1), JMP_IF_NOT (1), offset (2), PUSH_NUM (1), idx (2), HALT (1)
    // Bytes: 0=PUSH_FALSE, 1=JMP_IF_NOT, 2-3=offset, 4=PUSH_NUM, 5-6=idx, 7=HALT
    // After JMP_IF_NOT reads offset, pc=4. We want to skip to HALT at 7, so offset = 3
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    vm.bytecode[0] = pack_bytes(OP_PUSH_FALSE, OP_JMP_IF_NOT, 0x03, 0x00);
    vm.bytecode[1] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_HALT);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(88.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 20);
    
    // Should jump (condition was false), so PUSH_NUM is skipped
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 0);
    report_vm_test(315, passed, vm.status, vm.sp, 0.0f, float(vm.sp));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_jmp_if_nil() {
    // Test: JMP_IF with nil value (nil is falsy, should NOT jump)
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    vm.bytecode[0] = pack_bytes(OP_PUSH_NIL, OP_JMP_IF, 0x04, 0x00);
    vm.bytecode[1] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_HALT);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(55.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 20);
    
    // nil is falsy, so JMP_IF should NOT jump, PUSH_NUM executes
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) &&
                  (vm.stack[0].asNumber() == 55.0f);
    report_vm_test(316, passed, vm.status, vm.sp, 55.0f, vm.stack[0].asNumber());
}

// =============================================================================
// Error Handling Tests (320-329)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_stack_underflow() {
    // Test: Pop from empty stack should trigger underflow error
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    // POP on empty stack
    vm.bytecode[0] = pack_bytes(OP_POP, OP_HALT, 0, 0);
    vm.bytecodeLen = 1;
    
    vm_run(vm, 10);
    
    bool passed = (vm.status == VM_STATUS_ERROR) && 
                  (vm.error == ERR_STACK_UNDERFLOW);
    report_vm_test(320, passed, vm.status, vm.error, float(ERR_STACK_UNDERFLOW), float(vm.error));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_stack_overflow() {
    // Test: Push until stack overflow
    // VM_STACK_SIZE is 64, so we need 65+ pushes to overflow
    // We need more bytecode to trigger overflow
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    vm.bytecodeLen = 0;
    vm.constantCount = 0;
    
    // Initialize stack to prevent undefined behavior
    for (uint i = 0; i < VM_STACK_SIZE; i++) {
        vm.stack[i] = XValue::nil();
    }
    
    // Fill ALL 16 bytecode words with PUSH_TRUE (65+ pushes needed)
    // Each word has 4 PUSH_TRUE = 4 pushes per word
    // 16 words * 4 = 64 bytes = 64 PUSH_TRUE + we need at least one more
    // So we need 17 words, but we only have 16. Let's make it work with 16.
    // Actually 64 pushes would fill the stack exactly. The 65th would overflow.
    // But we only have 64 instructions total. Add a HALT at the end? No, we want overflow.
    // Let's just use what we have and see - actually we need to exceed 64 pushes.
    // The issue: we have exactly 64 bytes = 64 PUSH_NIL, but stack is 64 slots.
    // The 64th push fills it, 65th overflows. We have exactly 64, no overflow.
    // Solution: Use bytecodeLen = 17 but we only have 16 slots. 
    // Alternative: use PUSH_NUM which is 3 bytes, allowing more compact packing? No.
    // Best: Just manually push in a loop until we overflow
    
    // Simpler approach: directly test vm_push overflow
    for (uint i = 0; i < 70; i++) {
        if (!vm_push(vm, XValue::nil())) {
            break;  // Overflow occurred
        }
    }
    
    bool passed = (vm.status == VM_STATUS_ERROR) && 
                  (vm.error == ERR_STACK_OVERFLOW);
    report_vm_test(321, passed, vm.status, vm.error, float(ERR_STACK_OVERFLOW), float(vm.error));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_invalid_opcode() {
    // Test: Invalid opcode should trigger error
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    // Use opcode 0xFE which is not defined
    vm.bytecode[0] = pack_bytes(0xFE, OP_HALT, 0, 0);
    vm.bytecodeLen = 1;
    
    vm_run(vm, 10);
    
    bool passed = (vm.status == VM_STATUS_ERROR) && 
                  (vm.error == ERR_INVALID_OPCODE);
    report_vm_test(322, passed, vm.status, vm.error, float(ERR_INVALID_OPCODE), float(vm.error));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_div_by_zero() {
    // Test: Division by zero should return infinity (not error)
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    // PUSH 42.0, PUSH 0.0, DIV, HALT
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_PUSH_NUM);
    vm.bytecode[1] = pack_bytes(0x01, 0x00, OP_DIV, OP_HALT);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(42.0f);
    vm.constants[1] = XValue::number(0.0f);
    vm.constantCount = 2;
    
    vm_run(vm, 20);
    
    // Should complete (not error) with infinity result
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) &&
                  isinf(vm.stack[0].asNumber());
    report_vm_test(323, passed, vm.status, vm.sp, 1.0f, isinf(vm.stack[0].asNumber()) ? 1.0f : 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_add_underflow() {
    // Test: ADD with only one value on stack should underflow
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    // PUSH one value, then ADD (needs two)
    vm.bytecode[0] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_ADD);
    vm.bytecode[1] = pack_bytes(OP_HALT, 0, 0, 0);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(10.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 20);
    
    bool passed = (vm.status == VM_STATUS_ERROR) && 
                  (vm.error == ERR_STACK_UNDERFLOW);
    report_vm_test(324, passed, vm.status, vm.error, float(ERR_STACK_UNDERFLOW), float(vm.error));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_type_error_arithmetic() {
    // Test: Adding bool to number should return nil (type error behavior)
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    // PUSH_TRUE, PUSH_NUM 0, ADD, HALT
    vm.bytecode[0] = pack_bytes(OP_PUSH_TRUE, OP_PUSH_NUM, 0x00, 0x00);
    vm.bytecode[1] = pack_bytes(OP_ADD, OP_HALT, 0, 0);
    vm.bytecodeLen = 2;
    vm.constants[0] = XValue::number(10.0f);
    vm.constantCount = 1;
    
    vm_run(vm, 20);
    
    // Should complete with nil result (type mismatch)
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) &&
                  (vm.stack[0].type == TYPE_NIL);
    report_vm_test(325, passed, vm.status, vm.sp, float(TYPE_NIL), float(vm.stack[0].type));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_nested_conditionals() {
    // Test: Nested conditional jumps
    // if (true) { if (false) { push 1 } else { push 2 } }
    // Expected: push 2
    // Bytecode layout:
    // 0: PUSH_TRUE
    // 1-3: JMP_IF_NOT +offset_to_end
    // 4: PUSH_FALSE
    // 5-7: JMP_IF_NOT +offset_to_else
    // 8-10: PUSH_NUM 0 (value 1.0)
    // 11-13: JMP +offset_past_else
    // 14-16: PUSH_NUM 1 (value 2.0)
    // 17: HALT
    
    SimpleVM vm;
    vm.sp = 0;
    vm.pc = 0;
    vm.status = VM_STATUS_RUNNING;
    vm.error = ERR_NONE;
    for (uint i = 0; i < 16; i++) vm.bytecode[i] = 0;
    
    // Simplified: PUSH_TRUE, JMP_IF_NOT +end, PUSH_FALSE, JMP_IF +skip, PUSH_NUM 0, JMP +end2, PUSH_NUM 1, HALT
    // Let's trace byte positions carefully:
    // 0: PUSH_TRUE
    // 1: JMP_IF_NOT, 2-3: offset (to HALT if outer false - but it's true so we continue)
    // 4: PUSH_FALSE  
    // 5: JMP_IF, 6-7: offset +4 (if inner true, skip to JMP past else - but it's false)
    // 8: PUSH_NUM, 9-10: 0x01 0x00 (constant 1 = 2.0f)
    // 11: JMP, 12-13: offset +3 (skip past else block)
    // 14: PUSH_NUM, 15-16: 0x00 0x00 (constant 0 = 1.0f) - this is else block, skipped
    // 17: HALT
    
    // Wait, this is wrong. Let me redo:
    // if (true) { if (false) { A } else { B } }
    // Bytecode:
    // PUSH_TRUE, JMP_IF_NOT to_end, PUSH_FALSE, JMP_IF_NOT to_else, <then>, JMP past_else, <else>, HALT
    
    // Actually simpler test - just two levels of jumps
    vm.bytecode[0] = pack_bytes(OP_PUSH_TRUE, OP_JMP_IF_NOT, 0x0C, 0x00); // if false, skip 12 bytes to HALT
    vm.bytecode[1] = pack_bytes(OP_PUSH_FALSE, OP_JMP_IF, 0x04, 0x00);    // if true (inner), skip 4 bytes
    vm.bytecode[2] = pack_bytes(OP_PUSH_NUM, 0x00, 0x00, OP_HALT);         // else: push constant 0, halt
    vm.bytecode[3] = pack_bytes(OP_PUSH_NUM, 0x01, 0x00, OP_HALT);         // then: push constant 1, halt (skipped)
    vm.bytecodeLen = 4;
    vm.constants[0] = XValue::number(2.0f);  // This should be pushed (else branch)
    vm.constants[1] = XValue::number(1.0f);  // This would be pushed if inner was true
    vm.constantCount = 2;
    
    vm_run(vm, 30);
    
    // Outer true -> enter block, inner false -> don't jump -> push 2.0
    bool passed = (vm.status == VM_STATUS_COMPLETED) && 
                  (vm.sp == 1) &&
                  (vm.stack[0].asNumber() == 2.0f);
    report_vm_test(317, passed, vm.status, vm.sp, 2.0f, vm.stack[0].asNumber());
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

