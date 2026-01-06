// XScript Value Type Tests (32-bit)
// Tests for basic XValue operations

// Include runtime modules by inlining definitions for standalone compilation
// In actual usage, these would be imported

// =============================================================================
// Type Constants (from value.slang)
// =============================================================================

static const uint TYPE_NIL      = 0;
static const uint TYPE_BOOL     = 1;
static const uint TYPE_NUMBER   = 2;
static const uint TYPE_STRING   = 3;
static const uint TYPE_TABLE    = 4;
static const uint TYPE_FUNCTION = 5;

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
    
    float asNumber() { return asfloat(data); }
    bool asBool() { return data != 0; }
};

bool xvalue_equals(XValue a, XValue b) {
    if (a.type != b.type) return false;
    if (a.type == TYPE_NIL) return true;
    if (a.type == TYPE_NUMBER) return a.asNumber() == b.asNumber();
    return a.data == b.data;
}

// =============================================================================
// Test Result Buffer
// =============================================================================

struct TestResult {
    uint testId;
    uint passed;
    uint expectedType;
    uint actualType;
    float expectedValue;
    float actualValue;
};

RWStructuredBuffer<TestResult> g_testResults;
RWStructuredBuffer<uint> g_testCount;

void report_test(uint testId, bool passed, 
                 uint expectedType, uint actualType,
                 float expectedValue, float actualValue) {
    uint idx;
    InterlockedAdd(g_testCount[0], 1, idx);
    
    TestResult result;
    result.testId = testId;
    result.passed = passed ? 1 : 0;
    result.expectedType = expectedType;
    result.actualType = actualType;
    result.expectedValue = expectedValue;
    result.actualValue = actualValue;
    g_testResults[idx] = result;
}

// =============================================================================
// Value Creation Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_nil_creation() {
    XValue v = XValue::nil();
    
    bool passed = (v.type == TYPE_NIL) && (v.data == 0) && v.isNil();
    report_test(1, passed, TYPE_NIL, v.type, 0.0f, 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_bool_true() {
    XValue v = XValue::boolean(true);
    
    bool passed = (v.type == TYPE_BOOL) && (v.data == 1) && v.isTruthy();
    report_test(2, passed, TYPE_BOOL, v.type, 1.0f, float(v.data));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_bool_false() {
    XValue v = XValue::boolean(false);
    
    bool passed = (v.type == TYPE_BOOL) && (v.data == 0) && !v.isTruthy();
    report_test(3, passed, TYPE_BOOL, v.type, 0.0f, float(v.data));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_number_integer() {
    XValue v = XValue::number(42.0f);
    
    bool passed = (v.type == TYPE_NUMBER) && (v.asNumber() == 42.0f);
    report_test(4, passed, TYPE_NUMBER, v.type, 42.0f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_number_float() {
    XValue v = XValue::number(3.14159f);
    
    float diff = abs(v.asNumber() - 3.14159f);
    bool passed = (v.type == TYPE_NUMBER) && (diff < 0.0001f);
    report_test(5, passed, TYPE_NUMBER, v.type, 3.14159f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_number_negative() {
    XValue v = XValue::number(-100.5f);
    
    bool passed = (v.type == TYPE_NUMBER) && (v.asNumber() == -100.5f);
    report_test(6, passed, TYPE_NUMBER, v.type, -100.5f, v.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_number_zero() {
    XValue v = XValue::number(0.0f);
    
    bool passed = (v.type == TYPE_NUMBER) && (v.asNumber() == 0.0f) && v.isTruthy();
    report_test(7, passed, TYPE_NUMBER, v.type, 0.0f, v.asNumber());
}

// =============================================================================
// Equality Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_nil_equals_nil() {
    XValue a = XValue::nil();
    XValue b = XValue::nil();
    
    bool passed = xvalue_equals(a, b);
    report_test(10, passed, 1, passed ? 1 : 0, 0.0f, 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_number_equals() {
    XValue a = XValue::number(10.0f);
    XValue b = XValue::number(10.0f);
    
    bool passed = xvalue_equals(a, b);
    report_test(11, passed, 1, passed ? 1 : 0, 10.0f, b.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_number_not_equals() {
    XValue a = XValue::number(10.0f);
    XValue b = XValue::number(20.0f);
    
    bool passed = !xvalue_equals(a, b);
    report_test(12, passed, 1, passed ? 1 : 0, 10.0f, b.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_bool_equals() {
    XValue a = XValue::boolean(true);
    XValue b = XValue::boolean(true);
    
    bool passed = xvalue_equals(a, b);
    report_test(13, passed, 1, passed ? 1 : 0, 1.0f, float(b.data));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_different_types_not_equal() {
    XValue a = XValue::number(1.0f);
    XValue b = XValue::boolean(true);
    
    bool passed = !xvalue_equals(a, b);
    report_test(14, passed, 1, passed ? 1 : 0, 0.0f, 0.0f);
}

// =============================================================================
// Truthiness Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_nil_is_falsy() {
    XValue v = XValue::nil();
    
    bool passed = !v.isTruthy();
    report_test(20, passed, 0, v.isTruthy() ? 1 : 0, 0.0f, 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_false_is_falsy() {
    XValue v = XValue::boolean(false);
    
    bool passed = !v.isTruthy();
    report_test(21, passed, 0, v.isTruthy() ? 1 : 0, 0.0f, 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_true_is_truthy() {
    XValue v = XValue::boolean(true);
    
    bool passed = v.isTruthy();
    report_test(22, passed, 1, v.isTruthy() ? 1 : 0, 0.0f, 0.0f);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_zero_is_truthy() {
    // Unlike some languages, 0 is truthy in Lua-style semantics
    XValue v = XValue::number(0.0f);
    
    bool passed = v.isTruthy();
    report_test(23, passed, 1, v.isTruthy() ? 1 : 0, 0.0f, 0.0f);
}

// =============================================================================
// Run All Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void run_all_value_tests() {
    // Reset test count
    g_testCount[0] = 0;
    
    // Creation tests
    {
        XValue v = XValue::nil();
        bool passed = (v.type == TYPE_NIL) && v.isNil();
        report_test(1, passed, TYPE_NIL, v.type, 0.0f, 0.0f);
    }
    
    {
        XValue v = XValue::boolean(true);
        bool passed = (v.type == TYPE_BOOL) && v.isTruthy();
        report_test(2, passed, TYPE_BOOL, v.type, 1.0f, float(v.data));
    }
    
    {
        XValue v = XValue::boolean(false);
        bool passed = (v.type == TYPE_BOOL) && !v.isTruthy();
        report_test(3, passed, TYPE_BOOL, v.type, 0.0f, float(v.data));
    }
    
    {
        XValue v = XValue::number(42.0f);
        bool passed = (v.type == TYPE_NUMBER) && (v.asNumber() == 42.0f);
        report_test(4, passed, TYPE_NUMBER, v.type, 42.0f, v.asNumber());
    }
    
    {
        XValue v = XValue::number(3.14159f);
        float diff = abs(v.asNumber() - 3.14159f);
        bool passed = (v.type == TYPE_NUMBER) && (diff < 0.0001f);
        report_test(5, passed, TYPE_NUMBER, v.type, 3.14159f, v.asNumber());
    }
    
    {
        XValue v = XValue::number(-100.5f);
        bool passed = (v.type == TYPE_NUMBER) && (v.asNumber() == -100.5f);
        report_test(6, passed, TYPE_NUMBER, v.type, -100.5f, v.asNumber());
    }
    
    // Equality tests
    {
        XValue a = XValue::nil();
        XValue b = XValue::nil();
        bool passed = xvalue_equals(a, b);
        report_test(10, passed, 1, passed ? 1 : 0, 0.0f, 0.0f);
    }
    
    {
        XValue a = XValue::number(10.0f);
        XValue b = XValue::number(10.0f);
        bool passed = xvalue_equals(a, b);
        report_test(11, passed, 1, passed ? 1 : 0, 10.0f, b.asNumber());
    }
    
    {
        XValue a = XValue::number(10.0f);
        XValue b = XValue::number(20.0f);
        bool passed = !xvalue_equals(a, b);
        report_test(12, passed, 1, passed ? 1 : 0, 10.0f, b.asNumber());
    }
    
    // Truthiness tests
    {
        XValue v = XValue::nil();
        bool passed = !v.isTruthy();
        report_test(20, passed, 0, v.isTruthy() ? 1 : 0, 0.0f, 0.0f);
    }
    
    {
        XValue v = XValue::boolean(false);
        bool passed = !v.isTruthy();
        report_test(21, passed, 0, v.isTruthy() ? 1 : 0, 0.0f, 0.0f);
    }
    
    {
        XValue v = XValue::number(0.0f);
        bool passed = v.isTruthy();  // 0 is truthy in Lua
        report_test(23, passed, 1, v.isTruthy() ? 1 : 0, 0.0f, 0.0f);
    }
}

