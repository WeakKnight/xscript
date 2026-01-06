// XScript Arithmetic Tests (32-bit)
// Tests for arithmetic operations on XValue

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
    
    float asNumber() { return asfloat(data); }
};

// =============================================================================
// Arithmetic Operations
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

XValue xvalue_neg(XValue a) {
    return XValue::number(-a.asNumber());
}

XValue xvalue_pow(XValue a, XValue b) {
    return XValue::number(pow(a.asNumber(), b.asNumber()));
}

// =============================================================================
// Comparison Operations
// =============================================================================

XValue xvalue_eq(XValue a, XValue b) {
    if (a.type != b.type) return XValue::boolean(false);
    if (a.type == TYPE_NUMBER) {
        return XValue::boolean(a.asNumber() == b.asNumber());
    }
    return XValue::boolean(a.data == b.data);
}

XValue xvalue_lt(XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return XValue::boolean(a.asNumber() < b.asNumber());
    }
    return XValue::boolean(false);
}

XValue xvalue_le(XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return XValue::boolean(a.asNumber() <= b.asNumber());
    }
    return XValue::boolean(false);
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

void report_arith_test(uint testId, bool passed, float expected, float actual) {
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
// Addition Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_add_integers() {
    XValue a = XValue::number(10.0f);
    XValue b = XValue::number(20.0f);
    XValue c = xvalue_add(a, b);
    
    bool passed = c.asNumber() == 30.0f;
    report_arith_test(100, passed, 30.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_add_floats() {
    XValue a = XValue::number(1.5f);
    XValue b = XValue::number(2.5f);
    XValue c = xvalue_add(a, b);
    
    bool passed = c.asNumber() == 4.0f;
    report_arith_test(101, passed, 4.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_add_negative() {
    XValue a = XValue::number(10.0f);
    XValue b = XValue::number(-3.0f);
    XValue c = xvalue_add(a, b);
    
    bool passed = c.asNumber() == 7.0f;
    report_arith_test(102, passed, 7.0f, c.asNumber());
}

// =============================================================================
// Subtraction Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_sub_integers() {
    XValue a = XValue::number(30.0f);
    XValue b = XValue::number(10.0f);
    XValue c = xvalue_sub(a, b);
    
    bool passed = c.asNumber() == 20.0f;
    report_arith_test(110, passed, 20.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_sub_to_negative() {
    XValue a = XValue::number(5.0f);
    XValue b = XValue::number(10.0f);
    XValue c = xvalue_sub(a, b);
    
    bool passed = c.asNumber() == -5.0f;
    report_arith_test(111, passed, -5.0f, c.asNumber());
}

// =============================================================================
// Multiplication Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_mul_integers() {
    XValue a = XValue::number(6.0f);
    XValue b = XValue::number(7.0f);
    XValue c = xvalue_mul(a, b);
    
    bool passed = c.asNumber() == 42.0f;
    report_arith_test(120, passed, 42.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_mul_by_zero() {
    XValue a = XValue::number(100.0f);
    XValue b = XValue::number(0.0f);
    XValue c = xvalue_mul(a, b);
    
    bool passed = c.asNumber() == 0.0f;
    report_arith_test(121, passed, 0.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_mul_negatives() {
    XValue a = XValue::number(-3.0f);
    XValue b = XValue::number(-4.0f);
    XValue c = xvalue_mul(a, b);
    
    bool passed = c.asNumber() == 12.0f;
    report_arith_test(122, passed, 12.0f, c.asNumber());
}

// =============================================================================
// Division Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_div_integers() {
    XValue a = XValue::number(20.0f);
    XValue b = XValue::number(4.0f);
    XValue c = xvalue_div(a, b);
    
    bool passed = c.asNumber() == 5.0f;
    report_arith_test(130, passed, 5.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_div_fraction() {
    XValue a = XValue::number(10.0f);
    XValue b = XValue::number(4.0f);
    XValue c = xvalue_div(a, b);
    
    bool passed = c.asNumber() == 2.5f;
    report_arith_test(131, passed, 2.5f, c.asNumber());
}

// =============================================================================
// Modulo Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_mod_basic() {
    XValue a = XValue::number(17.0f);
    XValue b = XValue::number(5.0f);
    XValue c = xvalue_mod(a, b);
    
    bool passed = c.asNumber() == 2.0f;
    report_arith_test(140, passed, 2.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_mod_no_remainder() {
    XValue a = XValue::number(15.0f);
    XValue b = XValue::number(5.0f);
    XValue c = xvalue_mod(a, b);
    
    bool passed = c.asNumber() == 0.0f;
    report_arith_test(141, passed, 0.0f, c.asNumber());
}

// =============================================================================
// Power Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_pow_square() {
    XValue a = XValue::number(5.0f);
    XValue b = XValue::number(2.0f);
    XValue c = xvalue_pow(a, b);
    
    bool passed = c.asNumber() == 25.0f;
    report_arith_test(150, passed, 25.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_pow_cube() {
    XValue a = XValue::number(3.0f);
    XValue b = XValue::number(3.0f);
    XValue c = xvalue_pow(a, b);
    
    bool passed = c.asNumber() == 27.0f;
    report_arith_test(151, passed, 27.0f, c.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_pow_zero() {
    XValue a = XValue::number(100.0f);
    XValue b = XValue::number(0.0f);
    XValue c = xvalue_pow(a, b);
    
    bool passed = c.asNumber() == 1.0f;
    report_arith_test(152, passed, 1.0f, c.asNumber());
}

// =============================================================================
// Negation Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_neg_positive() {
    XValue a = XValue::number(42.0f);
    XValue b = xvalue_neg(a);
    
    bool passed = b.asNumber() == -42.0f;
    report_arith_test(160, passed, -42.0f, b.asNumber());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_neg_negative() {
    XValue a = XValue::number(-42.0f);
    XValue b = xvalue_neg(a);
    
    bool passed = b.asNumber() == 42.0f;
    report_arith_test(161, passed, 42.0f, b.asNumber());
}

// =============================================================================
// Comparison Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_lt_true() {
    XValue a = XValue::number(5.0f);
    XValue b = XValue::number(10.0f);
    XValue c = xvalue_lt(a, b);
    
    bool passed = c.data == 1;
    report_arith_test(170, passed, 1.0f, float(c.data));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_lt_false() {
    XValue a = XValue::number(10.0f);
    XValue b = XValue::number(5.0f);
    XValue c = xvalue_lt(a, b);
    
    bool passed = c.data == 0;
    report_arith_test(171, passed, 0.0f, float(c.data));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_le_equal() {
    XValue a = XValue::number(10.0f);
    XValue b = XValue::number(10.0f);
    XValue c = xvalue_le(a, b);
    
    bool passed = c.data == 1;
    report_arith_test(172, passed, 1.0f, float(c.data));
}

// =============================================================================
// Run All Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void run_all_arithmetic_tests() {
    g_testCount[0] = 0;
    
    // Addition
    {
        XValue c = xvalue_add(XValue::number(10.0f), XValue::number(20.0f));
        report_arith_test(100, c.asNumber() == 30.0f, 30.0f, c.asNumber());
    }
    {
        XValue c = xvalue_add(XValue::number(1.5f), XValue::number(2.5f));
        report_arith_test(101, c.asNumber() == 4.0f, 4.0f, c.asNumber());
    }
    {
        XValue c = xvalue_add(XValue::number(10.0f), XValue::number(-3.0f));
        report_arith_test(102, c.asNumber() == 7.0f, 7.0f, c.asNumber());
    }
    
    // Subtraction
    {
        XValue c = xvalue_sub(XValue::number(30.0f), XValue::number(10.0f));
        report_arith_test(110, c.asNumber() == 20.0f, 20.0f, c.asNumber());
    }
    {
        XValue c = xvalue_sub(XValue::number(5.0f), XValue::number(10.0f));
        report_arith_test(111, c.asNumber() == -5.0f, -5.0f, c.asNumber());
    }
    
    // Multiplication
    {
        XValue c = xvalue_mul(XValue::number(6.0f), XValue::number(7.0f));
        report_arith_test(120, c.asNumber() == 42.0f, 42.0f, c.asNumber());
    }
    {
        XValue c = xvalue_mul(XValue::number(100.0f), XValue::number(0.0f));
        report_arith_test(121, c.asNumber() == 0.0f, 0.0f, c.asNumber());
    }
    {
        XValue c = xvalue_mul(XValue::number(-3.0f), XValue::number(-4.0f));
        report_arith_test(122, c.asNumber() == 12.0f, 12.0f, c.asNumber());
    }
    
    // Division
    {
        XValue c = xvalue_div(XValue::number(20.0f), XValue::number(4.0f));
        report_arith_test(130, c.asNumber() == 5.0f, 5.0f, c.asNumber());
    }
    {
        XValue c = xvalue_div(XValue::number(10.0f), XValue::number(4.0f));
        report_arith_test(131, c.asNumber() == 2.5f, 2.5f, c.asNumber());
    }
    
    // Modulo
    {
        XValue c = xvalue_mod(XValue::number(17.0f), XValue::number(5.0f));
        report_arith_test(140, c.asNumber() == 2.0f, 2.0f, c.asNumber());
    }
    
    // Power
    {
        XValue c = xvalue_pow(XValue::number(5.0f), XValue::number(2.0f));
        report_arith_test(150, c.asNumber() == 25.0f, 25.0f, c.asNumber());
    }
    {
        XValue c = xvalue_pow(XValue::number(3.0f), XValue::number(3.0f));
        report_arith_test(151, c.asNumber() == 27.0f, 27.0f, c.asNumber());
    }
    
    // Negation
    {
        XValue c = xvalue_neg(XValue::number(42.0f));
        report_arith_test(160, c.asNumber() == -42.0f, -42.0f, c.asNumber());
    }
    {
        XValue c = xvalue_neg(XValue::number(-42.0f));
        report_arith_test(161, c.asNumber() == 42.0f, 42.0f, c.asNumber());
    }
    
    // Comparison
    {
        XValue c = xvalue_lt(XValue::number(5.0f), XValue::number(10.0f));
        report_arith_test(170, c.data == 1, 1.0f, float(c.data));
    }
    {
        XValue c = xvalue_lt(XValue::number(10.0f), XValue::number(5.0f));
        report_arith_test(171, c.data == 0, 0.0f, float(c.data));
    }
    {
        XValue c = xvalue_le(XValue::number(10.0f), XValue::number(10.0f));
        report_arith_test(172, c.data == 1, 1.0f, float(c.data));
    }
}

