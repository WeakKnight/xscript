// XScript Value System (32-bit version)
// Defines the core value types and operations for the XScript VM

// =============================================================================
// Type Constants
// =============================================================================

static const uint TYPE_NIL      = 0;
static const uint TYPE_BOOL     = 1;
static const uint TYPE_NUMBER   = 2;
static const uint TYPE_STRING   = 3;
static const uint TYPE_TABLE    = 4;
static const uint TYPE_FUNCTION = 5;
static const uint TYPE_USERDATA = 6;
static const uint TYPE_THREAD   = 7;

// First heap-allocated type (for ref counting checks)
static const uint TYPE_FIRST_HEAP = TYPE_STRING;

// =============================================================================
// Value Flags
// =============================================================================

static const uint FLAG_NONE   = 0x00;
static const uint FLAG_MARKED = 0x01;  // GC mark bit
static const uint FLAG_WEAK   = 0x02;  // Weak reference

// =============================================================================
// XValue - Core Value Type (32-bit)
// =============================================================================

struct XValue {
    uint type;       // Type tag
    uint flags;      // Flags (marked, weak, etc.)
    uint data;       // 32-bit value data or heap pointer

    // Create nil value
    static XValue nil() {
        XValue v;
        v.type = TYPE_NIL;
        v.flags = 0;
        v.data = 0;
        return v;
    }

    // Create boolean value
    static XValue boolean(bool b) {
        XValue v;
        v.type = TYPE_BOOL;
        v.flags = 0;
        v.data = b ? 1 : 0;
        return v;
    }

    // Create number value (32-bit float)
    static XValue number(float n) {
        XValue v;
        v.type = TYPE_NUMBER;
        v.flags = 0;
        v.data = asuint(n);
        return v;
    }

    // Create string value (from string pool index)
    static XValue string(uint stringIndex) {
        XValue v;
        v.type = TYPE_STRING;
        v.flags = 0;
        v.data = stringIndex;
        return v;
    }

    // Create table value (from heap offset)
    static XValue table(uint heapOffset) {
        XValue v;
        v.type = TYPE_TABLE;
        v.flags = 0;
        v.data = heapOffset;
        return v;
    }

    // Create function value (from function index)
    static XValue function(uint funcIndex) {
        XValue v;
        v.type = TYPE_FUNCTION;
        v.flags = 0;
        v.data = funcIndex;
        return v;
    }

    // Create userdata value
    static XValue userdata(uint udIndex) {
        XValue v;
        v.type = TYPE_USERDATA;
        v.flags = 0;
        v.data = udIndex;
        return v;
    }

    // Check if value is nil
    bool isNil() {
        return type == TYPE_NIL;
    }

    // Check if value is truthy (not nil and not false)
    bool isTruthy() {
        if (type == TYPE_NIL) return false;
        if (type == TYPE_BOOL) return data != 0;
        return true;
    }

    // Check if value is falsy
    bool isFalsy() {
        return !isTruthy();
    }

    // Check if value is a heap-allocated type
    bool isHeapType() {
        return type >= TYPE_FIRST_HEAP;
    }

    // Get as boolean
    bool asBool() {
        return data != 0;
    }

    // Get as number (32-bit float)
    float asNumber() {
        return asfloat(data);
    }

    // Get as float (same as asNumber for 32-bit)
    float asFloat() {
        return asfloat(data);
    }

    // Get as integer
    int asInt() {
        return int(asfloat(data));
    }

    // Get as uint (heap offset or index)
    uint asUint() {
        return data;
    }

    // Get type name index
    uint getTypeName() {
        return type;
    }
};

// =============================================================================
// Value Comparison
// =============================================================================

// Check if two values are equal
bool xvalue_equals(XValue a, XValue b) {
    if (a.type != b.type) return false;
    
    switch (a.type) {
        case TYPE_NIL:
            return true;
        case TYPE_BOOL:
        case TYPE_STRING:
        case TYPE_TABLE:
        case TYPE_FUNCTION:
        case TYPE_USERDATA:
            return a.data == b.data;
        case TYPE_NUMBER:
            return a.asNumber() == b.asNumber();
        default:
            return false;
    }
}

// Compare two values (for sorting)
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
int xvalue_compare(XValue a, XValue b) {
    // Different types: compare by type id
    if (a.type != b.type) {
        return a.type < b.type ? -1 : 1;
    }
    
    switch (a.type) {
        case TYPE_NIL:
            return 0;
        case TYPE_BOOL:
            if (a.data == b.data) return 0;
            return a.data < b.data ? -1 : 1;
        case TYPE_NUMBER:
            float na = a.asNumber();
            float nb = b.asNumber();
            if (na == nb) return 0;
            return na < nb ? -1 : 1;
        case TYPE_STRING:
            // String comparison done by string pool
            if (a.data == b.data) return 0;
            return a.data < b.data ? -1 : 1;
        default:
            // Other types: compare by pointer/index
            if (a.data == b.data) return 0;
            return a.data < b.data ? -1 : 1;
    }
}

// =============================================================================
// Arithmetic Operations (32-bit float)
// =============================================================================

// Add two values (assumes both are numbers)
XValue xvalue_add(XValue a, XValue b) {
    return XValue::number(a.asNumber() + b.asNumber());
}

// Subtract two values
XValue xvalue_sub(XValue a, XValue b) {
    return XValue::number(a.asNumber() - b.asNumber());
}

// Multiply two values
XValue xvalue_mul(XValue a, XValue b) {
    return XValue::number(a.asNumber() * b.asNumber());
}

// Divide two values
XValue xvalue_div(XValue a, XValue b) {
    return XValue::number(a.asNumber() / b.asNumber());
}

// Modulo two values
XValue xvalue_mod(XValue a, XValue b) {
    float na = a.asNumber();
    float nb = b.asNumber();
    return XValue::number(na - floor(na / nb) * nb);
}

// Power operation
XValue xvalue_pow(XValue a, XValue b) {
    return XValue::number(pow(a.asNumber(), b.asNumber()));
}

// Negate a value
XValue xvalue_neg(XValue a) {
    return XValue::number(-a.asNumber());
}

// =============================================================================
// Comparison Operations
// =============================================================================

XValue xvalue_eq(XValue a, XValue b) {
    return XValue::boolean(xvalue_equals(a, b));
}

XValue xvalue_ne(XValue a, XValue b) {
    return XValue::boolean(!xvalue_equals(a, b));
}

XValue xvalue_lt(XValue a, XValue b) {
    return XValue::boolean(xvalue_compare(a, b) < 0);
}

XValue xvalue_le(XValue a, XValue b) {
    return XValue::boolean(xvalue_compare(a, b) <= 0);
}

XValue xvalue_gt(XValue a, XValue b) {
    return XValue::boolean(xvalue_compare(a, b) > 0);
}

XValue xvalue_ge(XValue a, XValue b) {
    return XValue::boolean(xvalue_compare(a, b) >= 0);
}

// =============================================================================
// Logical Operations
// =============================================================================

XValue xvalue_not(XValue a) {
    return XValue::boolean(a.isFalsy());
}

XValue xvalue_and(XValue a, XValue b) {
    // Lua-style: return first falsy or last truthy
    if (a.isFalsy()) return a;
    return b;
}

XValue xvalue_or(XValue a, XValue b) {
    // Lua-style: return first truthy or last falsy
    if (a.isTruthy()) return a;
    return b;
}
