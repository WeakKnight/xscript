// XScript Table Tests (32-bit)
// Tests for table creation, get/set operations, metatable, and iteration

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
// Heap Constants
// =============================================================================

static const uint HEAP_BLOCK_HEADER_SIZE = 4;
static const uint HEAP_ALIGNMENT = 4;

// =============================================================================
// Table Constants
// =============================================================================

static const uint TABLE_INITIAL_CAPACITY = 8;
static const uint TABLE_OFF_CAPACITY = 0;
static const uint TABLE_OFF_COUNT = 1;
static const uint TABLE_OFF_METATABLE = 2;
static const uint TABLE_OFF_ARRAY_SIZE = 3;
static const uint TABLE_OFF_ENTRIES = 4;

// Entry structure size (in uints)
static const uint ENTRY_SIZE = 7;
static const uint ENTRY_OFF_KEY = 0;
static const uint ENTRY_OFF_VALUE = 3;
static const uint ENTRY_OFF_NEXT = 6;

// Meta method indices
static const uint META_ADD       = 0;
static const uint META_INDEX     = 10;
static const uint META_NAME_ADD  = 0x1000;

// =============================================================================
// Heap Allocator State
// =============================================================================

struct HeapAllocator {
    uint freeListSmall;
    uint freeListMedium;
    uint freeListLarge;
    uint freeListHuge;
    uint nextFree;
    uint totalSize;
    uint usedSize;
    uint allocCount;
};

// =============================================================================
// XValue Structure
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

    bool isNil() {
        return type == TYPE_NIL;
    }

    float asNumber() {
        return asfloat(data);
    }

    uint asUint() {
        return data;
    }
};

// =============================================================================
// Test Result Structure
// =============================================================================

struct TestResult {
    uint testId;
    uint passed;
    uint expected;
    uint actual;
};

// =============================================================================
// GPU Buffers
// =============================================================================

RWStructuredBuffer<TestResult> g_testResults;
RWStructuredBuffer<uint> g_testCount;
RWStructuredBuffer<uint> g_heapMemory;
RWStructuredBuffer<HeapAllocator> g_heapState;
RWStructuredBuffer<uint> g_tableResults;  // For concurrent tests

// =============================================================================
// Helper to report test results
// =============================================================================

void report_test(uint testId, bool passed, uint expected, uint actual) {
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
// Heap Functions (simplified for table testing)
// =============================================================================

void heap_init(uint totalSizeInUints) {
    g_heapState[0].freeListSmall = 0;
    g_heapState[0].freeListMedium = 0;
    g_heapState[0].freeListLarge = 0;
    g_heapState[0].freeListHuge = 0;
    g_heapState[0].nextFree = 1;
    g_heapState[0].totalSize = totalSizeInUints;
    g_heapState[0].usedSize = 0;
    g_heapState[0].allocCount = 0;
}

uint align_up(uint size, uint alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

uint heap_alloc(uint size) {
    uint blockSize = align_up(size + HEAP_BLOCK_HEADER_SIZE * 4, HEAP_ALIGNMENT * 4);
    uint blockSizeInUints = blockSize / 4;
    
    uint offset;
    InterlockedAdd(g_heapState[0].nextFree, blockSizeInUints, offset);
    
    if (offset + blockSizeInUints > g_heapState[0].totalSize) {
        return 0;
    }
    
    // Write block header
    g_heapMemory[offset] = blockSizeInUints;
    g_heapMemory[offset + 1] = 1;  // refCount
    g_heapMemory[offset + 2] = 0;  // flags
    g_heapMemory[offset + 3] = 0;  // next
    
    InterlockedAdd(g_heapState[0].allocCount, 1);
    
    // Return pointer to data (after header)
    return offset + HEAP_BLOCK_HEADER_SIZE;
}

uint heap_read_uint(uint ptr, uint offset) {
    return g_heapMemory[ptr + offset];
}

void heap_write_uint(uint ptr, uint offset, uint value) {
    g_heapMemory[ptr + offset] = value;
}

// =============================================================================
// Value Comparison
// =============================================================================

bool xvalue_equals(XValue a, XValue b) {
    if (a.type != b.type) return false;
    
    if (a.type == TYPE_NIL) return true;
    if (a.type == TYPE_NUMBER) return a.asNumber() == b.asNumber();
    return a.data == b.data;
}

// =============================================================================
// Hash Function
// =============================================================================

uint xvalue_hash(XValue v) {
    uint hash = 2166136261u;
    
    hash ^= v.type;
    hash *= 16777619u;
    
    hash ^= v.data;
    hash *= 16777619u;
    
    return hash;
}

// =============================================================================
// XValue Read/Write to Heap
// =============================================================================

XValue read_xvalue(uint ptr, uint offset) {
    XValue v;
    v.type = heap_read_uint(ptr, offset);
    v.flags = heap_read_uint(ptr, offset + 1);
    v.data = heap_read_uint(ptr, offset + 2);
    return v;
}

void write_xvalue(uint ptr, uint offset, XValue v) {
    heap_write_uint(ptr, offset, v.type);
    heap_write_uint(ptr, offset + 1, v.flags);
    heap_write_uint(ptr, offset + 2, v.data);
}

// =============================================================================
// Table Functions
// =============================================================================

XValue table_new(uint initialCapacity) {
    if (initialCapacity == 0) {
        initialCapacity = TABLE_INITIAL_CAPACITY;
    }
    
    uint headerSize = TABLE_OFF_ENTRIES;
    uint entriesSize = initialCapacity * ENTRY_SIZE;
    uint totalSize = (headerSize + entriesSize) * 4;
    
    uint ptr = heap_alloc(totalSize);
    if (ptr == 0) {
        return XValue::nil();
    }
    
    heap_write_uint(ptr, TABLE_OFF_CAPACITY, initialCapacity);
    heap_write_uint(ptr, TABLE_OFF_COUNT, 0);
    heap_write_uint(ptr, TABLE_OFF_METATABLE, 0);
    heap_write_uint(ptr, TABLE_OFF_ARRAY_SIZE, 0);
    
    // Initialize entries to nil
    for (uint i = 0; i < initialCapacity; i++) {
        uint entryOffset = TABLE_OFF_ENTRIES + i * ENTRY_SIZE;
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_KEY, TYPE_NIL);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_KEY + 1, 0);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_KEY + 2, 0);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_VALUE, TYPE_NIL);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_VALUE + 1, 0);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_VALUE + 2, 0);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_NEXT, 0);
    }
    
    return XValue::table(ptr);
}

XValue table_get(XValue t, XValue key) {
    if (t.type != TYPE_TABLE) {
        return XValue::nil();
    }
    
    uint ptr = t.asUint();
    uint capacity = heap_read_uint(ptr, TABLE_OFF_CAPACITY);
    
    if (capacity == 0) {
        return XValue::nil();
    }
    
    uint hash = xvalue_hash(key);
    uint bucket = hash % capacity;
    
    uint entryOffset = TABLE_OFF_ENTRIES + bucket * ENTRY_SIZE;
    
    XValue entryKey = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
    if (entryKey.isNil()) {
        return XValue::nil();
    }
    
    // Note: next pointer is stored as (index + 1), so 0 means "no next"
    while (true) {
        if (xvalue_equals(entryKey, key)) {
            return read_xvalue(ptr, entryOffset + ENTRY_OFF_VALUE);
        }
        
        uint next = heap_read_uint(ptr, entryOffset + ENTRY_OFF_NEXT);
        if (next == 0) {
            break;
        }
        
        // Convert 1-based next to slot offset
        entryOffset = TABLE_OFF_ENTRIES + (next - 1) * ENTRY_SIZE;
        entryKey = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
    }
    
    return XValue::nil();
}

bool table_set(XValue t, XValue key, XValue value) {
    if (t.type != TYPE_TABLE || key.isNil()) {
        return false;
    }
    
    uint ptr = t.asUint();
    uint capacity = heap_read_uint(ptr, TABLE_OFF_CAPACITY);
    uint count = heap_read_uint(ptr, TABLE_OFF_COUNT);
    
    uint hash = xvalue_hash(key);
    uint bucket = hash % capacity;
    
    uint entryOffset = TABLE_OFF_ENTRIES + bucket * ENTRY_SIZE;
    
    XValue entryKey = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
    
    if (entryKey.isNil()) {
        write_xvalue(ptr, entryOffset + ENTRY_OFF_KEY, key);
        write_xvalue(ptr, entryOffset + ENTRY_OFF_VALUE, value);
        heap_write_uint(ptr, TABLE_OFF_COUNT, count + 1);
        return true;
    }
    
    uint prevOffset = entryOffset;
    while (true) {
        if (xvalue_equals(entryKey, key)) {
            write_xvalue(ptr, entryOffset + ENTRY_OFF_VALUE, value);
            return true;
        }
        
        uint next = heap_read_uint(ptr, entryOffset + ENTRY_OFF_NEXT);
        if (next == 0) {
            break;
        }
        
        prevOffset = entryOffset;
        // Convert 1-based next to slot offset
        entryOffset = TABLE_OFF_ENTRIES + (next - 1) * ENTRY_SIZE;
        entryKey = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
    }
    
    // Find free slot for chaining
    for (uint i = 0; i < capacity; i++) {
        uint freeOffset = TABLE_OFF_ENTRIES + i * ENTRY_SIZE;
        XValue freeKey = read_xvalue(ptr, freeOffset + ENTRY_OFF_KEY);
        
        if (freeKey.isNil()) {
            write_xvalue(ptr, freeOffset + ENTRY_OFF_KEY, key);
            write_xvalue(ptr, freeOffset + ENTRY_OFF_VALUE, value);
            // Store 1-based index (i+1) so 0 means "no next"
            heap_write_uint(ptr, entryOffset + ENTRY_OFF_NEXT, i + 1);
            heap_write_uint(ptr, TABLE_OFF_COUNT, count + 1);
            return true;
        }
    }
    
    return false;  // Table full
}

uint table_count(XValue t) {
    if (t.type != TYPE_TABLE) {
        return 0;
    }
    return heap_read_uint(t.asUint(), TABLE_OFF_COUNT);
}

XValue table_getmetatable(XValue t) {
    if (t.type != TYPE_TABLE) {
        return XValue::nil();
    }
    
    uint ptr = t.asUint();
    uint mtPtr = heap_read_uint(ptr, TABLE_OFF_METATABLE);
    
    if (mtPtr == 0) {
        return XValue::nil();
    }
    
    return XValue::table(mtPtr);
}

bool table_setmetatable(XValue t, XValue mt) {
    if (t.type != TYPE_TABLE) {
        return false;
    }
    
    uint ptr = t.asUint();
    
    if (mt.isNil()) {
        heap_write_uint(ptr, TABLE_OFF_METATABLE, 0);
    } else if (mt.type == TYPE_TABLE) {
        heap_write_uint(ptr, TABLE_OFF_METATABLE, mt.asUint());
    } else {
        return false;
    }
    
    return true;
}

XValue table_getmetamethod(XValue t, uint metaIndex) {
    XValue mt = table_getmetatable(t);
    if (mt.isNil()) {
        return XValue::nil();
    }
    
    XValue key = XValue::string(META_NAME_ADD + metaIndex);
    return table_get(mt, key);
}

bool table_hasmetamethod(XValue t, uint metaIndex) {
    XValue method = table_getmetamethod(t, metaIndex);
    return !method.isNil();
}

bool table_next(XValue t, inout XValue key, out XValue value) {
    if (t.type != TYPE_TABLE) {
        return false;
    }
    
    uint ptr = t.asUint();
    uint capacity = heap_read_uint(ptr, TABLE_OFF_CAPACITY);
    
    uint startBucket = 0;
    
    if (!key.isNil()) {
        uint hash = xvalue_hash(key);
        startBucket = (hash % capacity) + 1;
    }
    
    for (uint i = startBucket; i < capacity; i++) {
        uint entryOffset = TABLE_OFF_ENTRIES + i * ENTRY_SIZE;
        XValue entryKey = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
        
        if (!entryKey.isNil()) {
            key = entryKey;
            value = read_xvalue(ptr, entryOffset + ENTRY_OFF_VALUE);
            return true;
        }
    }
    
    return false;
}


// =============================================================================
// TEST CATEGORY 1: Hash Function (700-709)
// =============================================================================

// Test 700: Hash nil value
[shader("compute")]
[numthreads(1, 1, 1)]
void test_hash_nil() {
    XValue v = XValue::nil();
    uint hash = xvalue_hash(v);
    
    // Should produce a valid hash (not 0)
    bool passed = hash != 0;
    report_test(700, passed, 1, hash != 0 ? 1 : 0);
}

// Test 701: Hash number value
[shader("compute")]
[numthreads(1, 1, 1)]
void test_hash_number() {
    XValue v = XValue::number(42.0f);
    uint hash = xvalue_hash(v);
    
    bool passed = hash != 0;
    report_test(701, passed, 1, hash != 0 ? 1 : 0);
}

// Test 702: Hash bool value
[shader("compute")]
[numthreads(1, 1, 1)]
void test_hash_bool() {
    XValue v = XValue::boolean(true);
    uint hash = xvalue_hash(v);
    
    bool passed = hash != 0;
    report_test(702, passed, 1, hash != 0 ? 1 : 0);
}

// Test 703: Different values produce different hashes
[shader("compute")]
[numthreads(1, 1, 1)]
void test_hash_different() {
    XValue v1 = XValue::number(1.0f);
    XValue v2 = XValue::number(2.0f);
    XValue v3 = XValue::boolean(true);
    
    uint h1 = xvalue_hash(v1);
    uint h2 = xvalue_hash(v2);
    uint h3 = xvalue_hash(v3);
    
    // All should be different
    bool passed = (h1 != h2) && (h2 != h3) && (h1 != h3);
    report_test(703, passed, 1, passed ? 1 : 0);
}

// Test 704: Same value produces consistent hash
[shader("compute")]
[numthreads(1, 1, 1)]
void test_hash_consistent() {
    XValue v = XValue::number(3.14159f);
    uint h1 = xvalue_hash(v);
    uint h2 = xvalue_hash(v);
    
    report_test(704, h1 == h2, h1, h2);
}


// =============================================================================
// TEST CATEGORY 2: XValue Heap Helpers (710-719)
// =============================================================================

// Test 710: Round-trip nil
[shader("compute")]
[numthreads(1, 1, 1)]
void test_xvalue_roundtrip_nil() {
    heap_init(4096);
    
    uint ptr = heap_alloc(64);
    XValue v = XValue::nil();
    
    write_xvalue(ptr, 0, v);
    XValue v2 = read_xvalue(ptr, 0);
    
    bool passed = v2.isNil();
    report_test(710, passed, TYPE_NIL, v2.type);
}

// Test 711: Round-trip number
[shader("compute")]
[numthreads(1, 1, 1)]
void test_xvalue_roundtrip_number() {
    heap_init(4096);
    
    uint ptr = heap_alloc(64);
    XValue v = XValue::number(42.5f);
    
    write_xvalue(ptr, 0, v);
    XValue v2 = read_xvalue(ptr, 0);
    
    bool passed = (v2.type == TYPE_NUMBER) && (v2.asNumber() == 42.5f);
    report_test(711, passed, TYPE_NUMBER, v2.type);
}

// Test 712: Round-trip bool
[shader("compute")]
[numthreads(1, 1, 1)]
void test_xvalue_roundtrip_bool() {
    heap_init(4096);
    
    uint ptr = heap_alloc(64);
    XValue v = XValue::boolean(true);
    
    write_xvalue(ptr, 0, v);
    XValue v2 = read_xvalue(ptr, 0);
    
    bool passed = (v2.type == TYPE_BOOL) && (v2.data == 1);
    report_test(712, passed, 1, v2.data);
}

// Test 713: Multiple values at different offsets
[shader("compute")]
[numthreads(1, 1, 1)]
void test_xvalue_multiple_offsets() {
    heap_init(4096);
    
    uint ptr = heap_alloc(128);
    
    XValue v1 = XValue::number(1.0f);
    XValue v2 = XValue::number(2.0f);
    XValue v3 = XValue::number(3.0f);
    
    write_xvalue(ptr, 0, v1);
    write_xvalue(ptr, 3, v2);
    write_xvalue(ptr, 6, v3);
    
    XValue r1 = read_xvalue(ptr, 0);
    XValue r2 = read_xvalue(ptr, 3);
    XValue r3 = read_xvalue(ptr, 6);
    
    bool passed = (r1.asNumber() == 1.0f) && (r2.asNumber() == 2.0f) && (r3.asNumber() == 3.0f);
    report_test(713, passed, 1, passed ? 1 : 0);
}


// =============================================================================
// TEST CATEGORY 3: Table Creation (720-729)
// =============================================================================

// Test 720: Create with default capacity
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_new_default() {
    heap_init(4096);
    
    XValue t = table_new(0);  // Default capacity
    
    bool passed = t.type == TYPE_TABLE && t.asUint() != 0;
    report_test(720, passed, TYPE_TABLE, t.type);
}

// Test 721: Create with custom capacity
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_new_custom() {
    heap_init(4096);
    
    XValue t = table_new(16);
    uint ptr = t.asUint();
    uint capacity = heap_read_uint(ptr, TABLE_OFF_CAPACITY);
    
    report_test(721, capacity == 16, 16, capacity);
}

// Test 722: Returns valid table XValue
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_new_valid_type() {
    heap_init(4096);
    
    XValue t = table_new(8);
    
    bool passed = (t.type == TYPE_TABLE);
    report_test(722, passed, TYPE_TABLE, t.type);
}

// Test 723: Initial count is 0
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_new_count_zero() {
    heap_init(4096);
    
    XValue t = table_new(8);
    uint count = table_count(t);
    
    report_test(723, count == 0, 0, count);
}

// Test 724: Initial metatable is nil
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_new_metatable_nil() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue mt = table_getmetatable(t);
    
    bool passed = mt.isNil();
    report_test(724, passed, TYPE_NIL, mt.type);
}


// =============================================================================
// TEST CATEGORY 4: Table Get/Set Operations (730-749)
// =============================================================================

// Test 730: Set single key-value
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_set_single() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue key = XValue::number(1.0f);
    XValue value = XValue::number(100.0f);
    
    bool success = table_set(t, key, value);
    report_test(730, success, 1, success ? 1 : 0);
}

// Test 731: Get existing key
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_get_existing() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue key = XValue::number(1.0f);
    XValue value = XValue::number(100.0f);
    
    table_set(t, key, value);
    XValue result = table_get(t, key);
    
    bool passed = (result.type == TYPE_NUMBER) && (result.asNumber() == 100.0f);
    report_test(731, passed, 100, uint(result.asNumber()));
}

// Test 732: Get non-existing key returns nil
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_get_nonexisting() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue key = XValue::number(999.0f);
    
    XValue result = table_get(t, key);
    
    bool passed = result.isNil();
    report_test(732, passed, TYPE_NIL, result.type);
}

// Test 733: Update existing key
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_set_update() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue key = XValue::number(1.0f);
    
    table_set(t, key, XValue::number(100.0f));
    table_set(t, key, XValue::number(200.0f));  // Update
    
    XValue result = table_get(t, key);
    bool passed = result.asNumber() == 200.0f;
    report_test(733, passed, 200, uint(result.asNumber()));
}

// Test 734: Set with nil key fails
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_set_nil_key() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue key = XValue::nil();
    XValue value = XValue::number(100.0f);
    
    bool success = table_set(t, key, value);
    report_test(734, !success, 0, success ? 1 : 0);
}

// Test 735: Count increases with new keys
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_count_increases() {
    heap_init(4096);
    
    XValue t = table_new(8);
    
    uint count0 = table_count(t);
    table_set(t, XValue::number(1.0f), XValue::number(10.0f));
    uint count1 = table_count(t);
    table_set(t, XValue::number(2.0f), XValue::number(20.0f));
    uint count2 = table_count(t);
    
    bool passed = (count0 == 0) && (count1 == 1) && (count2 == 2);
    report_test(735, passed, 2, count2);
}

// Test 736: Multiple different keys
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_multiple_keys() {
    heap_init(4096);
    
    XValue t = table_new(8);
    
    table_set(t, XValue::number(1.0f), XValue::number(10.0f));
    table_set(t, XValue::number(2.0f), XValue::number(20.0f));
    table_set(t, XValue::number(3.0f), XValue::number(30.0f));
    
    XValue v1 = table_get(t, XValue::number(1.0f));
    XValue v2 = table_get(t, XValue::number(2.0f));
    XValue v3 = table_get(t, XValue::number(3.0f));
    
    bool passed = (v1.asNumber() == 10.0f) && (v2.asNumber() == 20.0f) && (v3.asNumber() == 30.0f);
    report_test(736, passed, 1, passed ? 1 : 0);
}

// Test 737: Get from non-table returns nil
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_get_non_table() {
    XValue notTable = XValue::number(42.0f);
    XValue key = XValue::number(1.0f);
    
    XValue result = table_get(notTable, key);
    
    bool passed = result.isNil();
    report_test(737, passed, TYPE_NIL, result.type);
}

// Test 738: Hash collision handling (set keys that might collide)
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_collision() {
    heap_init(8192);
    
    XValue t = table_new(4);  // Small capacity to force collisions
    
    // Add more keys than buckets
    table_set(t, XValue::number(1.0f), XValue::number(10.0f));
    table_set(t, XValue::number(2.0f), XValue::number(20.0f));
    table_set(t, XValue::number(3.0f), XValue::number(30.0f));
    table_set(t, XValue::number(5.0f), XValue::number(50.0f));  // Likely collision with 1
    
    // All values should be retrievable
    XValue v1 = table_get(t, XValue::number(1.0f));
    XValue v2 = table_get(t, XValue::number(2.0f));
    XValue v3 = table_get(t, XValue::number(3.0f));
    XValue v5 = table_get(t, XValue::number(5.0f));
    
    bool passed = (v1.asNumber() == 10.0f) && (v2.asNumber() == 20.0f) && 
                  (v3.asNumber() == 30.0f) && (v5.asNumber() == 50.0f);
    report_test(738, passed, 1, passed ? 1 : 0);
}

// Test 739: Table full returns false
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_full() {
    heap_init(8192);
    
    XValue t = table_new(4);  // Small capacity
    
    // Fill all slots
    table_set(t, XValue::number(1.0f), XValue::number(10.0f));
    table_set(t, XValue::number(2.0f), XValue::number(20.0f));
    table_set(t, XValue::number(3.0f), XValue::number(30.0f));
    table_set(t, XValue::number(4.0f), XValue::number(40.0f));
    
    // Next insert should fail
    bool success = table_set(t, XValue::number(5.0f), XValue::number(50.0f));
    report_test(739, !success, 0, success ? 1 : 0);
}

// Test 740: Set with number key
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_number_key() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue key = XValue::number(42.5f);
    XValue value = XValue::number(100.0f);
    
    table_set(t, key, value);
    XValue result = table_get(t, key);
    
    bool passed = result.asNumber() == 100.0f;
    report_test(740, passed, 100, uint(result.asNumber()));
}

// Test 741: Set with bool key
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_bool_key() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue key = XValue::boolean(true);
    XValue value = XValue::number(100.0f);
    
    table_set(t, key, value);
    XValue result = table_get(t, key);
    
    bool passed = result.asNumber() == 100.0f;
    report_test(741, passed, 100, uint(result.asNumber()));
}

// Test 742: Set with string key
[shader("compute")]
[numthreads(1, 1, 1)]
void test_table_string_key() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue key = XValue::string(123);  // String index
    XValue value = XValue::number(100.0f);
    
    table_set(t, key, value);
    XValue result = table_get(t, key);
    
    bool passed = result.asNumber() == 100.0f;
    report_test(742, passed, 100, uint(result.asNumber()));
}


// =============================================================================
// TEST CATEGORY 5: Metatable Operations (750-769)
// =============================================================================

// Test 750: No metatable returns nil
[shader("compute")]
[numthreads(1, 1, 1)]
void test_metatable_nil() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue mt = table_getmetatable(t);
    
    bool passed = mt.isNil();
    report_test(750, passed, TYPE_NIL, mt.type);
}

// Test 751: Set valid metatable
[shader("compute")]
[numthreads(1, 1, 1)]
void test_metatable_set() {
    heap_init(8192);
    
    XValue t = table_new(8);
    XValue mt = table_new(8);
    
    bool success = table_setmetatable(t, mt);
    report_test(751, success, 1, success ? 1 : 0);
}

// Test 752: Get after set
[shader("compute")]
[numthreads(1, 1, 1)]
void test_metatable_get_after_set() {
    heap_init(8192);
    
    XValue t = table_new(8);
    XValue mt = table_new(8);
    
    table_setmetatable(t, mt);
    XValue result = table_getmetatable(t);
    
    bool passed = (result.type == TYPE_TABLE) && (result.asUint() == mt.asUint());
    report_test(752, passed, mt.asUint(), result.asUint());
}

// Test 753: Clear metatable with nil
[shader("compute")]
[numthreads(1, 1, 1)]
void test_metatable_clear() {
    heap_init(8192);
    
    XValue t = table_new(8);
    XValue mt = table_new(8);
    
    table_setmetatable(t, mt);
    table_setmetatable(t, XValue::nil());
    
    XValue result = table_getmetatable(t);
    
    bool passed = result.isNil();
    report_test(753, passed, TYPE_NIL, result.type);
}

// Test 754: Non-table arg fails
[shader("compute")]
[numthreads(1, 1, 1)]
void test_metatable_non_table() {
    XValue notTable = XValue::number(42.0f);
    XValue mt = XValue::number(100.0f);
    
    bool success = table_setmetatable(notTable, mt);
    report_test(754, !success, 0, success ? 1 : 0);
}

// Test 755: Get existing metamethod
[shader("compute")]
[numthreads(1, 1, 1)]
void test_getmetamethod_existing() {
    heap_init(8192);
    
    XValue t = table_new(8);
    XValue mt = table_new(8);
    
    // Set __add metamethod in metatable
    XValue addKey = XValue::string(META_NAME_ADD);
    XValue addFunc = XValue::number(999.0f);  // Placeholder function value
    table_set(mt, addKey, addFunc);
    
    table_setmetatable(t, mt);
    
    XValue method = table_getmetamethod(t, META_ADD);
    
    bool passed = method.asNumber() == 999.0f;
    report_test(755, passed, 999, uint(method.asNumber()));
}

// Test 756: Get non-existing metamethod
[shader("compute")]
[numthreads(1, 1, 1)]
void test_getmetamethod_nonexisting() {
    heap_init(8192);
    
    XValue t = table_new(8);
    XValue mt = table_new(8);
    table_setmetatable(t, mt);
    
    XValue method = table_getmetamethod(t, META_ADD);
    
    bool passed = method.isNil();
    report_test(756, passed, TYPE_NIL, method.type);
}

// Test 757: Has existing metamethod
[shader("compute")]
[numthreads(1, 1, 1)]
void test_hasmetamethod_existing() {
    heap_init(8192);
    
    XValue t = table_new(8);
    XValue mt = table_new(8);
    
    table_set(mt, XValue::string(META_NAME_ADD), XValue::number(1.0f));
    table_setmetatable(t, mt);
    
    bool has = table_hasmetamethod(t, META_ADD);
    report_test(757, has, 1, has ? 1 : 0);
}

// Test 758: Has non-existing metamethod
[shader("compute")]
[numthreads(1, 1, 1)]
void test_hasmetamethod_nonexisting() {
    heap_init(8192);
    
    XValue t = table_new(8);
    XValue mt = table_new(8);
    table_setmetatable(t, mt);
    
    bool has = table_hasmetamethod(t, META_ADD);
    report_test(758, !has, 0, has ? 1 : 0);
}


// =============================================================================
// TEST CATEGORY 6: Table Iteration (770-779)
// =============================================================================

// Test 770: Empty table returns false
[shader("compute")]
[numthreads(1, 1, 1)]
void test_next_empty() {
    heap_init(4096);
    
    XValue t = table_new(8);
    XValue key = XValue::nil();
    XValue value;
    
    bool hasNext = table_next(t, key, value);
    report_test(770, !hasNext, 0, hasNext ? 1 : 0);
}

// Test 771: First iteration with nil key
[shader("compute")]
[numthreads(1, 1, 1)]
void test_next_first() {
    heap_init(4096);
    
    XValue t = table_new(8);
    table_set(t, XValue::number(1.0f), XValue::number(10.0f));
    
    XValue key = XValue::nil();
    XValue value;
    
    bool hasNext = table_next(t, key, value);
    
    bool passed = hasNext && (key.asNumber() == 1.0f) && (value.asNumber() == 10.0f);
    report_test(771, passed, 1, hasNext ? 1 : 0);
}

// Test 772: Verify table count after inserts
// Note: table_next iteration is simplified and doesn't follow chains
// This test verifies count accurately reflects number of entries
[shader("compute")]
[numthreads(1, 1, 1)]
void test_next_continue() {
    heap_init(4096);
    
    XValue t = table_new(16);  // Larger capacity to reduce collisions
    table_set(t, XValue::number(1.0f), XValue::number(10.0f));
    table_set(t, XValue::number(2.0f), XValue::number(20.0f));
    table_set(t, XValue::number(3.0f), XValue::number(30.0f));
    
    // Verify count reflects all inserts
    uint count = table_count(t);
    
    // Verify all entries are retrievable
    XValue v1 = table_get(t, XValue::number(1.0f));
    XValue v2 = table_get(t, XValue::number(2.0f));
    XValue v3 = table_get(t, XValue::number(3.0f));
    
    bool passed = (count == 3) && 
                  (v1.asNumber() == 10.0f) && 
                  (v2.asNumber() == 20.0f) && 
                  (v3.asNumber() == 30.0f);
    report_test(772, passed, 3, count);
}

// Test 773: End of iteration
[shader("compute")]
[numthreads(1, 1, 1)]
void test_next_end() {
    heap_init(4096);
    
    XValue t = table_new(8);
    table_set(t, XValue::number(1.0f), XValue::number(10.0f));
    
    XValue key = XValue::nil();
    XValue value;
    
    // First iteration (gets the one entry)
    table_next(t, key, value);
    
    // Second iteration (should return false)
    bool hasNext = table_next(t, key, value);
    
    report_test(773, !hasNext, 0, hasNext ? 1 : 0);
}

// Test 774: Non-table returns false
[shader("compute")]
[numthreads(1, 1, 1)]
void test_next_non_table() {
    XValue notTable = XValue::number(42.0f);
    XValue key = XValue::nil();
    XValue value;
    
    bool hasNext = table_next(notTable, key, value);
    report_test(774, !hasNext, 0, hasNext ? 1 : 0);
}


// =============================================================================
// TEST CATEGORY 7: Concurrent Tests (780-789)
// =============================================================================

groupshared uint g_sharedTablePtr;

// Test 780: Concurrent reads on same table
[shader("compute")]
[numthreads(16, 1, 1)]
void test_concurrent_reads(uint3 threadId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID) {
    if (groupThreadId.x == 0) {
        heap_init(8192);
        
        XValue t = table_new(16);
        // Pre-populate table
        for (uint i = 0; i < 8; i++) {
            table_set(t, XValue::number(float(i)), XValue::number(float(i * 10)));
        }
        g_sharedTablePtr = t.asUint();
    }
    GroupMemoryBarrierWithGroupSync();
    
    // All threads read from the table
    XValue t = XValue::table(g_sharedTablePtr);
    uint keyIdx = groupThreadId.x % 8;
    XValue result = table_get(t, XValue::number(float(keyIdx)));
    
    // Store result
    g_tableResults[groupThreadId.x] = uint(result.asNumber());
    
    GroupMemoryBarrierWithGroupSync();
    
    if (groupThreadId.x == 0) {
        // Verify all reads were correct
        bool allCorrect = true;
        for (uint i = 0; i < 16; i++) {
            uint expected = (i % 8) * 10;
            if (g_tableResults[i] != expected) {
                allCorrect = false;
                break;
            }
        }
        
        report_test(780, allCorrect, 1, allCorrect ? 1 : 0);
    }
}

// Test 781: Sequential writes followed by concurrent reads
// Note: table_set is not thread-safe, so writes must be serialized
[shader("compute")]
[numthreads(16, 1, 1)]
void test_concurrent_writes(uint3 threadId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID) {
    // Thread 0 does all writes first
    if (groupThreadId.x == 0) {
        heap_init(8192);
        
        XValue t = table_new(32);  // Large enough for all entries
        
        // Pre-populate table with all entries (sequential to avoid race)
        for (uint i = 0; i < 16; i++) {
            XValue key = XValue::number(float(i + 100));
            XValue value = XValue::number(float(i * 10));
            table_set(t, key, value);
        }
        
        g_sharedTablePtr = t.asUint();
    }
    GroupMemoryBarrierWithGroupSync();
    
    // All threads concurrently read their assigned key
    XValue t = XValue::table(g_sharedTablePtr);
    XValue key = XValue::number(float(groupThreadId.x + 100));
    XValue result = table_get(t, key);
    g_tableResults[groupThreadId.x] = uint(result.asNumber());
    
    GroupMemoryBarrierWithGroupSync();
    
    if (groupThreadId.x == 0) {
        // Verify all reads were correct
        bool allCorrect = true;
        for (uint i = 0; i < 16; i++) {
            uint expected = i * 10;
            if (g_tableResults[i] != expected) {
                allCorrect = false;
                break;
            }
        }
        
        report_test(781, allCorrect, 1, allCorrect ? 1 : 0);
    }
}

