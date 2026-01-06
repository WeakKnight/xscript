// XScript GC Tests (32-bit)
// Tests for reference counting, automatic cleanup, weak references, and write barriers

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

static const uint ENTRY_SIZE = 7;
static const uint ENTRY_OFF_KEY = 0;
static const uint ENTRY_OFF_VALUE = 3;
static const uint ENTRY_OFF_NEXT = 6;

// =============================================================================
// GC Constants
// =============================================================================

static const uint FLAG_MARKED = 0x01;
static const uint FLAG_WEAK = 0x02;

// =============================================================================
// Structures
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

struct HeapBlockHeader {
    uint size;
    uint refCount;
    uint flags;
    uint next;
};

struct GCState {
    uint enabled;
    uint threshold;
    uint allocsSinceGC;
    uint totalCollected;
};

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
// GPU Buffers
// =============================================================================

RWStructuredBuffer<uint> g_heapMemory;
RWStructuredBuffer<HeapAllocator> g_heapState;
RWStructuredBuffer<GCState> g_gcState;
RWStructuredBuffer<uint> g_testResults;

// =============================================================================
// Heap Helper Functions
// =============================================================================

uint align_up(uint size, uint alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

HeapBlockHeader read_block_header(uint offset) {
    HeapBlockHeader header;
    header.size = g_heapMemory[offset];
    header.refCount = g_heapMemory[offset + 1];
    header.flags = g_heapMemory[offset + 2];
    header.next = g_heapMemory[offset + 3];
    return header;
}

void write_block_header(uint offset, HeapBlockHeader header) {
    g_heapMemory[offset] = header.size;
    g_heapMemory[offset + 1] = header.refCount;
    g_heapMemory[offset + 2] = header.flags;
    g_heapMemory[offset + 3] = header.next;
}

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

uint heap_alloc(uint size) {
    uint blockSize = align_up(size + HEAP_BLOCK_HEADER_SIZE * 4, HEAP_ALIGNMENT * 4);
    uint blockSizeInUints = blockSize / 4;
    
    uint offset = g_heapState[0].nextFree;
    
    if (offset + blockSizeInUints > g_heapState[0].totalSize) {
        return 0;
    }
    
    HeapBlockHeader header;
    header.size = blockSizeInUints;
    header.refCount = 1;
    header.flags = 0;
    header.next = 0;
    write_block_header(offset, header);
    
    g_heapState[0].nextFree = offset + blockSizeInUints;
    g_heapState[0].usedSize += blockSizeInUints;
    g_heapState[0].allocCount++;
    
    return offset + HEAP_BLOCK_HEADER_SIZE;
}

uint heap_get_refcount(uint ptr) {
    if (ptr < HEAP_BLOCK_HEADER_SIZE) return 0;
    uint headerOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    return g_heapMemory[headerOffset + 1];
}

void heap_incref(uint ptr) {
    if (ptr < HEAP_BLOCK_HEADER_SIZE) return;
    uint headerOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    InterlockedAdd(g_heapMemory[headerOffset + 1], 1);
}

bool heap_decref(uint ptr) {
    if (ptr < HEAP_BLOCK_HEADER_SIZE) return false;
    uint headerOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    
    uint oldCount;
    InterlockedAdd(g_heapMemory[headerOffset + 1], -1, oldCount);
    
    if (oldCount == 1) {
        // Mark as free
        g_heapMemory[headerOffset + 2] |= 0x01;
        return true;
    }
    return false;
}

uint heap_read_uint(uint ptr, uint offset) {
    return g_heapMemory[ptr + offset];
}

void heap_write_uint(uint ptr, uint offset, uint value) {
    g_heapMemory[ptr + offset] = value;
}

// =============================================================================
// String Pool Stubs (for testing)
// =============================================================================

void string_incref(uint index) {
    // Stub for testing
}

bool string_decref(uint index) {
    // Stub for testing
    return false;
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
// Table Creation (minimal for testing)
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

void table_setmetatable(XValue t, XValue mt) {
    if (t.type != TYPE_TABLE) return;
    uint ptr = t.asUint();
    if (mt.type == TYPE_TABLE) {
        heap_write_uint(ptr, TABLE_OFF_METATABLE, mt.asUint());
    } else {
        heap_write_uint(ptr, TABLE_OFF_METATABLE, 0);
    }
}

// =============================================================================
// GC Functions
// =============================================================================

// Worklist for iterative decref (avoid recursion on GPU)
static const uint GC_WORKLIST_SIZE = 32;

void gc_init() {
    g_gcState[0].enabled = 1;
    g_gcState[0].threshold = 1000;
    g_gcState[0].allocsSinceGC = 0;
    g_gcState[0].totalCollected = 0;
}

void xvalue_incref(inout XValue v) {
    if (v.type < TYPE_STRING) {
        return;
    }
    
    switch (v.type) {
        case TYPE_STRING:
            string_incref(v.asUint());
            break;
        case TYPE_TABLE:
        case TYPE_FUNCTION:
        case TYPE_USERDATA:
            heap_incref(v.asUint());
            break;
    }
}

// Iterative decref that processes tables without recursion
// Uses a local worklist to handle nested tables
void xvalue_decref(inout XValue v) {
    if (v.type < TYPE_STRING) {
        return;
    }
    
    // Worklist for tables that need processing
    uint worklist[GC_WORKLIST_SIZE];
    uint worklistHead = 0;
    uint worklistTail = 0;
    
    // Add initial value to worklist if it's a table
    if (v.type == TYPE_TABLE) {
        worklist[worklistTail] = v.asUint();
        worklistTail++;
    } else if (v.type == TYPE_STRING) {
        string_decref(v.asUint());
        return;
    } else {
        bool freed = heap_decref(v.asUint());
        if (freed) {
            InterlockedAdd(g_gcState[0].totalCollected, 1);
        }
        return;
    }
    
    // Process worklist iteratively
    while (worklistHead < worklistTail) {
        uint ptr = worklist[worklistHead];
        worklistHead++;
        
        uint oldCount = heap_get_refcount(ptr);
        
        if (oldCount <= 1) {
            // About to free - collect children first
            uint capacity = heap_read_uint(ptr, TABLE_OFF_CAPACITY);
            
            for (uint i = 0; i < capacity && worklistTail < GC_WORKLIST_SIZE; i++) {
                uint entryOffset = TABLE_OFF_ENTRIES + i * ENTRY_SIZE;
                
                XValue key = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
                if (!key.isNil()) {
                    // Decref key (non-table types only, to avoid recursion)
                    if (key.type == TYPE_STRING) {
                        string_decref(key.asUint());
                    } else if (key.type >= TYPE_TABLE) {
                        // Add to worklist if table
                        if (key.type == TYPE_TABLE && worklistTail < GC_WORKLIST_SIZE) {
                            worklist[worklistTail] = key.asUint();
                            worklistTail++;
                        } else {
                            heap_decref(key.asUint());
                        }
                    }
                    
                    XValue value = read_xvalue(ptr, entryOffset + ENTRY_OFF_VALUE);
                    if (value.type == TYPE_STRING) {
                        string_decref(value.asUint());
                    } else if (value.type >= TYPE_TABLE) {
                        if (value.type == TYPE_TABLE && worklistTail < GC_WORKLIST_SIZE) {
                            worklist[worklistTail] = value.asUint();
                            worklistTail++;
                        } else {
                            heap_decref(value.asUint());
                        }
                    }
                }
            }
            
            // Handle metatable
            uint mtPtr = heap_read_uint(ptr, TABLE_OFF_METATABLE);
            if (mtPtr != 0 && worklistTail < GC_WORKLIST_SIZE) {
                worklist[worklistTail] = mtPtr;
                worklistTail++;
            }
        }
        
        // Now decref the table itself
        bool freed = heap_decref(ptr);
        if (freed) {
            InterlockedAdd(g_gcState[0].totalCollected, 1);
        }
    }
}

void xvalue_assign(inout XValue dst, XValue src) {
    if (dst.type == src.type && dst.data == src.data) {
        return;
    }
    
    xvalue_incref(src);
    xvalue_decref(dst);
    dst = src;
}

void gc_write_barrier(XValue container, XValue oldValue, XValue newValue) {
    xvalue_incref(newValue);
    xvalue_decref(oldValue);
}

XValue xvalue_weak(XValue v) {
    XValue weak = v;
    weak.flags |= FLAG_WEAK;
    return weak;
}

bool xvalue_is_weak(XValue v) {
    return (v.flags & FLAG_WEAK) != 0;
}

XValue xvalue_strengthen(XValue weak) {
    if (!xvalue_is_weak(weak)) {
        return weak;
    }
    
    if (weak.type >= TYPE_STRING) {
        uint refCount = heap_get_refcount(weak.asUint());
        if (refCount == 0) {
            return XValue::nil();
        }
    }
    
    XValue strong = weak;
    strong.flags &= ~FLAG_WEAK;
    xvalue_incref(strong);
    return strong;
}

void gc_enable() {
    g_gcState[0].enabled = 1;
}

void gc_disable() {
    g_gcState[0].enabled = 0;
}

bool gc_is_enabled() {
    return g_gcState[0].enabled != 0;
}

// =============================================================================
// Test: Reference Counting Basics - Incref/Decref
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_incref_decref() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Test 1: Create table, refcount should be 1
    XValue t = table_new(4);
    uint refCount = heap_get_refcount(t.asUint());
    if (refCount == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Incref should increase refcount to 2
    xvalue_incref(t);
    refCount = heap_get_refcount(t.asUint());
    if (refCount == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Decref should decrease refcount to 1
    xvalue_decref(t);
    refCount = heap_get_refcount(t.asUint());
    if (refCount == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Incref nil should be no-op (no crash)
    XValue nilVal = XValue::nil();
    xvalue_incref(nilVal);  // Should not crash
    passed++;
    
    // Test 5: Incref number should be no-op (no crash)
    XValue numVal = XValue::number(42.0f);
    xvalue_incref(numVal);  // Should not crash
    passed++;
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Decref Frees Object
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_decref_frees() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create table
    XValue t = table_new(4);
    uint ptr = t.asUint();
    
    // Test 1: Initial refcount is 1
    uint refCount = heap_get_refcount(ptr);
    if (refCount == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Decref should free and return true (via gc_decref_table)
    uint collectedBefore = g_gcState[0].totalCollected;
    xvalue_decref(t);
    uint collectedAfter = g_gcState[0].totalCollected;
    
    if (collectedAfter > collectedBefore) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Refcount should be 0 after free
    refCount = heap_get_refcount(ptr);
    if (refCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Multiple References
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_multiple_refs() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create table
    XValue t = table_new(4);
    uint ptr = t.asUint();
    
    // Add 5 more references
    for (uint i = 0; i < 5; i++) {
        xvalue_incref(t);
    }
    
    // Test 1: Refcount should be 6
    uint refCount = heap_get_refcount(ptr);
    if (refCount == 6) {
        passed++;
    } else {
        failed++;
    }
    
    // Remove 3 references
    for (uint i = 0; i < 3; i++) {
        xvalue_decref(t);
    }
    
    // Test 2: Refcount should be 3
    refCount = heap_get_refcount(ptr);
    if (refCount == 3) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Object should still exist
    uint collectedBefore = g_gcState[0].totalCollected;
    if (refCount > 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Remove remaining references
    xvalue_decref(t);
    xvalue_decref(t);
    xvalue_decref(t);
    
    // Test 4: Object should be freed now
    uint collectedAfter = g_gcState[0].totalCollected;
    if (collectedAfter > collectedBefore) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Value Assignment
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_xvalue_assign() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create two tables
    XValue t1 = table_new(4);
    XValue t2 = table_new(4);
    uint ptr1 = t1.asUint();
    uint ptr2 = t2.asUint();
    
    // Test 1: Both have refcount 1
    if (heap_get_refcount(ptr1) == 1 && heap_get_refcount(ptr2) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Assign t2 to a variable holding t1
    XValue holder = t1;
    xvalue_incref(holder);  // holder now has a reference
    
    // Test 2: t1 refcount should be 2
    if (heap_get_refcount(ptr1) == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Now assign t2 to holder using xvalue_assign
    xvalue_assign(holder, t2);
    
    // Test 3: t1 refcount should be back to 1 (holder released it)
    if (heap_get_refcount(ptr1) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: t2 refcount should be 2 (original + holder)
    if (heap_get_refcount(ptr2) == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: holder should now point to t2
    if (holder.asUint() == ptr2) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Assign Same Value
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_assign_same_value() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create table
    XValue t = table_new(4);
    uint ptr = t.asUint();
    
    // Get initial refcount
    uint initialRefCount = heap_get_refcount(ptr);
    
    // Test 1: Initial refcount is 1
    if (initialRefCount == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Assign same value to itself
    xvalue_assign(t, t);
    
    // Test 2: Refcount should remain unchanged
    uint afterRefCount = heap_get_refcount(ptr);
    if (afterRefCount == initialRefCount) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Value should be unchanged
    if (t.asUint() == ptr) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Assignment Chain
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_assign_chain() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create three tables
    XValue a = table_new(4);
    XValue b = table_new(4);
    XValue c = table_new(4);
    
    uint ptrA = a.asUint();
    uint ptrB = b.asUint();
    uint ptrC = c.asUint();
    
    // All start with refcount 1
    // Now: A = B
    xvalue_assign(a, b);
    
    // Test 1: a now points to b's table
    if (a.asUint() == ptrB) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: b's refcount is 2 (b + a)
    if (heap_get_refcount(ptrB) == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Original a's table was freed (refcount was 1, now 0)
    if (heap_get_refcount(ptrA) == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Now: B = C
    xvalue_assign(b, c);
    
    // Test 4: ptrB refcount is 1 (only a still holds it)
    if (heap_get_refcount(ptrB) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: ptrC refcount is 2 (c + b)
    if (heap_get_refcount(ptrC) == 2) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Table with Metatable Cleanup
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_metatable_cleanup() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create table and metatable
    XValue t = table_new(4);
    XValue mt = table_new(4);
    uint ptrT = t.asUint();
    uint ptrMT = mt.asUint();
    
    // Set metatable (this should incref the metatable)
    table_setmetatable(t, mt);
    xvalue_incref(mt);  // Manual incref for metatable
    
    // Test 1: Metatable refcount should be 2 (original + from table)
    if (heap_get_refcount(ptrMT) == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Release original metatable reference
    xvalue_decref(mt);
    
    // Test 2: Metatable refcount should be 1 (only from table)
    if (heap_get_refcount(ptrMT) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Now destroy the table
    uint collectedBefore = g_gcState[0].totalCollected;
    xvalue_decref(t);
    uint collectedAfter = g_gcState[0].totalCollected;
    
    // Test 3: Table was collected
    if (collectedAfter > collectedBefore) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Metatable was also collected (cascading cleanup)
    if (heap_get_refcount(ptrMT) == 0) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Weak Reference No Incref
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_weak_ref_no_incref() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create table
    XValue t = table_new(4);
    uint ptr = t.asUint();
    
    // Test 1: Initial refcount is 1
    if (heap_get_refcount(ptr) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Create weak reference
    XValue weak = xvalue_weak(t);
    
    // Test 2: Weak ref has FLAG_WEAK set
    if (xvalue_is_weak(weak)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Refcount should still be 1 (weak ref doesn't incref)
    if (heap_get_refcount(ptr) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Original is not weak
    if (!xvalue_is_weak(t)) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Weak Reference Strengthen
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_weak_strengthen() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create table
    XValue t = table_new(4);
    uint ptr = t.asUint();
    
    // Create weak reference
    XValue weak = xvalue_weak(t);
    
    // Test 1: Initial refcount is 1
    if (heap_get_refcount(ptr) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Strengthen the weak reference
    XValue strong = xvalue_strengthen(weak);
    
    // Test 2: Strong ref is not weak
    if (!xvalue_is_weak(strong)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Refcount should now be 2 (original + strengthened)
    if (heap_get_refcount(ptr) == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Strong points to same table
    if (strong.asUint() == ptr) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Weak Reference After Target Freed
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_weak_after_free() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create table
    XValue t = table_new(4);
    uint ptr = t.asUint();
    
    // Create weak reference
    XValue weak = xvalue_weak(t);
    
    // Test 1: Can strengthen while target exists
    XValue strong1 = xvalue_strengthen(weak);
    if (!strong1.isNil() && strong1.asUint() == ptr) {
        passed++;
    } else {
        failed++;
    }
    
    // Release the strong reference we just got
    xvalue_decref(strong1);
    
    // Now release the original reference
    xvalue_decref(t);
    
    // Test 2: Target should be freed (refcount 0)
    if (heap_get_refcount(ptr) == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Strengthening weak ref after target freed returns nil
    XValue strong2 = xvalue_strengthen(weak);
    if (strong2.isNil()) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Write Barrier
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_write_barrier() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Create container and two values
    XValue container = table_new(4);
    XValue oldValue = table_new(4);
    XValue newValue = table_new(4);
    
    uint ptrOld = oldValue.asUint();
    uint ptrNew = newValue.asUint();
    
    // All start with refcount 1
    // Simulate: old value is in container, now being replaced by new value
    xvalue_incref(oldValue);  // Container holds old value
    
    // Test 1: Old value refcount is 2
    if (heap_get_refcount(ptrOld) == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: New value refcount is 1
    if (heap_get_refcount(ptrNew) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Apply write barrier
    gc_write_barrier(container, oldValue, newValue);
    
    // Test 3: Old value refcount decreased to 1
    if (heap_get_refcount(ptrOld) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: New value refcount increased to 2
    if (heap_get_refcount(ptrNew) == 2) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: GC Enable/Disable
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_gc_enable_disable() {
    uint passed = 0;
    uint failed = 0;
    
    gc_init();
    
    // Test 1: GC is enabled by default
    if (gc_is_enabled()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Disable GC
    gc_disable();
    if (!gc_is_enabled()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Enable GC
    gc_enable();
    if (gc_is_enabled()) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: GC Statistics
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_gc_stats() {
    uint passed = 0;
    uint failed = 0;
    
    heap_init(10000);
    gc_init();
    
    // Test 1: Initial collected count is 0
    if (g_gcState[0].totalCollected == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Create and free a table
    XValue t1 = table_new(4);
    xvalue_decref(t1);
    
    // Test 2: Collected count increased
    if (g_gcState[0].totalCollected == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Create and free another table
    XValue t2 = table_new(4);
    xvalue_decref(t2);
    
    // Test 3: Collected count increased again
    if (g_gcState[0].totalCollected == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Create table but don't free (still has reference)
    XValue t3 = table_new(4);
    
    // Test 4: Collected count unchanged
    if (g_gcState[0].totalCollected == 2) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

