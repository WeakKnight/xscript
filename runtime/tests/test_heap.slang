// XScript Heap Memory Tests (32-bit)
// Tests for heap allocation, reference counting, and memory access

// =============================================================================
// Constants
// =============================================================================

static const uint HEAP_BLOCK_HEADER_SIZE = 4;  // 4 uints = 16 bytes
static const uint HEAP_ALIGNMENT = 4;          // 4-uint alignment

static const uint BLOCK_FLAG_FREE = 0x01;
static const uint BLOCK_FLAG_MARKED = 0x02;

// Size classes for pool allocation
static const uint SIZE_CLASS_SMALL = 16;   // <= 64 bytes
static const uint SIZE_CLASS_MEDIUM = 64;  // <= 256 bytes
static const uint SIZE_CLASS_LARGE = 256;  // <= 1024 bytes

// =============================================================================
// Heap Block Header
// =============================================================================

struct HeapBlockHeader {
    uint size;       // Block size in uints (including header)
    uint refCount;   // Reference count
    uint flags;      // Block flags
    uint next;       // Next block in free list (0 = end)
};

// =============================================================================
// Heap Allocator State
// =============================================================================

struct HeapAllocator {
    uint freeListSmall;   // Free list head for small blocks
    uint freeListMedium;  // Free list head for medium blocks
    uint freeListLarge;   // Free list head for large blocks
    uint freeListHuge;    // Free list head for huge blocks
    uint nextFree;        // Next allocation offset
    uint totalSize;       // Total heap size
    uint usedSize;        // Currently used size
    uint allocCount;      // Number of allocations
};

// =============================================================================
// Heap Memory Buffers (provided by Python test runner)
// =============================================================================

RWStructuredBuffer<uint> g_heapMemory;
RWStructuredBuffer<HeapAllocator> g_heapState;

// =============================================================================
// Helper Functions
// =============================================================================

uint align_up(uint size, uint alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

uint get_size_class(uint size) {
    if (size <= SIZE_CLASS_SMALL) return 0;
    if (size <= SIZE_CLASS_MEDIUM) return 1;
    if (size <= SIZE_CLASS_LARGE) return 2;
    return 3;
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

// =============================================================================
// Heap Functions
// =============================================================================

void heap_init(uint totalSizeInUints) {
    g_heapState[0].freeListSmall = 0;
    g_heapState[0].freeListMedium = 0;
    g_heapState[0].freeListLarge = 0;
    g_heapState[0].freeListHuge = 0;
    g_heapState[0].nextFree = 1;  // Start at 1, 0 is reserved for null
    g_heapState[0].totalSize = totalSizeInUints;
    g_heapState[0].usedSize = 0;
    g_heapState[0].allocCount = 0;
}

uint heap_alloc(uint size) {
    uint blockSize = align_up(size + HEAP_BLOCK_HEADER_SIZE * 4, HEAP_ALIGNMENT * 4);
    uint blockSizeInUints = blockSize / 4;
    
    // Allocate from end of heap (simplified - no free list reuse in tests)
    uint offset = g_heapState[0].nextFree;
    g_heapState[0].nextFree += blockSizeInUints;
    
    // Check if we exceeded heap size
    if (g_heapState[0].nextFree > g_heapState[0].totalSize) {
        g_heapState[0].nextFree = offset;  // Rollback
        return 0;
    }
    
    // Write block header
    HeapBlockHeader header;
    header.size = blockSizeInUints;
    header.refCount = 1;
    header.flags = 0;
    header.next = 0;
    write_block_header(offset, header);
    
    g_heapState[0].usedSize += blockSizeInUints;
    g_heapState[0].allocCount += 1;
    
    return offset + HEAP_BLOCK_HEADER_SIZE;
}

void heap_free(uint ptr) {
    if (ptr == 0) return;
    
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    HeapBlockHeader header = read_block_header(blockOffset);
    
    // Mark as free
    header.flags = BLOCK_FLAG_FREE;
    header.refCount = 0;
    write_block_header(blockOffset, header);
    
    g_heapState[0].allocCount -= 1;
}

void heap_incref(uint ptr) {
    if (ptr == 0) return;
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    g_heapMemory[blockOffset + 1] += 1;
}

bool heap_decref(uint ptr) {
    if (ptr == 0) return false;
    
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    uint oldCount = g_heapMemory[blockOffset + 1];
    g_heapMemory[blockOffset + 1] -= 1;
    
    if (oldCount == 1) {
        heap_free(ptr);
        return true;
    }
    
    return false;
}

uint heap_get_refcount(uint ptr) {
    if (ptr == 0) return 0;
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    return g_heapMemory[blockOffset + 1];
}

uint heap_read_uint(uint ptr, uint offset) {
    return g_heapMemory[ptr + offset];
}

void heap_write_uint(uint ptr, uint offset, uint value) {
    g_heapMemory[ptr + offset] = value;
}

float heap_read_float(uint ptr, uint offset) {
    return asfloat(g_heapMemory[ptr + offset]);
}

void heap_write_float(uint ptr, uint offset, float value) {
    g_heapMemory[ptr + offset] = asuint(value);
}

void heap_get_stats(out uint totalSize, out uint usedSize, out uint allocCount) {
    totalSize = g_heapState[0].totalSize;
    usedSize = g_heapState[0].usedSize;
    allocCount = g_heapState[0].allocCount;
}

// =============================================================================
// Test Result Buffer
// =============================================================================

struct TestResult {
    uint testId;
    uint passed;
    uint expected;
    uint actual;
};

RWStructuredBuffer<TestResult> g_testResults;
RWStructuredBuffer<uint> g_testCount;

void report_heap_test(uint testId, bool passed, uint expected, uint actual) {
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
// Helper Function Tests (500-509)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_align_up_basic() {
    // Test: align 5 to 4-boundary = 8
    uint result = align_up(5, 4);
    bool passed = (result == 8);
    report_heap_test(500, passed, 8, result);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_align_up_already_aligned() {
    // Test: align 16 to 4-boundary = 16
    uint result = align_up(16, 4);
    bool passed = (result == 16);
    report_heap_test(501, passed, 16, result);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_align_up_zero() {
    // Test: align 0 to 4-boundary = 0
    uint result = align_up(0, 4);
    bool passed = (result == 0);
    report_heap_test(502, passed, 0, result);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_size_class_small() {
    // Test: size 16 = class 0 (small)
    uint result = get_size_class(16);
    bool passed = (result == 0);
    report_heap_test(503, passed, 0, result);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_size_class_medium() {
    // Test: size 64 = class 1 (medium)
    uint result = get_size_class(64);
    bool passed = (result == 1);
    report_heap_test(504, passed, 1, result);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_size_class_large() {
    // Test: size 256 = class 2 (large)
    uint result = get_size_class(256);
    bool passed = (result == 2);
    report_heap_test(505, passed, 2, result);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_size_class_huge() {
    // Test: size 1000 = class 3 (huge)
    uint result = get_size_class(1000);
    bool passed = (result == 3);
    report_heap_test(506, passed, 3, result);
}

// =============================================================================
// Heap Initialization Tests (510-519)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_init() {
    // Initialize heap with 1024 uints
    heap_init(1024);
    
    bool passed = (g_heapState[0].nextFree == 1) &&
                  (g_heapState[0].totalSize == 1024) &&
                  (g_heapState[0].usedSize == 0) &&
                  (g_heapState[0].allocCount == 0);
    report_heap_test(510, passed, 1024, g_heapState[0].totalSize);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_get_stats_after_init() {
    heap_init(2048);
    
    uint totalSize, usedSize, allocCount;
    heap_get_stats(totalSize, usedSize, allocCount);
    
    bool passed = (totalSize == 2048) && (usedSize == 0) && (allocCount == 0);
    report_heap_test(511, passed, 2048, totalSize);
}

// =============================================================================
// Basic Allocation Tests (520-529)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_alloc_single() {
    heap_init(1024);
    
    // Allocate 16 bytes
    uint ptr = heap_alloc(16);
    
    // Pointer should be non-zero and at offset 1 + 4 (header) = 5
    bool passed = (ptr != 0) && (ptr == 1 + HEAP_BLOCK_HEADER_SIZE);
    report_heap_test(520, passed, 1 + HEAP_BLOCK_HEADER_SIZE, ptr);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_alloc_updates_stats() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    
    uint totalSize, usedSize, allocCount;
    heap_get_stats(totalSize, usedSize, allocCount);
    
    bool passed = (allocCount == 1) && (usedSize > 0);
    report_heap_test(521, passed, 1, allocCount);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_alloc_multiple() {
    heap_init(1024);
    
    uint ptr1 = heap_alloc(16);
    uint ptr2 = heap_alloc(16);
    uint ptr3 = heap_alloc(16);
    
    // All pointers should be different and non-zero
    bool passed = (ptr1 != 0) && (ptr2 != 0) && (ptr3 != 0) &&
                  (ptr1 != ptr2) && (ptr2 != ptr3) && (ptr1 != ptr3);
    report_heap_test(522, passed, 3, (ptr1 != 0) + (ptr2 != 0) + (ptr3 != 0));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_alloc_sequential() {
    heap_init(1024);
    
    uint ptr1 = heap_alloc(16);
    uint ptr2 = heap_alloc(16);
    
    // ptr2 should come after ptr1 with proper spacing
    bool passed = (ptr2 > ptr1);
    report_heap_test(523, passed, 1, ptr2 > ptr1 ? 1 : 0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_alloc_fails_when_full() {
    // Initialize with tiny heap
    heap_init(10);
    
    // Try to allocate more than available
    uint ptr = heap_alloc(100);
    
    // Should fail and return 0
    bool passed = (ptr == 0);
    report_heap_test(524, passed, 0, ptr);
}

// =============================================================================
// Memory Access Tests (530-539)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_write_read_uint() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    
    heap_write_uint(ptr, 0, 42);
    uint value = heap_read_uint(ptr, 0);
    
    bool passed = (value == 42);
    report_heap_test(530, passed, 42, value);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_write_read_multiple_uint() {
    heap_init(1024);
    
    uint ptr = heap_alloc(32);
    
    heap_write_uint(ptr, 0, 100);
    heap_write_uint(ptr, 1, 200);
    heap_write_uint(ptr, 2, 300);
    
    uint v0 = heap_read_uint(ptr, 0);
    uint v1 = heap_read_uint(ptr, 1);
    uint v2 = heap_read_uint(ptr, 2);
    
    bool passed = (v0 == 100) && (v1 == 200) && (v2 == 300);
    report_heap_test(531, passed, 100, v0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_write_read_float() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    
    heap_write_float(ptr, 0, 3.14159f);
    float value = heap_read_float(ptr, 0);
    
    bool passed = abs(value - 3.14159f) < 0.0001f;
    report_heap_test(532, passed, asuint(3.14159f), asuint(value));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_write_read_negative_float() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    
    heap_write_float(ptr, 0, -42.5f);
    float value = heap_read_float(ptr, 0);
    
    bool passed = (value == -42.5f);
    report_heap_test(533, passed, asuint(-42.5f), asuint(value));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_separate_allocations() {
    heap_init(1024);
    
    uint ptr1 = heap_alloc(16);
    uint ptr2 = heap_alloc(16);
    
    // Write different values to each allocation
    heap_write_uint(ptr1, 0, 111);
    heap_write_uint(ptr2, 0, 222);
    
    // Verify they don't interfere
    uint v1 = heap_read_uint(ptr1, 0);
    uint v2 = heap_read_uint(ptr2, 0);
    
    bool passed = (v1 == 111) && (v2 == 222);
    report_heap_test(534, passed, 111, v1);
}

// =============================================================================
// Reference Counting Tests (540-549)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_initial_refcount() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    uint refcount = heap_get_refcount(ptr);
    
    // Initial refcount should be 1
    bool passed = (refcount == 1);
    report_heap_test(540, passed, 1, refcount);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_incref() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    heap_incref(ptr);
    
    uint refcount = heap_get_refcount(ptr);
    
    // After incref, refcount should be 2
    bool passed = (refcount == 2);
    report_heap_test(541, passed, 2, refcount);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_incref_multiple() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    heap_incref(ptr);
    heap_incref(ptr);
    heap_incref(ptr);
    
    uint refcount = heap_get_refcount(ptr);
    
    // After 3 increfs, refcount should be 4
    bool passed = (refcount == 4);
    report_heap_test(542, passed, 4, refcount);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_decref_no_free() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    heap_incref(ptr);  // refcount = 2
    
    bool freed = heap_decref(ptr);  // refcount = 1
    uint refcount = heap_get_refcount(ptr);
    
    // Should not be freed, refcount should be 1
    bool passed = (!freed) && (refcount == 1);
    report_heap_test(543, passed, 1, refcount);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_decref_with_free() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    
    uint allocCountBefore = g_heapState[0].allocCount;
    bool freed = heap_decref(ptr);  // refcount 1 -> 0, should free
    uint allocCountAfter = g_heapState[0].allocCount;
    
    // Should be freed, allocCount should decrease
    bool passed = freed && (allocCountAfter == allocCountBefore - 1);
    report_heap_test(544, passed, allocCountBefore - 1, allocCountAfter);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_get_refcount_null() {
    // Getting refcount of null pointer should return 0
    uint refcount = heap_get_refcount(0);
    
    bool passed = (refcount == 0);
    report_heap_test(545, passed, 0, refcount);
}

// =============================================================================
// Free and Reuse Tests (550-559)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_free() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    
    // Get stats using the helper function (which works in other tests)
    uint totalSize, usedSize, allocCount;
    heap_get_stats(totalSize, usedSize, allocCount);
    
    heap_free(ptr);
    
    uint totalSize2, usedSize2, allocCount2;
    heap_get_stats(totalSize2, usedSize2, allocCount2);
    
    // After alloc, count should be 1; after free, count should be 0
    bool passed = (allocCount == 1) && (allocCount2 == 0);
    report_heap_test(550, passed, 0, allocCount2);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_free_null() {
    heap_init(1024);
    
    uint allocCountBefore = g_heapState[0].allocCount;
    heap_free(0);  // Should do nothing
    uint allocCountAfter = g_heapState[0].allocCount;
    
    bool passed = (allocCountAfter == allocCountBefore);
    report_heap_test(551, passed, allocCountBefore, allocCountAfter);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_block_marked_free() {
    heap_init(1024);
    
    uint ptr = heap_alloc(16);
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    
    heap_free(ptr);
    
    HeapBlockHeader header = read_block_header(blockOffset);
    
    bool passed = (header.flags & BLOCK_FLAG_FREE) != 0;
    report_heap_test(552, passed, BLOCK_FLAG_FREE, header.flags & BLOCK_FLAG_FREE);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_heap_multiple_alloc_free() {
    heap_init(1024);
    
    // Allocate 3 blocks
    uint ptr1 = heap_alloc(16);
    uint ptr2 = heap_alloc(16);
    uint ptr3 = heap_alloc(16);
    
    // Free middle one
    heap_free(ptr2);
    
    uint allocCount = g_heapState[0].allocCount;
    
    bool passed = (allocCount == 2);
    report_heap_test(553, passed, 2, allocCount);
}

// =============================================================================
// Atomic Heap Functions (for concurrent tests)
// =============================================================================

// Thread-safe allocation using atomic operations
uint heap_alloc_atomic(uint size) {
    uint blockSize = align_up(size + HEAP_BLOCK_HEADER_SIZE * 4, HEAP_ALIGNMENT * 4);
    uint blockSizeInUints = blockSize / 4;
    
    // Atomically reserve space
    uint offset;
    InterlockedAdd(g_heapState[0].nextFree, blockSizeInUints, offset);
    
    // Check if we exceeded heap size
    if (offset + blockSizeInUints > g_heapState[0].totalSize) {
        // Allocation failed - can't easily rollback in concurrent scenario
        return 0;
    }
    
    // Write block header
    HeapBlockHeader header;
    header.size = blockSizeInUints;
    header.refCount = 1;
    header.flags = 0;
    header.next = 0;
    write_block_header(offset, header);
    
    InterlockedAdd(g_heapState[0].usedSize, blockSizeInUints);
    InterlockedAdd(g_heapState[0].allocCount, 1);
    
    return offset + HEAP_BLOCK_HEADER_SIZE;
}

// Thread-safe incref
void heap_incref_atomic(uint ptr) {
    if (ptr == 0) return;
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    InterlockedAdd(g_heapMemory[blockOffset + 1], 1);
}

// Thread-safe decref (returns old refcount)
uint heap_decref_atomic(uint ptr) {
    if (ptr == 0) return 0;
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    uint oldCount;
    InterlockedAdd(g_heapMemory[blockOffset + 1], -1, oldCount);
    return oldCount;
}

// =============================================================================
// Buffer for storing concurrent test results
// =============================================================================

RWStructuredBuffer<uint> g_allocResults;  // Store pointers from concurrent allocations

// =============================================================================
// Concurrent Allocation Tests (560-569)
// =============================================================================

// Test: 32 threads each allocate one block
[shader("compute")]
[numthreads(32, 1, 1)]
void test_concurrent_alloc_32(uint3 threadId : SV_DispatchThreadID) {
    // Only thread 0 initializes
    if (threadId.x == 0) {
        heap_init(8192);  // Large heap for 32 allocations
        g_testCount[0] = 0;
    }
    
    // Barrier to ensure init completes
    GroupMemoryBarrierWithGroupSync();
    
    // Each thread allocates
    uint ptr = heap_alloc_atomic(16);
    
    // Store result
    g_allocResults[threadId.x] = ptr;
    
    // Barrier before checking
    GroupMemoryBarrierWithGroupSync();
    
    // Only thread 0 checks results
    if (threadId.x == 0) {
        uint successCount = 0;
        uint uniqueCount = 0;
        
        // Count successful allocations
        for (uint i = 0; i < 32; i++) {
            if (g_allocResults[i] != 0) {
                successCount++;
            }
        }
        
        // Check all pointers are unique (no overlap)
        bool allUnique = true;
        for (uint i = 0; i < 32; i++) {
            for (uint j = i + 1; j < 32; j++) {
                if (g_allocResults[i] != 0 && g_allocResults[i] == g_allocResults[j]) {
                    allUnique = false;
                }
            }
        }
        
        // Check allocCount matches
        uint totalSize, usedSize, allocCount;
        heap_get_stats(totalSize, usedSize, allocCount);
        
        bool passed = (successCount == 32) && allUnique && (allocCount == 32);
        report_heap_test(560, passed, 32, successCount);
    }
}

// Test: 64 threads each allocate one block
[shader("compute")]
[numthreads(64, 1, 1)]
void test_concurrent_alloc_64(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x == 0) {
        heap_init(16384);  // Large heap for 64 allocations
        g_testCount[0] = 0;
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    uint ptr = heap_alloc_atomic(16);
    g_allocResults[threadId.x] = ptr;
    
    GroupMemoryBarrierWithGroupSync();
    
    if (threadId.x == 0) {
        uint successCount = 0;
        for (uint i = 0; i < 64; i++) {
            if (g_allocResults[i] != 0) successCount++;
        }
        
        uint totalSize, usedSize, allocCount;
        heap_get_stats(totalSize, usedSize, allocCount);
        
        bool passed = (successCount == 64) && (allocCount == 64);
        report_heap_test(561, passed, 64, successCount);
    }
}

// Test: Concurrent incref from 32 threads
[shader("compute")]
[numthreads(32, 1, 1)]
void test_concurrent_incref_32(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x == 0) {
        heap_init(1024);
        g_testCount[0] = 0;
        // Allocate one block
        g_allocResults[0] = heap_alloc_atomic(16);
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    uint ptr = g_allocResults[0];
    
    // All 32 threads increment refcount
    heap_incref_atomic(ptr);
    
    GroupMemoryBarrierWithGroupSync();
    
    if (threadId.x == 0) {
        // Initial refcount is 1, plus 32 increments = 33
        uint refcount = heap_get_refcount(ptr);
        bool passed = (refcount == 33);
        report_heap_test(562, passed, 33, refcount);
    }
}

// Test: Concurrent decref from 16 threads (with initial refcount set high)
[shader("compute")]
[numthreads(16, 1, 1)]
void test_concurrent_decref_16(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x == 0) {
        heap_init(1024);
        g_testCount[0] = 0;
        
        // Allocate and set high initial refcount
        uint ptr = heap_alloc_atomic(16);
        g_allocResults[0] = ptr;
        
        // Increment refcount to 20 (start at 1, add 19)
        for (uint i = 0; i < 19; i++) {
            heap_incref_atomic(ptr);
        }
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    uint ptr = g_allocResults[0];
    
    // All 16 threads decrement refcount
    heap_decref_atomic(ptr);
    
    GroupMemoryBarrierWithGroupSync();
    
    if (threadId.x == 0) {
        // Started at 20, minus 16 = 4
        uint refcount = heap_get_refcount(ptr);
        bool passed = (refcount == 4);
        report_heap_test(563, passed, 4, refcount);
    }
}

// Test: Multiple groups allocating concurrently
[shader("compute")]
[numthreads(8, 1, 1)]
void test_concurrent_alloc_multigroup(uint3 threadId : SV_DispatchThreadID) {
    // This test is dispatched with (4, 1, 1) groups = 32 total threads
    uint globalId = threadId.x;
    
    if (globalId == 0) {
        heap_init(8192);
        g_testCount[0] = 0;
    }
    
    AllMemoryBarrierWithGroupSync();
    
    uint ptr = heap_alloc_atomic(16);
    g_allocResults[globalId] = ptr;
    
    AllMemoryBarrierWithGroupSync();
    
    if (globalId == 0) {
        uint successCount = 0;
        for (uint i = 0; i < 8; i++) {  // Only count first group
            if (g_allocResults[i] != 0) successCount++;
        }
        
        bool passed = (successCount == 8);
        report_heap_test(564, passed, 8, successCount);
    }
}

// Test: Stress test with different allocation sizes
[shader("compute")]
[numthreads(16, 1, 1)]
void test_concurrent_alloc_varying_sizes(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x == 0) {
        heap_init(16384);
        g_testCount[0] = 0;
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // Each thread allocates different size based on thread ID
    uint allocSize = 8 + (threadId.x * 4);  // 8, 12, 16, 20, ... 68 bytes
    uint ptr = heap_alloc_atomic(allocSize);
    g_allocResults[threadId.x] = ptr;
    
    GroupMemoryBarrierWithGroupSync();
    
    if (threadId.x == 0) {
        uint successCount = 0;
        for (uint i = 0; i < 16; i++) {
            if (g_allocResults[i] != 0) successCount++;
        }
        
        uint totalSize, usedSize, allocCount;
        heap_get_stats(totalSize, usedSize, allocCount);
        
        bool passed = (successCount == 16) && (allocCount == 16);
        report_heap_test(565, passed, 16, successCount);
    }
}

// =============================================================================
// Run All Tests
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void run_all_heap_tests() {
    g_testCount[0] = 0;
    
    // Helper function tests
    {
        uint result = align_up(5, 4);
        report_heap_test(500, result == 8, 8, result);
    }
    {
        uint result = align_up(16, 4);
        report_heap_test(501, result == 16, 16, result);
    }
    {
        uint result = get_size_class(16);
        report_heap_test(503, result == 0, 0, result);
    }
    {
        uint result = get_size_class(64);
        report_heap_test(504, result == 1, 1, result);
    }
    
    // Init test
    {
        heap_init(1024);
        report_heap_test(510, g_heapState[0].totalSize == 1024, 1024, g_heapState[0].totalSize);
    }
    
    // Allocation test
    {
        heap_init(1024);
        uint ptr = heap_alloc(16);
        report_heap_test(520, ptr != 0, 1, ptr != 0 ? 1 : 0);
    }
    
    // Memory access test
    {
        heap_init(1024);
        uint ptr = heap_alloc(16);
        heap_write_uint(ptr, 0, 42);
        uint value = heap_read_uint(ptr, 0);
        report_heap_test(530, value == 42, 42, value);
    }
    
    // Refcount test
    {
        heap_init(1024);
        uint ptr = heap_alloc(16);
        uint refcount = heap_get_refcount(ptr);
        report_heap_test(540, refcount == 1, 1, refcount);
    }
}

