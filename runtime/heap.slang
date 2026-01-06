// XScript Heap Memory Management
// Pool-based allocator for GPU-side dynamic memory (32-bit)

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
// Heap Memory Buffer
// =============================================================================

// Global heap memory buffer (set by host)
RWStructuredBuffer<uint> g_heapMemory;

// Global allocator state
RWStructuredBuffer<HeapAllocator> g_heapState;

// =============================================================================
// Helper Functions
// =============================================================================

// Align size up to alignment boundary
uint align_up(uint size, uint alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Get size class for a given size
uint get_size_class(uint size) {
    if (size <= SIZE_CLASS_SMALL) return 0;
    if (size <= SIZE_CLASS_MEDIUM) return 1;
    if (size <= SIZE_CLASS_LARGE) return 2;
    return 3;
}

// Get free list head for size class
uint get_free_list_head(uint sizeClass) {
    switch (sizeClass) {
        case 0: return g_heapState[0].freeListSmall;
        case 1: return g_heapState[0].freeListMedium;
        case 2: return g_heapState[0].freeListLarge;
        default: return g_heapState[0].freeListHuge;
    }
}

// Set free list head for size class
void set_free_list_head(uint sizeClass, uint value) {
    switch (sizeClass) {
        case 0: g_heapState[0].freeListSmall = value; break;
        case 1: g_heapState[0].freeListMedium = value; break;
        case 2: g_heapState[0].freeListLarge = value; break;
        default: g_heapState[0].freeListHuge = value; break;
    }
}

// Read block header
HeapBlockHeader read_block_header(uint offset) {
    HeapBlockHeader header;
    header.size = g_heapMemory[offset];
    header.refCount = g_heapMemory[offset + 1];
    header.flags = g_heapMemory[offset + 2];
    header.next = g_heapMemory[offset + 3];
    return header;
}

// Write block header
void write_block_header(uint offset, HeapBlockHeader header) {
    g_heapMemory[offset] = header.size;
    g_heapMemory[offset + 1] = header.refCount;
    g_heapMemory[offset + 2] = header.flags;
    g_heapMemory[offset + 3] = header.next;
}

// =============================================================================
// Allocation Functions
// =============================================================================

// Allocate memory from heap
// Returns offset to usable memory (after header), or 0 on failure
uint heap_alloc(uint size) {
    // Calculate total block size including header
    uint blockSize = align_up(size + HEAP_BLOCK_HEADER_SIZE * 4, HEAP_ALIGNMENT * 4);
    uint blockSizeInUints = blockSize / 4;
    
    // Get size class
    uint sizeClass = get_size_class(blockSizeInUints);
    
    // Try to allocate from free list
    uint freeListHead = get_free_list_head(sizeClass);
    
    if (freeListHead != 0) {
        // Pop from free list
        HeapBlockHeader header = read_block_header(freeListHead);
        
        uint oldHead;
        InterlockedExchange(g_heapState[0].freeListSmall + sizeClass, header.next, oldHead);
        
        // Update header
        header.refCount = 1;
        header.flags = 0;
        header.next = 0;
        write_block_header(freeListHead, header);
        
        InterlockedAdd(g_heapState[0].allocCount, 1);
        
        return freeListHead + HEAP_BLOCK_HEADER_SIZE;
    }
    
    // Allocate from end of heap
    uint offset;
    InterlockedAdd(g_heapState[0].nextFree, blockSizeInUints, offset);
    
    // Check if we exceeded heap size
    if (offset + blockSizeInUints > g_heapState[0].totalSize) {
        // Allocation failed - rollback
        InterlockedAdd(g_heapState[0].nextFree, -int(blockSizeInUints));
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

// Free memory back to heap
void heap_free(uint ptr) {
    if (ptr == 0) return;
    
    // Get block header offset
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    HeapBlockHeader header = read_block_header(blockOffset);
    
    // Get size class
    uint sizeClass = get_size_class(header.size);
    
    // Mark as free
    header.flags = BLOCK_FLAG_FREE;
    
    // Add to free list
    uint oldHead;
    InterlockedExchange(g_heapState[0].freeListSmall + sizeClass, blockOffset, oldHead);
    header.next = oldHead;
    
    write_block_header(blockOffset, header);
    
    InterlockedAdd(g_heapState[0].allocCount, -1);
}

// =============================================================================
// Reference Counting
// =============================================================================

// Increment reference count
void heap_incref(uint ptr) {
    if (ptr == 0) return;
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    InterlockedAdd(g_heapMemory[blockOffset + 1], 1);
}

// Decrement reference count, free if zero
// Returns true if object was freed
bool heap_decref(uint ptr) {
    if (ptr == 0) return false;
    
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    
    uint oldCount;
    InterlockedAdd(g_heapMemory[blockOffset + 1], -1, oldCount);
    
    if (oldCount == 1) {
        heap_free(ptr);
        return true;
    }
    
    return false;
}

// Get reference count
uint heap_get_refcount(uint ptr) {
    if (ptr == 0) return 0;
    uint blockOffset = ptr - HEAP_BLOCK_HEADER_SIZE;
    return g_heapMemory[blockOffset + 1];
}

// =============================================================================
// Memory Access
// =============================================================================

// Read uint from heap
uint heap_read_uint(uint ptr, uint offset) {
    return g_heapMemory[ptr + offset];
}

// Write uint to heap
void heap_write_uint(uint ptr, uint offset, uint value) {
    g_heapMemory[ptr + offset] = value;
}

// Read float from heap
float heap_read_float(uint ptr, uint offset) {
    return asfloat(g_heapMemory[ptr + offset]);
}

// Write float to heap
void heap_write_float(uint ptr, uint offset, float value) {
    g_heapMemory[ptr + offset] = asuint(value);
}

// =============================================================================
// Heap Initialization
// =============================================================================

// Initialize heap (called once at startup)
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

// Get heap statistics
void heap_get_stats(out uint totalSize, out uint usedSize, out uint allocCount) {
    totalSize = g_heapState[0].totalSize;
    usedSize = g_heapState[0].usedSize;
    allocCount = g_heapState[0].allocCount;
}

