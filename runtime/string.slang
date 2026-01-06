// XScript String Pool (32-bit)
// Interned string storage with hash-based lookup

// =============================================================================
// Constants
// =============================================================================

static const uint STRING_POOL_CAPACITY = 1024;
static const uint STRING_MAX_LENGTH = 256;

// String header offsets (in uints)
static const uint STRING_OFF_HASH = 0;
static const uint STRING_OFF_LENGTH = 1;
static const uint STRING_OFF_REFCOUNT = 2;
static const uint STRING_OFF_NEXT = 3;
static const uint STRING_OFF_CHARS = 4;  // Characters start here

// =============================================================================
// String Pool State
// =============================================================================

struct StringPoolState {
    uint count;          // Number of strings in pool
    uint nextFree;       // Next allocation offset
    uint totalSize;      // Total pool size
};

// Global string pool buffers
RWStructuredBuffer<uint> g_stringData;      // String data storage
RWStructuredBuffer<uint> g_stringHashTable; // Hash table for lookup
RWStructuredBuffer<StringPoolState> g_stringState;

// =============================================================================
// Hash Function
// =============================================================================

// FNV-1a hash for characters
uint string_hash_chars(uint ptr, uint length) {
    uint hash = 2166136261u;  // FNV offset basis
    
    uint numUints = (length + 3) / 4;
    for (uint i = 0; i < numUints; i++) {
        uint chars = g_stringData[ptr + STRING_OFF_CHARS + i];
        
        hash ^= (chars & 0xFF);
        hash *= 16777619u;
        hash ^= ((chars >> 8) & 0xFF);
        hash *= 16777619u;
        hash ^= ((chars >> 16) & 0xFF);
        hash *= 16777619u;
        hash ^= ((chars >> 24) & 0xFF);
        hash *= 16777619u;
    }
    
    return hash;
}

// =============================================================================
// String Operations
// =============================================================================

// Get string length
uint string_length(uint stringIndex) {
    if (stringIndex == 0) return 0;
    return g_stringData[stringIndex + STRING_OFF_LENGTH];
}

// Get string hash
uint string_gethash(uint stringIndex) {
    if (stringIndex == 0) return 0;
    return g_stringData[stringIndex + STRING_OFF_HASH];
}

// Compare two strings for equality
bool string_equals(uint a, uint b) {
    if (a == b) return true;
    if (a == 0 || b == 0) return false;
    
    uint lenA = string_length(a);
    uint lenB = string_length(b);
    
    if (lenA != lenB) return false;
    
    // Compare hashes first (fast path)
    if (string_gethash(a) != string_gethash(b)) return false;
    
    // Compare characters
    uint numUints = (lenA + 3) / 4;
    for (uint i = 0; i < numUints; i++) {
        if (g_stringData[a + STRING_OFF_CHARS + i] != g_stringData[b + STRING_OFF_CHARS + i]) {
            return false;
        }
    }
    
    return true;
}

// Get character at index
uint string_charat(uint stringIndex, uint index) {
    if (stringIndex == 0) return 0;
    
    uint length = string_length(stringIndex);
    if (index >= length) return 0;
    
    uint uintIndex = index / 4;
    uint byteIndex = index % 4;
    
    uint data = g_stringData[stringIndex + STRING_OFF_CHARS + uintIndex];
    return (data >> (byteIndex * 8)) & 0xFF;
}

// =============================================================================
// String Pool Operations
// =============================================================================

// Intern a string (find or create)
// chars: buffer containing character data
// length: string length
// Returns string index
uint string_intern(RWStructuredBuffer<uint> chars, uint offset, uint length) {
    if (length == 0) return 0;
    if (length > STRING_MAX_LENGTH) length = STRING_MAX_LENGTH;
    
    // Calculate hash
    uint hash = 2166136261u;
    uint numUints = (length + 3) / 4;
    
    for (uint i = 0; i < numUints; i++) {
        uint c = chars[offset + i];
        hash ^= (c & 0xFF);
        hash *= 16777619u;
        hash ^= ((c >> 8) & 0xFF);
        hash *= 16777619u;
        hash ^= ((c >> 16) & 0xFF);
        hash *= 16777619u;
        hash ^= ((c >> 24) & 0xFF);
        hash *= 16777619u;
    }
    
    // Look up in hash table
    uint bucket = hash % STRING_POOL_CAPACITY;
    uint existing = g_stringHashTable[bucket];
    
    while (existing != 0) {
        // Check if this is the same string
        if (g_stringData[existing + STRING_OFF_HASH] == hash &&
            g_stringData[existing + STRING_OFF_LENGTH] == length) {
            // Compare characters
            bool match = true;
            for (uint i = 0; i < numUints && match; i++) {
                if (g_stringData[existing + STRING_OFF_CHARS + i] != chars[offset + i]) {
                    match = false;
                }
            }
            
            if (match) {
                // Found existing string, increment refcount
                InterlockedAdd(g_stringData[existing + STRING_OFF_REFCOUNT], 1);
                return existing;
            }
        }
        
        existing = g_stringData[existing + STRING_OFF_NEXT];
    }
    
    // String not found, create new entry
    uint headerSize = STRING_OFF_CHARS;
    uint dataSize = numUints;
    uint totalSize = headerSize + dataSize;
    
    // Allocate space
    uint ptr;
    InterlockedAdd(g_stringState[0].nextFree, totalSize, ptr);
    
    if (ptr + totalSize > g_stringState[0].totalSize) {
        // Pool full
        return 0;
    }
    
    // Write header
    g_stringData[ptr + STRING_OFF_HASH] = hash;
    g_stringData[ptr + STRING_OFF_LENGTH] = length;
    g_stringData[ptr + STRING_OFF_REFCOUNT] = 1;
    
    // Link into hash table
    uint oldHead;
    InterlockedExchange(g_stringHashTable[bucket], ptr, oldHead);
    g_stringData[ptr + STRING_OFF_NEXT] = oldHead;
    
    // Copy characters
    for (uint i = 0; i < numUints; i++) {
        g_stringData[ptr + STRING_OFF_CHARS + i] = chars[offset + i];
    }
    
    InterlockedAdd(g_stringState[0].count, 1);
    
    return ptr;
}

// Reference counting for strings
void string_incref(uint stringIndex) {
    if (stringIndex == 0) return;
    InterlockedAdd(g_stringData[stringIndex + STRING_OFF_REFCOUNT], 1);
}

bool string_decref(uint stringIndex) {
    if (stringIndex == 0) return false;
    
    uint oldCount;
    InterlockedAdd(g_stringData[stringIndex + STRING_OFF_REFCOUNT], -1, oldCount);
    
    // Note: We don't actually free strings from the pool for simplicity
    // In a production implementation, we would need GC for the string pool
    return oldCount == 1;
}

// =============================================================================
// String Concatenation
// =============================================================================

// Concatenate two strings
uint string_concat(uint a, uint b) {
    if (a == 0) return b;
    if (b == 0) return a;
    
    uint lenA = string_length(a);
    uint lenB = string_length(b);
    uint totalLen = lenA + lenB;
    
    if (totalLen > STRING_MAX_LENGTH) {
        totalLen = STRING_MAX_LENGTH;
        lenB = totalLen - lenA;
    }
    
    // Create temporary buffer for concatenated string
    // This is simplified - in practice we'd need proper temp allocation
    uint numUints = (totalLen + 3) / 4;
    
    // For now, return 0 (concatenation not fully implemented)
    // Full implementation would allocate temp buffer and call string_intern
    return 0;
}

// =============================================================================
// String Pool Initialization
// =============================================================================

void string_pool_init(uint totalSize) {
    g_stringState[0].count = 0;
    g_stringState[0].nextFree = 1;  // 0 is reserved for empty/null
    g_stringState[0].totalSize = totalSize;
    
    // Clear hash table
    for (uint i = 0; i < STRING_POOL_CAPACITY; i++) {
        g_stringHashTable[i] = 0;
    }
}
