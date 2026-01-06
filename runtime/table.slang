// XScript Table Implementation (32-bit)
// Hash table with metatable support

import value;
import heap;

// =============================================================================
// Constants
// =============================================================================

static const uint TABLE_INITIAL_CAPACITY = 8;
static const uint TABLE_LOAD_FACTOR_NUM = 3;   // Load factor = 3/4
static const uint TABLE_LOAD_FACTOR_DEN = 4;

// Table structure offsets (in uints from table pointer)
static const uint TABLE_OFF_CAPACITY = 0;
static const uint TABLE_OFF_COUNT = 1;
static const uint TABLE_OFF_METATABLE = 2;
static const uint TABLE_OFF_ARRAY_SIZE = 3;
static const uint TABLE_OFF_ENTRIES = 4;  // Entries start here

// Entry structure size (in uints)
// Each entry: key(3 uints) + value(3 uints) + next(1 uint) = 7 uints
static const uint ENTRY_SIZE = 7;
static const uint ENTRY_OFF_KEY = 0;
static const uint ENTRY_OFF_VALUE = 3;
static const uint ENTRY_OFF_NEXT = 6;

// =============================================================================
// Meta Method Indices
// =============================================================================

static const uint META_ADD       = 0;
static const uint META_SUB       = 1;
static const uint META_MUL       = 2;
static const uint META_DIV       = 3;
static const uint META_MOD       = 4;
static const uint META_POW       = 5;
static const uint META_NEG       = 6;
static const uint META_EQ        = 7;
static const uint META_LT        = 8;
static const uint META_LE        = 9;
static const uint META_INDEX     = 10;
static const uint META_NEWINDEX  = 11;
static const uint META_CALL      = 12;
static const uint META_TOSTRING  = 13;
static const uint META_LEN       = 14;
static const uint META_GC        = 15;
static const uint META_COUNT     = 16;

// Meta method name string indices (pre-defined in string pool)
static const uint META_NAME_ADD       = 0x1000;  // "__add"
static const uint META_NAME_SUB       = 0x1001;  // "__sub"
static const uint META_NAME_MUL       = 0x1002;  // etc.
static const uint META_NAME_DIV       = 0x1003;
static const uint META_NAME_MOD       = 0x1004;
static const uint META_NAME_POW       = 0x1005;
static const uint META_NAME_NEG       = 0x1006;
static const uint META_NAME_EQ        = 0x1007;
static const uint META_NAME_LT        = 0x1008;
static const uint META_NAME_LE        = 0x1009;
static const uint META_NAME_INDEX     = 0x100A;
static const uint META_NAME_NEWINDEX  = 0x100B;
static const uint META_NAME_CALL      = 0x100C;
static const uint META_NAME_TOSTRING  = 0x100D;
static const uint META_NAME_LEN       = 0x100E;
static const uint META_NAME_GC        = 0x100F;

// =============================================================================
// Hash Function (32-bit)
// =============================================================================

// FNV-1a hash for XValue (32-bit version)
uint xvalue_hash(XValue v) {
    uint hash = 2166136261u;  // FNV offset basis
    
    // Hash type
    hash ^= v.type;
    hash *= 16777619u;  // FNV prime
    
    // Hash data (32-bit)
    hash ^= v.data;
    hash *= 16777619u;
    
    return hash;
}

// =============================================================================
// Table Operations
// =============================================================================

// Create a new table
XValue table_new(uint initialCapacity) {
    if (initialCapacity == 0) {
        initialCapacity = TABLE_INITIAL_CAPACITY;
    }
    
    // Calculate size: header + entries
    uint headerSize = TABLE_OFF_ENTRIES;
    uint entriesSize = initialCapacity * ENTRY_SIZE;
    uint totalSize = (headerSize + entriesSize) * 4;  // in bytes
    
    // Allocate from heap
    uint ptr = heap_alloc(totalSize);
    if (ptr == 0) {
        return XValue::nil();
    }
    
    // Initialize header
    heap_write_uint(ptr, TABLE_OFF_CAPACITY, initialCapacity);
    heap_write_uint(ptr, TABLE_OFF_COUNT, 0);
    heap_write_uint(ptr, TABLE_OFF_METATABLE, 0);
    heap_write_uint(ptr, TABLE_OFF_ARRAY_SIZE, 0);
    
    // Initialize entries to nil
    for (uint i = 0; i < initialCapacity; i++) {
        uint entryOffset = TABLE_OFF_ENTRIES + i * ENTRY_SIZE;
        // Key = nil
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_KEY, TYPE_NIL);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_KEY + 1, 0);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_KEY + 2, 0);
        // Value = nil
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_VALUE, TYPE_NIL);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_VALUE + 1, 0);
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_VALUE + 2, 0);
        // Next = 0 (no chain)
        heap_write_uint(ptr, entryOffset + ENTRY_OFF_NEXT, 0);
    }
    
    return XValue::table(ptr);
}

// Read XValue from heap at offset (32-bit version)
XValue read_xvalue(uint ptr, uint offset) {
    XValue v;
    v.type = heap_read_uint(ptr, offset);
    v.flags = heap_read_uint(ptr, offset + 1);
    v.data = heap_read_uint(ptr, offset + 2);
    return v;
}

// Write XValue to heap at offset (32-bit version)
void write_xvalue(uint ptr, uint offset, XValue v) {
    heap_write_uint(ptr, offset, v.type);
    heap_write_uint(ptr, offset + 1, v.flags);
    heap_write_uint(ptr, offset + 2, v.data);
}

// Get value from table
XValue table_get(XValue t, XValue key) {
    if (t.type != TYPE_TABLE) {
        return XValue::nil();
    }
    
    uint ptr = t.asUint();
    uint capacity = heap_read_uint(ptr, TABLE_OFF_CAPACITY);
    
    if (capacity == 0) {
        return XValue::nil();
    }
    
    // Calculate bucket index
    uint hash = xvalue_hash(key);
    uint bucket = hash % capacity;
    
    // Search in bucket
    uint entryOffset = TABLE_OFF_ENTRIES + bucket * ENTRY_SIZE;
    
    // Check if bucket is empty
    XValue entryKey = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
    if (entryKey.isNil()) {
        return XValue::nil();
    }
    
    // Linear probing with chaining
    while (true) {
        if (xvalue_equals(entryKey, key)) {
            return read_xvalue(ptr, entryOffset + ENTRY_OFF_VALUE);
        }
        
        uint next = heap_read_uint(ptr, entryOffset + ENTRY_OFF_NEXT);
        if (next == 0) {
            break;
        }
        
        entryOffset = TABLE_OFF_ENTRIES + next * ENTRY_SIZE;
        entryKey = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
    }
    
    return XValue::nil();
}

// Set value in table
bool table_set(XValue t, XValue key, XValue value) {
    if (t.type != TYPE_TABLE || key.isNil()) {
        return false;
    }
    
    uint ptr = t.asUint();
    uint capacity = heap_read_uint(ptr, TABLE_OFF_CAPACITY);
    uint count = heap_read_uint(ptr, TABLE_OFF_COUNT);
    
    // Calculate bucket index
    uint hash = xvalue_hash(key);
    uint bucket = hash % capacity;
    
    uint entryOffset = TABLE_OFF_ENTRIES + bucket * ENTRY_SIZE;
    
    // Check if bucket is empty
    XValue entryKey = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
    
    if (entryKey.isNil()) {
        // Empty slot, insert here
        write_xvalue(ptr, entryOffset + ENTRY_OFF_KEY, key);
        write_xvalue(ptr, entryOffset + ENTRY_OFF_VALUE, value);
        heap_write_uint(ptr, TABLE_OFF_COUNT, count + 1);
        return true;
    }
    
    // Search for existing key or find end of chain
    uint prevOffset = entryOffset;
    while (true) {
        if (xvalue_equals(entryKey, key)) {
            // Key exists, update value
            write_xvalue(ptr, entryOffset + ENTRY_OFF_VALUE, value);
            return true;
        }
        
        uint next = heap_read_uint(ptr, entryOffset + ENTRY_OFF_NEXT);
        if (next == 0) {
            break;
        }
        
        prevOffset = entryOffset;
        entryOffset = TABLE_OFF_ENTRIES + next * ENTRY_SIZE;
        entryKey = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
    }
    
    // Need to add to chain - find free slot
    for (uint i = 0; i < capacity; i++) {
        uint freeOffset = TABLE_OFF_ENTRIES + i * ENTRY_SIZE;
        XValue freeKey = read_xvalue(ptr, freeOffset + ENTRY_OFF_KEY);
        
        if (freeKey.isNil()) {
            // Found free slot
            write_xvalue(ptr, freeOffset + ENTRY_OFF_KEY, key);
            write_xvalue(ptr, freeOffset + ENTRY_OFF_VALUE, value);
            heap_write_uint(ptr, entryOffset + ENTRY_OFF_NEXT, i);
            heap_write_uint(ptr, TABLE_OFF_COUNT, count + 1);
            return true;
        }
    }
    
    // Table is full (should resize, but simplified for now)
    return false;
}

// Get table count
uint table_count(XValue t) {
    if (t.type != TYPE_TABLE) {
        return 0;
    }
    return heap_read_uint(t.asUint(), TABLE_OFF_COUNT);
}

// =============================================================================
// Metatable Operations
// =============================================================================

// Get metatable of a table
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

// Set metatable of a table
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

// Get metamethod from metatable
XValue table_getmetamethod(XValue t, uint metaIndex) {
    XValue mt = table_getmetatable(t);
    if (mt.isNil()) {
        return XValue::nil();
    }
    
    // Create key for metamethod (using pre-defined string indices)
    XValue key = XValue::string(META_NAME_ADD + metaIndex);
    
    return table_get(mt, key);
}

// Check if table has metamethod
bool table_hasmetamethod(XValue t, uint metaIndex) {
    XValue method = table_getmetamethod(t, metaIndex);
    return !method.isNil();
}

// =============================================================================
// Table Iteration (for pairs/ipairs)
// =============================================================================

// Get next key-value pair after given key
// Returns true if there is a next pair, false if end of table
bool table_next(XValue t, inout XValue key, out XValue value) {
    if (t.type != TYPE_TABLE) {
        return false;
    }
    
    uint ptr = t.asUint();
    uint capacity = heap_read_uint(ptr, TABLE_OFF_CAPACITY);
    
    uint startBucket = 0;
    
    // If key is not nil, find current position and start from next
    if (!key.isNil()) {
        uint hash = xvalue_hash(key);
        startBucket = (hash % capacity) + 1;
    }
    
    // Search for next non-nil entry
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
