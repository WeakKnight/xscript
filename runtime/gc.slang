// XScript Garbage Collection (32-bit)
// Reference counting with cycle detection support

// =============================================================================
// GC State
// =============================================================================

struct GCState {
    uint enabled;         // GC enabled flag
    uint threshold;       // Allocation threshold before GC
    uint allocsSinceGC;   // Allocations since last GC
    uint totalCollected;  // Total objects collected
};

RWStructuredBuffer<GCState> g_gcState;

// =============================================================================
// Reference Counting
// =============================================================================

// Increment reference count for a value
void xvalue_incref(inout XValue v) {
    if (v.type < TYPE_STRING) {
        // Non-heap types don't need ref counting
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

// Decrement reference count for a value
void xvalue_decref(inout XValue v) {
    if (v.type < TYPE_STRING) {
        return;
    }
    
    bool freed = false;
    
    switch (v.type) {
        case TYPE_STRING:
            freed = string_decref(v.asUint());
            break;
        case TYPE_TABLE:
            freed = gc_decref_table(v);
            break;
        case TYPE_FUNCTION:
        case TYPE_USERDATA:
            freed = heap_decref(v.asUint());
            break;
    }
    
    if (freed) {
        InterlockedAdd(g_gcState[0].totalCollected, 1);
    }
}

// Decrement reference count for table (recursive)
bool gc_decref_table(XValue t) {
    if (t.type != TYPE_TABLE) return false;
    
    uint ptr = t.asUint();
    uint oldCount = heap_get_refcount(ptr);
    
    if (oldCount <= 1) {
        // About to free - first decref all contained values
        uint capacity = heap_read_uint(ptr, TABLE_OFF_CAPACITY);
        
        for (uint i = 0; i < capacity; i++) {
            uint entryOffset = TABLE_OFF_ENTRIES + i * ENTRY_SIZE;
            
            XValue key = read_xvalue(ptr, entryOffset + ENTRY_OFF_KEY);
            if (!key.isNil()) {
                xvalue_decref(key);
                
                XValue value = read_xvalue(ptr, entryOffset + ENTRY_OFF_VALUE);
                xvalue_decref(value);
            }
        }
        
        // Decref metatable if any
        uint mtPtr = heap_read_uint(ptr, TABLE_OFF_METATABLE);
        if (mtPtr != 0) {
            XValue mt = XValue::table(mtPtr);
            xvalue_decref(mt);
        }
    }
    
    return heap_decref(ptr);
}

// =============================================================================
// Value Assignment with Reference Counting
// =============================================================================

// Assign value with proper reference counting
void xvalue_assign(inout XValue dst, XValue src) {
    // Check if same value
    if (dst.type == src.type && dst.data == src.data) {
        return;
    }
    
    // Increment source refcount first (in case src references dst)
    xvalue_incref(src);
    
    // Decrement destination refcount
    xvalue_decref(dst);
    
    // Copy value
    dst = src;
}

// =============================================================================
// Write Barrier
// =============================================================================

// Write barrier for table modifications
// Call this when writing a reference into a table
void gc_write_barrier(XValue container, XValue oldValue, XValue newValue) {
    // For reference counting, we just need proper incref/decref
    xvalue_incref(newValue);
    xvalue_decref(oldValue);
}

// =============================================================================
// Cycle Detection (Simplified)
// =============================================================================

// Mark value as potentially cyclic
void gc_mark_cyclic(XValue v) {
    if (v.type >= TYPE_STRING) {
        v.flags |= FLAG_MARKED;
    }
}

// Check if value is marked as cyclic
bool gc_is_marked(XValue v) {
    return (v.flags & FLAG_MARKED) != 0;
}

// Clear cyclic mark
void gc_unmark(inout XValue v) {
    v.flags &= ~FLAG_MARKED;
}

// =============================================================================
// Weak References
// =============================================================================

// Create a weak reference to a value
XValue xvalue_weak(XValue v) {
    XValue weak = v;
    weak.flags |= FLAG_WEAK;
    // Don't increment refcount for weak refs
    return weak;
}

// Check if value is a weak reference
bool xvalue_is_weak(XValue v) {
    return (v.flags & FLAG_WEAK) != 0;
}

// Strengthen a weak reference (returns nil if target was collected)
XValue xvalue_strengthen(XValue weak) {
    if (!xvalue_is_weak(weak)) {
        return weak;
    }
    
    // Check if target still exists
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

// =============================================================================
// GC Control
// =============================================================================

// Enable GC
void gc_enable() {
    g_gcState[0].enabled = 1;
}

// Disable GC
void gc_disable() {
    g_gcState[0].enabled = 0;
}

// Check if GC is enabled
bool gc_is_enabled() {
    return g_gcState[0].enabled != 0;
}

// Get GC statistics
void gc_get_stats(out uint totalCollected, out uint allocsSinceGC) {
    totalCollected = g_gcState[0].totalCollected;
    allocsSinceGC = g_gcState[0].allocsSinceGC;
}

// Reset GC statistics
void gc_reset_stats() {
    g_gcState[0].allocsSinceGC = 0;
}

// =============================================================================
// GC Initialization
// =============================================================================

void gc_init() {
    g_gcState[0].enabled = 1;
    g_gcState[0].threshold = 1000;
    g_gcState[0].allocsSinceGC = 0;
    g_gcState[0].totalCollected = 0;
}
