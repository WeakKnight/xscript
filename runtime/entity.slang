// XScript Entity Pool (32-bit)
// Entity storage and ID management for ECS-style operations

// =============================================================================
// Constants
// =============================================================================

static const uint ENTITY_MAX_COUNT = 65536;       // Maximum entities
static const uint ENTITY_GENERATION_BITS = 12;    // 4096 generations
static const uint ENTITY_INDEX_BITS = 20;         // ~1M entity indices
static const uint ENTITY_INDEX_MASK = 0x000FFFFF;
static const uint ENTITY_GENERATION_MASK = 0xFFF00000;
static const uint ENTITY_GENERATION_SHIFT = 20;

static const uint INVALID_ENTITY = 0xFFFFFFFF;

// Entity flags
static const uint ENTITY_FLAG_NONE = 0x00;
static const uint ENTITY_FLAG_ACTIVE = 0x01;
static const uint ENTITY_FLAG_DESTROYED = 0x02;
static const uint ENTITY_FLAG_PENDING_SPAWN = 0x04;

// =============================================================================
// Entity Slot Structure
// =============================================================================

struct EntitySlot {
    uint tablePtr;      // Pointer to entity's Table in heap
    uint generation;    // Generation counter for handle validation
    uint flags;         // Entity state flags
    uint reserved;      // Reserved for future use
};

// =============================================================================
// Entity Pool State
// =============================================================================

struct EntityPoolState {
    uint activeCount;       // Number of active entities
    uint highWaterMark;     // Highest used index + 1
    uint freeListHead;      // Head of free slot list
    uint freeListCount;     // Number of free slots
    uint destroyedCount;    // Entities marked for destruction
    uint padding0;
    uint padding1;
    uint padding2;
};

// =============================================================================
// Entity Pool Buffers
// =============================================================================

// Entity slot storage
RWStructuredBuffer<EntitySlot> g_entityPool;

// Entity pool state
RWStructuredBuffer<EntityPoolState> g_entityPoolState;

// Free list (indices of free slots)
RWStructuredBuffer<uint> g_entityFreeList;

// Destroy list (entity IDs marked for destruction)
RWStructuredBuffer<uint> g_entityDestroyList;
RWStructuredBuffer<uint> g_entityDestroyCount;

// =============================================================================
// Entity ID Helper Functions
// =============================================================================

// Create entity ID from index and generation
uint entity_make_id(uint index, uint generation) {
    return (generation << ENTITY_GENERATION_SHIFT) | (index & ENTITY_INDEX_MASK);
}

// Extract index from entity ID
uint entity_get_index(uint entityId) {
    return entityId & ENTITY_INDEX_MASK;
}

// Extract generation from entity ID
uint entity_get_generation(uint entityId) {
    return (entityId & ENTITY_GENERATION_MASK) >> ENTITY_GENERATION_SHIFT;
}

// =============================================================================
// Entity Pool Initialization
// =============================================================================

// Initialize entity pool (called once at startup)
void entity_pool_init() {
    g_entityPoolState[0].activeCount = 0;
    g_entityPoolState[0].highWaterMark = 0;
    g_entityPoolState[0].freeListHead = 0;
    g_entityPoolState[0].freeListCount = 0;
    g_entityPoolState[0].destroyedCount = 0;
    
    g_entityDestroyCount[0] = 0;
}

// =============================================================================
// Entity Creation
// =============================================================================

// Create a new entity with the given table
// Returns entity ID, or INVALID_ENTITY on failure
uint entity_create(uint tablePtr) {
    uint index;
    uint generation;
    
    // Try to get from free list first
    uint freeCount;
    InterlockedAdd(g_entityPoolState[0].freeListCount, -1, freeCount);
    
    if (freeCount > 0) {
        // Pop from free list
        uint freeListIndex = freeCount - 1;
        index = g_entityFreeList[freeListIndex];
        generation = g_entityPool[index].generation;
    } else {
        // Restore free list count (we decremented but didn't use)
        InterlockedAdd(g_entityPoolState[0].freeListCount, 1);
        
        // Allocate new slot from high water mark
        InterlockedAdd(g_entityPoolState[0].highWaterMark, 1, index);
        
        if (index >= ENTITY_MAX_COUNT) {
            // Rollback and fail
            InterlockedAdd(g_entityPoolState[0].highWaterMark, -1);
            return INVALID_ENTITY;
        }
        
        generation = 0;
    }
    
    // Initialize entity slot
    EntitySlot slot;
    slot.tablePtr = tablePtr;
    slot.generation = generation;
    slot.flags = ENTITY_FLAG_ACTIVE;
    slot.reserved = 0;
    g_entityPool[index] = slot;
    
    // Increment active count
    InterlockedAdd(g_entityPoolState[0].activeCount, 1);
    
    return entity_make_id(index, generation);
}

// =============================================================================
// Entity Validation
// =============================================================================

// Check if entity ID is valid
bool entity_is_valid(uint entityId) {
    if (entityId == INVALID_ENTITY) {
        return false;
    }
    
    uint index = entity_get_index(entityId);
    uint generation = entity_get_generation(entityId);
    
    if (index >= g_entityPoolState[0].highWaterMark) {
        return false;
    }
    
    EntitySlot slot = g_entityPool[index];
    
    // Check generation matches and entity is active
    return (slot.generation == generation) && 
           ((slot.flags & ENTITY_FLAG_ACTIVE) != 0) &&
           ((slot.flags & ENTITY_FLAG_DESTROYED) == 0);
}

// =============================================================================
// Entity Access
// =============================================================================

// Get entity table pointer by ID
// Returns 0 if entity is invalid
uint entity_get_table_ptr(uint entityId) {
    if (!entity_is_valid(entityId)) {
        return 0;
    }
    
    uint index = entity_get_index(entityId);
    return g_entityPool[index].tablePtr;
}

// Get entity table as XValue
XValue entity_get_table(uint entityId) {
    uint tablePtr = entity_get_table_ptr(entityId);
    if (tablePtr == 0) {
        return XValue::nil();
    }
    return XValue::table(tablePtr);
}

// Get entity flags
uint entity_get_flags(uint entityId) {
    if (entityId == INVALID_ENTITY) {
        return 0;
    }
    
    uint index = entity_get_index(entityId);
    if (index >= g_entityPoolState[0].highWaterMark) {
        return 0;
    }
    
    return g_entityPool[index].flags;
}

// =============================================================================
// Entity Destruction
// =============================================================================

// Mark entity for destruction (deferred)
// Returns true if entity was valid and marked
bool entity_destroy(uint entityId) {
    if (!entity_is_valid(entityId)) {
        return false;
    }
    
    uint index = entity_get_index(entityId);
    
    // Set destroyed flag (atomic)
    uint oldFlags;
    InterlockedOr(g_entityPool[index].flags, ENTITY_FLAG_DESTROYED, oldFlags);
    
    // If not already destroyed, add to destroy list
    if ((oldFlags & ENTITY_FLAG_DESTROYED) == 0) {
        uint destroyIndex;
        InterlockedAdd(g_entityDestroyCount[0], 1, destroyIndex);
        g_entityDestroyList[destroyIndex] = entityId;
        
        InterlockedAdd(g_entityPoolState[0].destroyedCount, 1);
        return true;
    }
    
    return false;
}

// Process destroy list (called from host after dispatch)
// This actually frees the entity slots
void entity_process_destroy_list() {
    uint destroyCount = g_entityDestroyCount[0];
    
    for (uint i = 0; i < destroyCount; i++) {
        uint entityId = g_entityDestroyList[i];
        uint index = entity_get_index(entityId);
        
        EntitySlot slot = g_entityPool[index];
        
        // Verify still marked for destruction
        if ((slot.flags & ENTITY_FLAG_DESTROYED) != 0) {
            // Clear slot
            slot.tablePtr = 0;
            slot.generation = slot.generation + 1;  // Increment generation
            slot.flags = ENTITY_FLAG_NONE;
            g_entityPool[index] = slot;
            
            // Add to free list
            uint freeIndex;
            InterlockedAdd(g_entityPoolState[0].freeListCount, 1, freeIndex);
            g_entityFreeList[freeIndex] = index;
            
            // Decrement active count
            InterlockedAdd(g_entityPoolState[0].activeCount, -1);
        }
    }
    
    // Reset destroy count and destroyed counter
    g_entityDestroyCount[0] = 0;
    g_entityPoolState[0].destroyedCount = 0;
}

// =============================================================================
// Entity Pool Statistics
// =============================================================================

// Get entity pool statistics
void entity_get_stats(out uint activeCount, out uint highWaterMark, out uint destroyedCount) {
    activeCount = g_entityPoolState[0].activeCount;
    highWaterMark = g_entityPoolState[0].highWaterMark;
    destroyedCount = g_entityPoolState[0].destroyedCount;
}

// Get total active entity count
uint entity_get_active_count() {
    return g_entityPoolState[0].activeCount;
}

// =============================================================================
// Entity Iteration (for dispatch)
// =============================================================================

// Get entity ID by pool index (for iteration)
// Returns INVALID_ENTITY if slot is not active
uint entity_get_by_index(uint index) {
    if (index >= g_entityPoolState[0].highWaterMark) {
        return INVALID_ENTITY;
    }
    
    EntitySlot slot = g_entityPool[index];
    
    if ((slot.flags & ENTITY_FLAG_ACTIVE) == 0 ||
        (slot.flags & ENTITY_FLAG_DESTROYED) != 0) {
        return INVALID_ENTITY;
    }
    
    return entity_make_id(index, slot.generation);
}

// =============================================================================
// Compute Shader Entry Points
// =============================================================================

// Initialize entity pool
[shader("compute")]
[numthreads(1, 1, 1)]
void entity_pool_init_kernel() {
    entity_pool_init();
}

// Process destroy list kernel
[shader("compute")]
[numthreads(1, 1, 1)]
void entity_process_destroy_kernel() {
    entity_process_destroy_list();
}

