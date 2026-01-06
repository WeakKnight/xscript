// XScript Entity Pool Tests (32-bit)
// Tests for entity creation, validation, destruction, and pool management

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
// Entity Constants
// =============================================================================

static const uint ENTITY_MAX_COUNT = 65536;
static const uint ENTITY_INDEX_MASK = 0x000FFFFF;
static const uint ENTITY_GENERATION_MASK = 0xFFF00000;
static const uint ENTITY_GENERATION_SHIFT = 20;
static const uint INVALID_ENTITY = 0xFFFFFFFF;

static const uint ENTITY_FLAG_NONE = 0x00;
static const uint ENTITY_FLAG_ACTIVE = 0x01;
static const uint ENTITY_FLAG_DESTROYED = 0x02;

// =============================================================================
// Heap Constants
// =============================================================================

static const uint HEAP_BLOCK_HEADER_SIZE = 4;
static const uint HEAP_ALIGNMENT = 4;

// =============================================================================
// Structures
// =============================================================================

struct EntitySlot {
    uint tablePtr;
    uint generation;
    uint flags;
    uint reserved;
};

struct EntityPoolState {
    uint activeCount;
    uint highWaterMark;
    uint freeListHead;
    uint freeListCount;
    uint destroyedCount;
    uint padding0;
    uint padding1;
    uint padding2;
};

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

    uint asUint() {
        return data;
    }
};

// =============================================================================
// Test Buffers
// =============================================================================

RWStructuredBuffer<EntitySlot> g_entityPool;
RWStructuredBuffer<EntityPoolState> g_entityPoolState;
RWStructuredBuffer<uint> g_entityFreeList;
RWStructuredBuffer<uint> g_entityDestroyList;
RWStructuredBuffer<uint> g_entityDestroyCount;

RWStructuredBuffer<uint> g_heapMemory;
RWStructuredBuffer<HeapAllocator> g_heapState;

RWStructuredBuffer<uint> g_testResults;

// =============================================================================
// Helper Functions
// =============================================================================

uint entity_make_id(uint index, uint generation) {
    return (generation << ENTITY_GENERATION_SHIFT) | (index & ENTITY_INDEX_MASK);
}

uint entity_get_index(uint entityId) {
    return entityId & ENTITY_INDEX_MASK;
}

uint entity_get_generation(uint entityId) {
    return (entityId & ENTITY_GENERATION_MASK) >> ENTITY_GENERATION_SHIFT;
}

void entity_pool_init() {
    g_entityPoolState[0].activeCount = 0;
    g_entityPoolState[0].highWaterMark = 0;
    g_entityPoolState[0].freeListHead = 0;
    g_entityPoolState[0].freeListCount = 0;
    g_entityPoolState[0].destroyedCount = 0;
    g_entityDestroyCount[0] = 0;
}

uint entity_create(uint tablePtr) {
    uint index;
    uint generation;
    
    uint freeCount;
    InterlockedAdd(g_entityPoolState[0].freeListCount, -1, freeCount);
    
    if (freeCount > 0) {
        uint freeListIndex = freeCount - 1;
        index = g_entityFreeList[freeListIndex];
        generation = g_entityPool[index].generation;
    } else {
        InterlockedAdd(g_entityPoolState[0].freeListCount, 1);
        InterlockedAdd(g_entityPoolState[0].highWaterMark, 1, index);
        
        if (index >= ENTITY_MAX_COUNT) {
            InterlockedAdd(g_entityPoolState[0].highWaterMark, -1);
            return INVALID_ENTITY;
        }
        
        generation = 0;
    }
    
    EntitySlot slot;
    slot.tablePtr = tablePtr;
    slot.generation = generation;
    slot.flags = ENTITY_FLAG_ACTIVE;
    slot.reserved = 0;
    g_entityPool[index] = slot;
    
    InterlockedAdd(g_entityPoolState[0].activeCount, 1);
    
    return entity_make_id(index, generation);
}

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
    
    return (slot.generation == generation) && 
           ((slot.flags & ENTITY_FLAG_ACTIVE) != 0) &&
           ((slot.flags & ENTITY_FLAG_DESTROYED) == 0);
}

uint entity_get_table_ptr(uint entityId) {
    if (!entity_is_valid(entityId)) {
        return 0;
    }
    
    uint index = entity_get_index(entityId);
    return g_entityPool[index].tablePtr;
}

bool entity_destroy(uint entityId) {
    if (!entity_is_valid(entityId)) {
        return false;
    }
    
    uint index = entity_get_index(entityId);
    
    uint oldFlags;
    InterlockedOr(g_entityPool[index].flags, ENTITY_FLAG_DESTROYED, oldFlags);
    
    if ((oldFlags & ENTITY_FLAG_DESTROYED) == 0) {
        uint destroyIndex;
        InterlockedAdd(g_entityDestroyCount[0], 1, destroyIndex);
        g_entityDestroyList[destroyIndex] = entityId;
        
        InterlockedAdd(g_entityPoolState[0].destroyedCount, 1);
        return true;
    }
    
    return false;
}

void entity_process_destroy_list() {
    uint destroyCount = g_entityDestroyCount[0];
    
    for (uint i = 0; i < destroyCount; i++) {
        uint entityId = g_entityDestroyList[i];
        uint index = entity_get_index(entityId);
        
        EntitySlot slot = g_entityPool[index];
        
        if ((slot.flags & ENTITY_FLAG_DESTROYED) != 0) {
            slot.tablePtr = 0;
            slot.generation = slot.generation + 1;
            slot.flags = ENTITY_FLAG_NONE;
            g_entityPool[index] = slot;
            
            uint freeIndex;
            InterlockedAdd(g_entityPoolState[0].freeListCount, 1, freeIndex);
            g_entityFreeList[freeIndex] = index;
            
            InterlockedAdd(g_entityPoolState[0].activeCount, -1);
        }
    }
    
    g_entityDestroyCount[0] = 0;
    g_entityPoolState[0].destroyedCount = 0;
}

// =============================================================================
// Test: Entity ID Encoding/Decoding
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_entity_id_encoding() {
    uint passed = 0;
    uint failed = 0;
    
    // Test 1: Create ID with index 0, generation 0
    {
        uint id = entity_make_id(0, 0);
        uint index = entity_get_index(id);
        uint gen = entity_get_generation(id);
        
        if (index == 0 && gen == 0) {
            passed++;
        } else {
            failed++;
        }
    }
    
    // Test 2: Create ID with index 100, generation 5
    {
        uint id = entity_make_id(100, 5);
        uint index = entity_get_index(id);
        uint gen = entity_get_generation(id);
        
        if (index == 100 && gen == 5) {
            passed++;
        } else {
            failed++;
        }
    }
    
    // Test 3: Max index value
    {
        uint maxIndex = ENTITY_INDEX_MASK;
        uint id = entity_make_id(maxIndex, 0);
        uint index = entity_get_index(id);
        
        if (index == maxIndex) {
            passed++;
        } else {
            failed++;
        }
    }
    
    // Test 4: Max generation value
    {
        uint maxGen = 0xFFF;  // 12 bits
        uint id = entity_make_id(0, maxGen);
        uint gen = entity_get_generation(id);
        
        if (gen == maxGen) {
            passed++;
        } else {
            failed++;
        }
    }
    
    // Test 5: Combined max values
    {
        uint maxIndex = ENTITY_INDEX_MASK;
        uint maxGen = 0xFFF;
        uint id = entity_make_id(maxIndex, maxGen);
        uint index = entity_get_index(id);
        uint gen = entity_get_generation(id);
        
        if (index == maxIndex && gen == maxGen) {
            passed++;
        } else {
            failed++;
        }
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Entity Pool Initialization
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_entity_pool_init() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize pool
    entity_pool_init();
    
    // Test 1: Active count is 0
    if (g_entityPoolState[0].activeCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: High water mark is 0
    if (g_entityPoolState[0].highWaterMark == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Free list is empty
    if (g_entityPoolState[0].freeListCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Destroy count is 0
    if (g_entityDestroyCount[0] == 0) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Entity Creation
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_entity_create() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize pool
    entity_pool_init();
    
    // Test 1: Create first entity
    uint tablePtr1 = 100;
    uint entityId1 = entity_create(tablePtr1);
    
    if (entityId1 != INVALID_ENTITY) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: First entity has index 0
    if (entity_get_index(entityId1) == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Active count is 1
    if (g_entityPoolState[0].activeCount == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: High water mark is 1
    if (g_entityPoolState[0].highWaterMark == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: Create second entity
    uint tablePtr2 = 200;
    uint entityId2 = entity_create(tablePtr2);
    
    if (entityId2 != INVALID_ENTITY && entity_get_index(entityId2) == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 6: Active count is 2
    if (g_entityPoolState[0].activeCount == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 7: Entities are valid
    if (entity_is_valid(entityId1) && entity_is_valid(entityId2)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 8: Table pointers are correct
    if (entity_get_table_ptr(entityId1) == tablePtr1 &&
        entity_get_table_ptr(entityId2) == tablePtr2) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Entity Validation
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_entity_validation() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize pool
    entity_pool_init();
    
    // Create an entity
    uint entityId = entity_create(100);
    
    // Test 1: Valid entity is valid
    if (entity_is_valid(entityId)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: INVALID_ENTITY is not valid
    if (!entity_is_valid(INVALID_ENTITY)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Non-existent index is not valid
    uint fakeId = entity_make_id(999, 0);
    if (!entity_is_valid(fakeId)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Wrong generation is not valid
    uint wrongGen = entity_make_id(entity_get_index(entityId), 999);
    if (!entity_is_valid(wrongGen)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: Get table ptr of invalid entity returns 0
    if (entity_get_table_ptr(INVALID_ENTITY) == 0) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Entity Destruction
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_entity_destroy() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize pool
    entity_pool_init();
    
    // Create entities
    uint entityId1 = entity_create(100);
    uint entityId2 = entity_create(200);
    
    // Test 1: Mark entity for destruction
    bool destroyed = entity_destroy(entityId1);
    if (destroyed) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Destroyed entity is no longer valid
    if (!entity_is_valid(entityId1)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Other entity still valid
    if (entity_is_valid(entityId2)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Destroy count is 1
    if (g_entityDestroyCount[0] == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: Can't destroy same entity twice
    bool destroyedAgain = entity_destroy(entityId1);
    if (!destroyedAgain) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 6: Can't destroy invalid entity
    bool destroyedInvalid = entity_destroy(INVALID_ENTITY);
    if (!destroyedInvalid) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Entity Reuse (Free List)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_entity_reuse() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize pool
    entity_pool_init();
    
    // Create entity
    uint entityId1 = entity_create(100);
    uint index1 = entity_get_index(entityId1);
    uint gen1 = entity_get_generation(entityId1);
    
    // Destroy it
    entity_destroy(entityId1);
    
    // Process destroy list (simulating host processing)
    entity_process_destroy_list();
    
    // Test 1: Free list now has 1 entry
    if (g_entityPoolState[0].freeListCount == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Active count is 0
    if (g_entityPoolState[0].activeCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Create new entity (should reuse slot)
    uint entityId2 = entity_create(200);
    uint index2 = entity_get_index(entityId2);
    uint gen2 = entity_get_generation(entityId2);
    
    // Test 3: Same index reused
    if (index1 == index2) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Generation incremented
    if (gen2 == gen1 + 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: Old entity ID no longer valid
    if (!entity_is_valid(entityId1)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 6: New entity ID is valid
    if (entity_is_valid(entityId2)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 7: Free list is empty again
    if (g_entityPoolState[0].freeListCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Multiple Entity Operations
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_entity_multiple_ops() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize pool
    entity_pool_init();
    
    // Create 10 entities
    uint entities[10];
    for (uint i = 0; i < 10; i++) {
        entities[i] = entity_create(100 + i);
    }
    
    // Test 1: All entities created
    bool allValid = true;
    for (uint i = 0; i < 10; i++) {
        if (!entity_is_valid(entities[i])) {
            allValid = false;
            break;
        }
    }
    if (allValid) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Active count is 10
    if (g_entityPoolState[0].activeCount == 10) {
        passed++;
    } else {
        failed++;
    }
    
    // Destroy every other entity
    for (uint i = 0; i < 10; i += 2) {
        entity_destroy(entities[i]);
    }
    
    // Test 3: 5 entities destroyed
    if (g_entityDestroyCount[0] == 5) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Odd-indexed entities still valid
    bool oddValid = true;
    for (uint i = 1; i < 10; i += 2) {
        if (!entity_is_valid(entities[i])) {
            oddValid = false;
            break;
        }
    }
    if (oddValid) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: Even-indexed entities not valid
    bool evenInvalid = true;
    for (uint i = 0; i < 10; i += 2) {
        if (entity_is_valid(entities[i])) {
            evenInvalid = false;
            break;
        }
    }
    if (evenInvalid) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

