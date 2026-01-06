// XScript Dispatch Tests (32-bit)
// Tests for ECS system dispatch kernel

// =============================================================================
// Constants
// =============================================================================

static const uint TYPE_NIL      = 0;
static const uint TYPE_BOOL     = 1;
static const uint TYPE_NUMBER   = 2;
static const uint TYPE_STRING   = 3;
static const uint TYPE_TABLE    = 4;
static const uint TYPE_FUNCTION = 5;

static const uint DISPATCH_MAX_REQUIRED_KEYS = 8;
static const uint DISPATCH_THREAD_GROUP_SIZE = 64;

static const uint DISPATCH_STATUS_IDLE = 0;
static const uint DISPATCH_STATUS_RUNNING = 1;
static const uint DISPATCH_STATUS_COMPLETED = 2;

static const uint ENTITY_FLAG_ACTIVE = 0x01;
static const uint INVALID_ENTITY = 0xFFFFFFFF;

// =============================================================================
// Structures
// =============================================================================

struct DispatchConfig {
    uint functionIndex;
    uint entityCount;
    uint requiredKeyCount;
    uint requiredKeys[DISPATCH_MAX_REQUIRED_KEYS];
    float dt;
    uint flags;
    uint padding0;
    uint padding1;
};

struct DispatchState {
    uint processedCount;
    uint skippedCount;
    uint errorCount;
    uint spawnCount;
    uint destroyCount;
    uint status;
    uint padding0;
    uint padding1;
};

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

    uint asUint() {
        return data;
    }
};

// =============================================================================
// Test Buffers
// =============================================================================

RWStructuredBuffer<DispatchConfig> g_dispatchConfig;
RWStructuredBuffer<DispatchState> g_dispatchState;
RWStructuredBuffer<uint> g_dispatchEntityList;
RWStructuredBuffer<uint> g_dispatchResults;
RWStructuredBuffer<uint> g_filteredEntityCount;

RWStructuredBuffer<EntitySlot> g_entityPool;
RWStructuredBuffer<EntityPoolState> g_entityPoolState;

RWStructuredBuffer<uint> g_testResults;

// Mock current entity state
static uint g_currentEntityId = INVALID_ENTITY;
static uint g_currentEntityTablePtr = 0;

// =============================================================================
// Helper Functions
// =============================================================================

void dispatch_init() {
    g_dispatchState[0].processedCount = 0;
    g_dispatchState[0].skippedCount = 0;
    g_dispatchState[0].errorCount = 0;
    g_dispatchState[0].spawnCount = 0;
    g_dispatchState[0].destroyCount = 0;
    g_dispatchState[0].status = DISPATCH_STATUS_IDLE;
}

uint entity_get_index(uint entityId) {
    return entityId & 0x000FFFFF;
}

uint entity_get_generation(uint entityId) {
    return (entityId & 0xFFF00000) >> 20;
}

uint entity_make_id(uint index, uint generation) {
    return (generation << 20) | (index & 0x000FFFFF);
}

uint entity_get_table_ptr(uint entityId) {
    if (entityId == INVALID_ENTITY) {
        return 0;
    }
    
    uint index = entity_get_index(entityId);
    if (index >= g_entityPoolState[0].highWaterMark) {
        return 0;
    }
    
    EntitySlot slot = g_entityPool[index];
    uint gen = entity_get_generation(entityId);
    
    if (slot.generation != gen || (slot.flags & ENTITY_FLAG_ACTIVE) == 0) {
        return 0;
    }
    
    return slot.tablePtr;
}

uint entity_get_by_index(uint index) {
    if (index >= g_entityPoolState[0].highWaterMark) {
        return INVALID_ENTITY;
    }
    
    EntitySlot slot = g_entityPool[index];
    
    if ((slot.flags & ENTITY_FLAG_ACTIVE) == 0) {
        return INVALID_ENTITY;
    }
    
    return entity_make_id(index, slot.generation);
}

void vm_set_current_entity(uint entityId, uint tablePtr) {
    g_currentEntityId = entityId;
    g_currentEntityTablePtr = tablePtr;
}

void vm_clear_current_entity() {
    g_currentEntityId = INVALID_ENTITY;
    g_currentEntityTablePtr = 0;
}

XValue vm_get_current_entity() {
    if (g_currentEntityTablePtr == 0) {
        return XValue::nil();
    }
    return XValue::table(g_currentEntityTablePtr);
}

// Simplified table_get for testing (returns nil for all)
XValue table_get(XValue t, XValue key) {
    return XValue::nil();
}

bool dispatch_entity_matches(XValue entityTable, uint requiredKeys[DISPATCH_MAX_REQUIRED_KEYS], uint requiredKeyCount) {
    if (entityTable.type != TYPE_TABLE) {
        return false;
    }
    
    for (uint i = 0; i < requiredKeyCount; i++) {
        XValue key = XValue::string(requiredKeys[i]);
        XValue value = table_get(entityTable, key);
        
        if (value.isNil()) {
            return false;
        }
    }
    
    return true;
}

// =============================================================================
// Test: Dispatch Initialization
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_dispatch_init() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize dispatch
    dispatch_init();
    
    // Test 1: Processed count is 0
    if (g_dispatchState[0].processedCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Skipped count is 0
    if (g_dispatchState[0].skippedCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Error count is 0
    if (g_dispatchState[0].errorCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Status is idle
    if (g_dispatchState[0].status == DISPATCH_STATUS_IDLE) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Dispatch Configuration
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_dispatch_config() {
    uint passed = 0;
    uint failed = 0;
    
    // Set up dispatch configuration - initialize ALL fields to avoid undefined values
    DispatchConfig config;
    config.functionIndex = 42;
    config.entityCount = 1000;
    config.requiredKeyCount = 2;
    // Initialize all array elements to avoid undefined UAV writes
    for (uint i = 0; i < DISPATCH_MAX_REQUIRED_KEYS; i++) {
        config.requiredKeys[i] = 0;
    }
    config.requiredKeys[0] = 100;  // "position" string index
    config.requiredKeys[1] = 101;  // "velocity" string index
    config.dt = 0.016f;
    config.flags = 0;
    config.padding0 = 0;
    config.padding1 = 0;
    
    g_dispatchConfig[0] = config;
    
    // Read back and verify
    DispatchConfig readConfig = g_dispatchConfig[0];
    
    // Test 1: Function index
    if (readConfig.functionIndex == 42) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Entity count
    if (readConfig.entityCount == 1000) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Required key count
    if (readConfig.requiredKeyCount == 2) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: First required key
    if (readConfig.requiredKeys[0] == 100) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: Second required key
    if (readConfig.requiredKeys[1] == 101) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 6: Delta time
    if (readConfig.dt > 0.015f && readConfig.dt < 0.017f) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Entity Matching (No Filter)
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_dispatch_no_filter() {
    uint passed = 0;
    uint failed = 0;
    
    // Create mock entity table
    XValue entityTable = XValue::table(12345);
    
    // Empty required keys (no filter)
    uint requiredKeys[DISPATCH_MAX_REQUIRED_KEYS];
    for (uint i = 0; i < DISPATCH_MAX_REQUIRED_KEYS; i++) {
        requiredKeys[i] = 0;
    }
    
    // Test 1: With 0 required keys, should always match
    bool matches = dispatch_entity_matches(entityTable, requiredKeys, 0);
    if (matches) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Nil value should not match
    XValue nilValue = XValue::nil();
    matches = dispatch_entity_matches(nilValue, requiredKeys, 0);
    if (!matches) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Entity List Building
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_dispatch_entity_list() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize
    g_entityPoolState[0].activeCount = 0;
    g_entityPoolState[0].highWaterMark = 5;
    g_filteredEntityCount[0] = 0;
    
    // Create 5 mock entities (indices 0-4)
    for (uint i = 0; i < 5; i++) {
        EntitySlot slot;
        slot.tablePtr = 1000 + i;
        slot.generation = 0;
        slot.flags = ENTITY_FLAG_ACTIVE;
        slot.reserved = 0;
        g_entityPool[i] = slot;
    }
    
    // Test 1: entity_get_by_index returns valid entity
    uint entityId = entity_get_by_index(0);
    if (entityId != INVALID_ENTITY) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: entity_get_by_index with out of range returns INVALID
    entityId = entity_get_by_index(100);
    if (entityId == INVALID_ENTITY) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: entity_get_table_ptr returns correct pointer
    entityId = entity_get_by_index(2);
    uint tablePtr = entity_get_table_ptr(entityId);
    if (tablePtr == 1002) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Inactive entity returns INVALID
    g_entityPool[3].flags = 0;  // Deactivate
    entityId = entity_get_by_index(3);
    if (entityId == INVALID_ENTITY) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Current Entity Management
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_dispatch_current_entity() {
    uint passed = 0;
    uint failed = 0;
    
    // Test 1: Initial state is cleared
    vm_clear_current_entity();
    XValue entity = vm_get_current_entity();
    if (entity.isNil()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Set current entity
    vm_set_current_entity(42, 12345);
    entity = vm_get_current_entity();
    if (!entity.isNil() && entity.type == TYPE_TABLE && entity.asUint() == 12345) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Clear current entity
    vm_clear_current_entity();
    entity = vm_get_current_entity();
    if (entity.isNil()) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Dispatch Status Transitions
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_dispatch_status() {
    uint passed = 0;
    uint failed = 0;
    
    // Test 1: Initial status is idle after init
    dispatch_init();
    if (g_dispatchState[0].status == DISPATCH_STATUS_IDLE) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Can set status to running
    g_dispatchState[0].status = DISPATCH_STATUS_RUNNING;
    if (g_dispatchState[0].status == DISPATCH_STATUS_RUNNING) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Can set status to completed
    g_dispatchState[0].status = DISPATCH_STATUS_COMPLETED;
    if (g_dispatchState[0].status == DISPATCH_STATUS_COMPLETED) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Parallel Dispatch Simulation
// =============================================================================

[shader("compute")]
[numthreads(DISPATCH_THREAD_GROUP_SIZE, 1, 1)]
void test_dispatch_parallel(uint3 threadId : SV_DispatchThreadID) {
    uint entityIndex = threadId.x;
    
    // Simulate processing up to 64 entities
    if (entityIndex >= 64) {
        return;
    }
    
    // Mock: each thread "processes" one entity
    InterlockedAdd(g_dispatchState[0].processedCount, 1);
    g_dispatchResults[entityIndex] = 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_dispatch_parallel_verify() {
    uint passed = 0;
    uint failed = 0;
    
    // Test 1: All 64 entities were processed
    if (g_dispatchState[0].processedCount == 64) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: All results marked as processed
    bool allProcessed = true;
    for (uint i = 0; i < 64; i++) {
        if (g_dispatchResults[i] != 1) {
            allProcessed = false;
            break;
        }
    }
    if (allProcessed) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

