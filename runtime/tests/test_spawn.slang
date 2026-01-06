// XScript Spawn Buffer Tests (32-bit)
// Tests for GPU-side spawn buffer operations

// =============================================================================
// Constants
// =============================================================================

static const uint SPAWN_BUFFER_CAPACITY = 65536;
static const uint SPAWN_PENDING_FLAG = 0x80000000;

static const uint SPAWN_STATUS_PENDING = 0;
static const uint SPAWN_STATUS_COMMITTED = 1;
static const uint SPAWN_STATUS_FAILED = 2;

static const uint INVALID_ENTITY = 0xFFFFFFFF;

// =============================================================================
// Structures
// =============================================================================

struct SpawnRequest {
    uint tablePtr;
    uint sourceEntityId;
    uint sourceThreadId;
    uint status;
};

struct SpawnBufferState {
    uint spawnCount;
    uint committedCount;
    uint failedCount;
    uint capacityUsed;
};

// =============================================================================
// Test Buffers
// =============================================================================

RWStructuredBuffer<SpawnRequest> g_spawnBuffer;
RWStructuredBuffer<SpawnBufferState> g_spawnBufferState;
RWStructuredBuffer<uint> g_spawnCount;
RWStructuredBuffer<uint> g_spawnedEntityIds;

RWStructuredBuffer<uint> g_testResults;

// =============================================================================
// Helper Functions
// =============================================================================

void spawn_buffer_init() {
    g_spawnCount[0] = 0;
    g_spawnBufferState[0].spawnCount = 0;
    g_spawnBufferState[0].committedCount = 0;
    g_spawnBufferState[0].failedCount = 0;
}

void spawn_buffer_reset() {
    uint currentCount = g_spawnCount[0];
    if (currentCount > g_spawnBufferState[0].capacityUsed) {
        g_spawnBufferState[0].capacityUsed = currentCount;
    }
    
    g_spawnCount[0] = 0;
    g_spawnBufferState[0].spawnCount = 0;
}

uint spawn_entity_gpu(uint tablePtr, uint sourceEntityId, uint threadId) {
    uint spawnIndex;
    InterlockedAdd(g_spawnCount[0], 1, spawnIndex);
    
    if (spawnIndex >= SPAWN_BUFFER_CAPACITY) {
        InterlockedAdd(g_spawnCount[0], -1);
        InterlockedAdd(g_spawnBufferState[0].failedCount, 1);
        return 0xFFFFFFFF;
    }
    
    SpawnRequest request;
    request.tablePtr = tablePtr;
    request.sourceEntityId = sourceEntityId;
    request.sourceThreadId = threadId;
    request.status = SPAWN_STATUS_PENDING;
    
    g_spawnBuffer[spawnIndex] = request;
    
    return spawnIndex | SPAWN_PENDING_FLAG;
}

uint spawn_entity_gpu_simple(uint tablePtr, uint threadId) {
    return spawn_entity_gpu(tablePtr, INVALID_ENTITY, threadId);
}

uint spawn_get_pending_count() {
    return g_spawnCount[0];
}

bool spawn_is_pending(uint spawnId) {
    return (spawnId & SPAWN_PENDING_FLAG) != 0;
}

uint spawn_get_index(uint spawnId) {
    return spawnId & ~SPAWN_PENDING_FLAG;
}

void spawn_commit_single(uint spawnIndex, uint entityId) {
    if (spawnIndex < g_spawnCount[0]) {
        g_spawnBuffer[spawnIndex].status = SPAWN_STATUS_COMMITTED;
        g_spawnedEntityIds[spawnIndex] = entityId;
        InterlockedAdd(g_spawnBufferState[0].committedCount, 1);
    }
}

void spawn_fail_single(uint spawnIndex) {
    if (spawnIndex < g_spawnCount[0]) {
        g_spawnBuffer[spawnIndex].status = SPAWN_STATUS_FAILED;
        g_spawnedEntityIds[spawnIndex] = INVALID_ENTITY;
        InterlockedAdd(g_spawnBufferState[0].failedCount, 1);
    }
}

// =============================================================================
// Test: Spawn Buffer Initialization
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_spawn_buffer_init() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize spawn buffer
    spawn_buffer_init();
    
    // Test 1: Spawn count is 0
    if (g_spawnCount[0] == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Committed count is 0
    if (g_spawnBufferState[0].committedCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Failed count is 0
    if (g_spawnBufferState[0].failedCount == 0) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Single Spawn
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_spawn_single() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize
    spawn_buffer_init();
    
    // Test 1: Spawn an entity
    uint tablePtr = 12345;
    uint spawnId = spawn_entity_gpu_simple(tablePtr, 0);
    
    if (spawnId != INVALID_ENTITY) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Spawn ID has pending flag
    if (spawn_is_pending(spawnId)) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Spawn index is 0
    if (spawn_get_index(spawnId) == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Pending count is 1
    if (spawn_get_pending_count() == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: Request has correct table pointer
    SpawnRequest request = g_spawnBuffer[0];
    if (request.tablePtr == tablePtr) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 6: Request status is pending
    if (request.status == SPAWN_STATUS_PENDING) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Multiple Spawns
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_spawn_multiple() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize
    spawn_buffer_init();
    
    // Spawn 10 entities
    uint spawnIds[10];
    for (uint i = 0; i < 10; i++) {
        spawnIds[i] = spawn_entity_gpu_simple(100 + i, i);
    }
    
    // Test 1: All spawns successful
    bool allValid = true;
    for (uint i = 0; i < 10; i++) {
        if (spawnIds[i] == INVALID_ENTITY || !spawn_is_pending(spawnIds[i])) {
            allValid = false;
            break;
        }
    }
    if (allValid) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Pending count is 10
    if (spawn_get_pending_count() == 10) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Each spawn has unique index
    bool uniqueIndices = true;
    for (uint i = 0; i < 10; i++) {
        if (spawn_get_index(spawnIds[i]) != i) {
            uniqueIndices = false;
            break;
        }
    }
    if (uniqueIndices) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Each request has correct table pointer
    bool correctPtrs = true;
    for (uint i = 0; i < 10; i++) {
        if (g_spawnBuffer[i].tablePtr != 100 + i) {
            correctPtrs = false;
            break;
        }
    }
    if (correctPtrs) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Spawn with Source Entity
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_spawn_with_source() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize
    spawn_buffer_init();
    
    // Spawn with source entity tracking
    uint sourceEntity = 42;
    uint threadId = 7;
    uint spawnId = spawn_entity_gpu(200, sourceEntity, threadId);
    
    // Test 1: Spawn successful
    if (spawnId != INVALID_ENTITY) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Request has correct source entity
    SpawnRequest request = g_spawnBuffer[0];
    if (request.sourceEntityId == sourceEntity) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Request has correct thread ID
    if (request.sourceThreadId == threadId) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Spawn Commit
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_spawn_commit() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize
    spawn_buffer_init();
    
    // Spawn entities
    uint spawnId1 = spawn_entity_gpu_simple(100, 0);
    uint spawnId2 = spawn_entity_gpu_simple(200, 0);
    
    // Commit first spawn with entity ID 1000
    uint entityId1 = 1000;
    spawn_commit_single(spawn_get_index(spawnId1), entityId1);
    
    // Test 1: First spawn status is committed
    if (g_spawnBuffer[0].status == SPAWN_STATUS_COMMITTED) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: Entity ID stored correctly
    if (g_spawnedEntityIds[0] == entityId1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Committed count is 1
    if (g_spawnBufferState[0].committedCount == 1) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: Second spawn still pending
    if (g_spawnBuffer[1].status == SPAWN_STATUS_PENDING) {
        passed++;
    } else {
        failed++;
    }
    
    // Fail second spawn
    spawn_fail_single(spawn_get_index(spawnId2));
    
    // Test 5: Second spawn status is failed
    if (g_spawnBuffer[1].status == SPAWN_STATUS_FAILED) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 6: Failed count is 1
    if (g_spawnBufferState[0].failedCount == 1) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Spawn Buffer Reset
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void test_spawn_buffer_reset() {
    uint passed = 0;
    uint failed = 0;
    
    // Initialize
    spawn_buffer_init();
    
    // Spawn some entities
    for (uint i = 0; i < 5; i++) {
        spawn_entity_gpu_simple(100 + i, i);
    }
    
    // Test 1: Pending count is 5
    if (spawn_get_pending_count() == 5) {
        passed++;
    } else {
        failed++;
    }
    
    // Reset buffer
    spawn_buffer_reset();
    
    // Test 2: Pending count is 0 after reset
    if (spawn_get_pending_count() == 0) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: Capacity used is recorded
    if (g_spawnBufferState[0].capacityUsed >= 5) {
        passed++;
    } else {
        failed++;
    }
    
    // Spawn new entities after reset
    uint newSpawnId = spawn_entity_gpu_simple(500, 0);
    
    // Test 4: New spawn gets index 0
    if (spawn_get_index(newSpawnId) == 0) {
        passed++;
    } else {
        failed++;
    }
    
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

// =============================================================================
// Test: Parallel Spawns (simulated)
// =============================================================================

[shader("compute")]
[numthreads(64, 1, 1)]
void test_spawn_parallel(uint3 threadId : SV_DispatchThreadID) {
    // Each thread spawns one entity
    uint tid = threadId.x;
    
    // Use thread ID as table pointer (for verification)
    uint tablePtr = 1000 + tid;
    uint spawnId = spawn_entity_gpu_simple(tablePtr, tid);
    
    // Store result for verification
    g_testResults[tid] = spawnId;
}

// Verify parallel spawn results
[shader("compute")]
[numthreads(1, 1, 1)]
void test_spawn_parallel_verify() {
    uint passed = 0;
    uint failed = 0;
    
    // Test 1: 64 spawns occurred
    if (spawn_get_pending_count() == 64) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 2: All spawns have pending flag
    bool allPending = true;
    for (uint i = 0; i < 64; i++) {
        uint spawnId = g_testResults[i];
        if (!spawn_is_pending(spawnId)) {
            allPending = false;
            break;
        }
    }
    if (allPending) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: All indices are unique (0-63)
    uint indexUsed[64];
    for (uint i = 0; i < 64; i++) {
        indexUsed[i] = 0;
    }
    
    bool uniqueIndices = true;
    for (uint i = 0; i < 64; i++) {
        uint spawnId = g_testResults[i];
        uint idx = spawn_get_index(spawnId);
        if (idx < 64) {
            if (indexUsed[idx] != 0) {
                uniqueIndices = false;
                break;
            }
            indexUsed[idx] = 1;
        } else {
            uniqueIndices = false;
            break;
        }
    }
    if (uniqueIndices) {
        passed++;
    } else {
        failed++;
    }
    
    // Store final results
    g_testResults[0] = passed;
    g_testResults[1] = failed;
}

