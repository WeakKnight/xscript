// XScript Spawn Buffer (32-bit)
// GPU-side atomic spawn buffer for entity creation during dispatch

// =============================================================================
// Constants
// =============================================================================

static const uint SPAWN_BUFFER_CAPACITY = 65536;  // Max spawns per dispatch
static const uint SPAWN_PENDING_FLAG = 0x80000000;  // Flag for pending spawn ID

// Spawn request status
static const uint SPAWN_STATUS_PENDING = 0;
static const uint SPAWN_STATUS_COMMITTED = 1;
static const uint SPAWN_STATUS_FAILED = 2;

// =============================================================================
// Spawn Request Structure
// =============================================================================

struct SpawnRequest {
    uint tablePtr;          // Pointer to spawned table in heap
    uint sourceEntityId;    // Entity that spawned this (for tracking)
    uint sourceThreadId;    // GPU thread that created this
    uint status;            // Request status
};

// =============================================================================
// Spawn Buffer State
// =============================================================================

struct SpawnBufferState {
    uint spawnCount;        // Number of pending spawns
    uint committedCount;    // Number of committed spawns
    uint failedCount;       // Number of failed spawns
    uint capacityUsed;      // Peak capacity usage
};

// =============================================================================
// Spawn Buffer Buffers
// =============================================================================

// Spawn request buffer
RWStructuredBuffer<SpawnRequest> g_spawnBuffer;

// Spawn buffer state
RWStructuredBuffer<SpawnBufferState> g_spawnBufferState;

// Atomic spawn counter (separate for fast atomic access)
RWStructuredBuffer<uint> g_spawnCount;

// Committed entity IDs (filled after commit)
RWStructuredBuffer<uint> g_spawnedEntityIds;

// =============================================================================
// Spawn Buffer Initialization
// =============================================================================

// Initialize spawn buffer (called at start of each dispatch)
void spawn_buffer_init() {
    g_spawnCount[0] = 0;
    g_spawnBufferState[0].spawnCount = 0;
    g_spawnBufferState[0].committedCount = 0;
    g_spawnBufferState[0].failedCount = 0;
}

// Reset spawn buffer (called after commit)
void spawn_buffer_reset() {
    // Update capacity used for statistics
    uint currentCount = g_spawnCount[0];
    if (currentCount > g_spawnBufferState[0].capacityUsed) {
        g_spawnBufferState[0].capacityUsed = currentCount;
    }
    
    g_spawnCount[0] = 0;
    g_spawnBufferState[0].spawnCount = 0;
}

// =============================================================================
// GPU-Side Spawn Functions
// =============================================================================

// Spawn an entity from GPU thread
// tablePtr: pointer to the table data in heap
// sourceEntityId: the entity doing the spawning (or INVALID_ENTITY)
// Returns: temporary spawn index with SPAWN_PENDING_FLAG, or INVALID (0xFFFFFFFF)
uint spawn_entity_gpu(uint tablePtr, uint sourceEntityId, uint threadId) {
    // Atomically allocate spawn slot
    uint spawnIndex;
    InterlockedAdd(g_spawnCount[0], 1, spawnIndex);
    
    // Check capacity
    if (spawnIndex >= SPAWN_BUFFER_CAPACITY) {
        // Buffer overflow - rollback
        InterlockedAdd(g_spawnCount[0], -1);
        InterlockedAdd(g_spawnBufferState[0].failedCount, 1);
        return 0xFFFFFFFF;
    }
    
    // Create spawn request
    SpawnRequest request;
    request.tablePtr = tablePtr;
    request.sourceEntityId = sourceEntityId;
    request.sourceThreadId = threadId;
    request.status = SPAWN_STATUS_PENDING;
    
    g_spawnBuffer[spawnIndex] = request;
    
    // Return pending spawn ID (marked with flag so host knows it's not a real entity yet)
    return spawnIndex | SPAWN_PENDING_FLAG;
}

// Simplified spawn without source tracking
uint spawn_entity_gpu_simple(uint tablePtr, uint threadId) {
    return spawn_entity_gpu(tablePtr, 0xFFFFFFFF, threadId);
}

// =============================================================================
// Spawn Buffer Query Functions
// =============================================================================

// Get number of pending spawns
uint spawn_get_pending_count() {
    return g_spawnCount[0];
}

// Get spawn request by index
SpawnRequest spawn_get_request(uint index) {
    return g_spawnBuffer[index];
}

// Check if spawn ID is a pending spawn (not yet committed)
bool spawn_is_pending(uint spawnId) {
    return (spawnId & SPAWN_PENDING_FLAG) != 0;
}

// Get spawn index from pending spawn ID
uint spawn_get_index(uint spawnId) {
    return spawnId & ~SPAWN_PENDING_FLAG;
}

// =============================================================================
// Spawn Commit Functions (called from host-side processing)
// =============================================================================

// Commit a single spawn request to entity pool
// This is called for each pending spawn during the commit phase
// entityId: the actual entity ID assigned by entity_create()
void spawn_commit_single(uint spawnIndex, uint entityId) {
    if (spawnIndex < g_spawnCount[0]) {
        g_spawnBuffer[spawnIndex].status = SPAWN_STATUS_COMMITTED;
        g_spawnedEntityIds[spawnIndex] = entityId;
        InterlockedAdd(g_spawnBufferState[0].committedCount, 1);
    }
}

// Mark a spawn as failed
void spawn_fail_single(uint spawnIndex) {
    if (spawnIndex < g_spawnCount[0]) {
        g_spawnBuffer[spawnIndex].status = SPAWN_STATUS_FAILED;
        g_spawnedEntityIds[spawnIndex] = 0xFFFFFFFF;
        InterlockedAdd(g_spawnBufferState[0].failedCount, 1);
    }
}

// Get committed entity ID for a spawn
uint spawn_get_committed_entity(uint spawnIndex) {
    if (spawnIndex < g_spawnCount[0] && 
        g_spawnBuffer[spawnIndex].status == SPAWN_STATUS_COMMITTED) {
        return g_spawnedEntityIds[spawnIndex];
    }
    return 0xFFFFFFFF;
}

// =============================================================================
// Spawn Buffer Statistics
// =============================================================================

// Get spawn buffer statistics
void spawn_get_stats(out uint pendingCount, out uint committedCount, out uint failedCount) {
    pendingCount = g_spawnBufferState[0].spawnCount;
    committedCount = g_spawnBufferState[0].committedCount;
    failedCount = g_spawnBufferState[0].failedCount;
}

// Get peak capacity usage
uint spawn_get_peak_usage() {
    return g_spawnBufferState[0].capacityUsed;
}

// =============================================================================
// Compute Shader Entry Points
// =============================================================================

// Initialize spawn buffer kernel
[shader("compute")]
[numthreads(1, 1, 1)]
void spawn_buffer_init_kernel() {
    spawn_buffer_init();
}

// Reset spawn buffer kernel
[shader("compute")]
[numthreads(1, 1, 1)]
void spawn_buffer_reset_kernel() {
    spawn_buffer_reset();
}

// Commit spawns kernel (processes all pending spawns)
// This would be called after entity_create() is done for each spawn
[shader("compute")]
[numthreads(64, 1, 1)]
void spawn_commit_kernel(
    uint3 threadId : SV_DispatchThreadID,
    uniform uint spawnCount
) {
    uint idx = threadId.x;
    if (idx >= spawnCount) return;
    
    SpawnRequest request = g_spawnBuffer[idx];
    
    if (request.status == SPAWN_STATUS_PENDING) {
        // In a real implementation, entity_create would be called here
        // For now, we just mark it as needing host processing
        g_spawnBufferState[0].spawnCount = spawnCount;
    }
}

