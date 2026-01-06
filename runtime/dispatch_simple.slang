// XScript Simple Dispatch (32-bit)
// Minimal dispatch kernel for testing without full VM

import value;
import entity;

// =============================================================================
// Constants
// =============================================================================

static const uint DISPATCH_MAX_REQUIRED_KEYS = 8;
static const uint DISPATCH_THREAD_GROUP_SIZE = 64;

static const uint DISPATCH_STATUS_IDLE = 0;
static const uint DISPATCH_STATUS_RUNNING = 1;
static const uint DISPATCH_STATUS_COMPLETED = 2;
static const uint DISPATCH_STATUS_ERROR = 3;

// =============================================================================
// Dispatch Structures
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

// =============================================================================
// Buffers
// =============================================================================

RWStructuredBuffer<DispatchConfig> g_dispatchConfig;
RWStructuredBuffer<DispatchState> g_dispatchState;
RWStructuredBuffer<uint> g_dispatchEntityList;
RWStructuredBuffer<uint> g_dispatchResults;

// =============================================================================
// Init Kernel
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void simple_dispatch_init() {
    g_dispatchState[0].processedCount = 0;
    g_dispatchState[0].skippedCount = 0;
    g_dispatchState[0].errorCount = 0;
    g_dispatchState[0].spawnCount = 0;
    g_dispatchState[0].destroyCount = 0;
    g_dispatchState[0].status = DISPATCH_STATUS_RUNNING;
}

// =============================================================================
// Main Simple Dispatch Kernel
// =============================================================================

[shader("compute")]
[numthreads(DISPATCH_THREAD_GROUP_SIZE, 1, 1)]
void simple_dispatch_main(uint3 threadId : SV_DispatchThreadID) {
    uint entityIndex = threadId.x;
    DispatchConfig config = g_dispatchConfig[0];
    
    // Bounds check
    if (entityIndex >= config.entityCount) {
        return;
    }
    
    // Get entity from dispatch list
    uint entityId = g_dispatchEntityList[entityIndex];
    
    // Get entity table pointer
    uint tablePtr = entity_get_table_ptr(entityId);
    if (tablePtr == 0) {
        InterlockedAdd(g_dispatchState[0].skippedCount, 1);
        g_dispatchResults[entityIndex] = 0;
        return;
    }
    
    // Mark as processed (simple test - no VM execution)
    InterlockedAdd(g_dispatchState[0].processedCount, 1);
    g_dispatchResults[entityIndex] = 1;
}

// =============================================================================
// Finalize Kernel
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void simple_dispatch_finalize() {
    g_dispatchState[0].status = DISPATCH_STATUS_COMPLETED;
}

