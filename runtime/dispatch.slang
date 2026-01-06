// XScript ECS Dispatch (32-bit)
// System dispatch kernel for parallel entity processing

// =============================================================================
// Constants
// =============================================================================

static const uint DISPATCH_MAX_REQUIRED_KEYS = 8;   // Max components to filter
static const uint DISPATCH_THREAD_GROUP_SIZE = 64;  // Threads per group

// Dispatch status
static const uint DISPATCH_STATUS_IDLE = 0;
static const uint DISPATCH_STATUS_RUNNING = 1;
static const uint DISPATCH_STATUS_COMPLETED = 2;
static const uint DISPATCH_STATUS_ERROR = 3;

// =============================================================================
// Dispatch Configuration
// =============================================================================

struct DispatchConfig {
    uint functionIndex;         // Function to execute (in function table)
    uint entityCount;           // Number of entities to process
    uint requiredKeyCount;      // Number of required component keys
    uint requiredKeys[DISPATCH_MAX_REQUIRED_KEYS];  // String indices of required keys
    float dt;                   // Delta time (passed to system function)
    uint flags;                 // Dispatch flags
    uint padding0;
    uint padding1;
};

struct DispatchState {
    uint processedCount;        // Entities processed
    uint skippedCount;          // Entities skipped (didn't match filter)
    uint errorCount;            // Entities that hit errors
    uint spawnCount;            // Entities spawned during dispatch
    uint destroyCount;          // Entities marked for destruction
    uint status;                // Current dispatch status
    uint padding0;
    uint padding1;
};

// =============================================================================
// Dispatch Buffers
// =============================================================================

// Dispatch configuration (set by host)
RWStructuredBuffer<DispatchConfig> g_dispatchConfig;

// Dispatch state (updated during execution)
RWStructuredBuffer<DispatchState> g_dispatchState;

// Entity list to process (entity IDs or indices)
RWStructuredBuffer<uint> g_dispatchEntityList;

// Per-thread dispatch results (for debugging)
RWStructuredBuffer<uint> g_dispatchResults;

// =============================================================================
// External References (from other modules)
// =============================================================================

// From entity.slang
// uint entity_get_table_ptr(uint entityId);
// bool entity_is_valid(uint entityId);

// From spawn.slang  
// uint spawn_entity_gpu(uint tablePtr, uint sourceEntityId, uint threadId);

// From ops.slang
// void vm_set_current_entity(uint entityId, uint tablePtr);
// void vm_clear_current_entity();
// bool vm_has_component(XValue entity, XValue key);

// From table.slang
// XValue table_get(XValue t, XValue key);

// =============================================================================
// Dispatch Helper Functions
// =============================================================================

// Initialize dispatch state
void dispatch_init() {
    g_dispatchState[0].processedCount = 0;
    g_dispatchState[0].skippedCount = 0;
    g_dispatchState[0].errorCount = 0;
    g_dispatchState[0].spawnCount = 0;
    g_dispatchState[0].destroyCount = 0;
    g_dispatchState[0].status = DISPATCH_STATUS_IDLE;
}

// Check if entity matches required components
// entityTable: the entity's table
// requiredKeys: array of string indices for required component keys
// requiredKeyCount: number of required keys
bool dispatch_entity_matches(XValue entityTable, uint requiredKeys[DISPATCH_MAX_REQUIRED_KEYS], uint requiredKeyCount) {
    if (entityTable.type != TYPE_TABLE) {
        return false;
    }
    
    // Check each required key exists in the table
    for (uint i = 0; i < requiredKeyCount; i++) {
        XValue key = XValue::string(requiredKeys[i]);
        XValue value = table_get(entityTable, key);
        
        if (value.isNil()) {
            return false;  // Missing required component
        }
    }
    
    return true;
}

// Get dispatch statistics
void dispatch_get_stats(out uint processedCount, out uint skippedCount, out uint errorCount) {
    processedCount = g_dispatchState[0].processedCount;
    skippedCount = g_dispatchState[0].skippedCount;
    errorCount = g_dispatchState[0].errorCount;
}

// =============================================================================
// System Dispatch Kernel
// =============================================================================

// Main dispatch kernel - executes a system function on all matching entities
// Each thread processes one entity
[shader("compute")]
[numthreads(DISPATCH_THREAD_GROUP_SIZE, 1, 1)]
void system_dispatch(
    uint3 threadId : SV_DispatchThreadID,
    uint3 groupId : SV_GroupID,
    uint3 groupThreadId : SV_GroupThreadID
) {
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
        // Entity not valid, skip
        InterlockedAdd(g_dispatchState[0].skippedCount, 1);
        g_dispatchResults[entityIndex] = 0;  // Skipped
        return;
    }
    
    XValue entityTable = XValue::table(tablePtr);
    
    // Check if entity matches required components
    if (config.requiredKeyCount > 0) {
        if (!dispatch_entity_matches(entityTable, config.requiredKeys, config.requiredKeyCount)) {
            InterlockedAdd(g_dispatchState[0].skippedCount, 1);
            g_dispatchResults[entityIndex] = 0;  // Skipped
            return;
        }
    }
    
    // Set current entity for ECS operations in VM
    vm_set_current_entity(entityId, tablePtr);
    
    // Execute system function
    // In a full implementation, this would:
    // 1. Create a VM state for this thread
    // 2. Push the entity table and dt as arguments
    // 3. Call the function at config.functionIndex
    // 4. Handle any errors
    
    // For now, mark as processed
    InterlockedAdd(g_dispatchState[0].processedCount, 1);
    g_dispatchResults[entityIndex] = 1;  // Processed
    
    // Clear current entity
    vm_clear_current_entity();
}

// =============================================================================
// Filtered Dispatch Kernel
// =============================================================================

// Dispatch with inline filtering - skips entities that don't match
// More efficient when many entities don't match the filter
[shader("compute")]
[numthreads(DISPATCH_THREAD_GROUP_SIZE, 1, 1)]
void system_dispatch_filtered(
    uint3 threadId : SV_DispatchThreadID,
    uniform uint totalEntityCount,
    uniform uint functionIndex,
    uniform float dt
) {
    uint entityIndex = threadId.x;
    
    if (entityIndex >= totalEntityCount) {
        return;
    }
    
    // Get entity by pool index (iterating all entities)
    uint entityId = entity_get_by_index(entityIndex);
    
    if (entityId == 0xFFFFFFFF) {
        // Not a valid/active entity
        return;
    }
    
    // Get entity table
    uint tablePtr = entity_get_table_ptr(entityId);
    if (tablePtr == 0) {
        return;
    }
    
    XValue entityTable = XValue::table(tablePtr);
    
    // Set current entity
    vm_set_current_entity(entityId, tablePtr);
    
    // Process entity (VM execution would happen here)
    InterlockedAdd(g_dispatchState[0].processedCount, 1);
    
    // Clear current entity
    vm_clear_current_entity();
}

// =============================================================================
// Dispatch Initialization Kernel
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void dispatch_init_kernel() {
    dispatch_init();
    g_dispatchState[0].status = DISPATCH_STATUS_RUNNING;
}

// =============================================================================
// Dispatch Finalization Kernel
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void dispatch_finalize_kernel() {
    g_dispatchState[0].status = DISPATCH_STATUS_COMPLETED;
}

// =============================================================================
// Build Entity List Kernel
// =============================================================================

// Builds a list of entities matching the filter criteria
// This is a pre-pass to create a compact list for dispatch
RWStructuredBuffer<uint> g_filteredEntityCount;

[shader("compute")]
[numthreads(DISPATCH_THREAD_GROUP_SIZE, 1, 1)]
void build_entity_list(
    uint3 threadId : SV_DispatchThreadID,
    uniform uint totalEntityCount
) {
    uint entityIndex = threadId.x;
    
    if (entityIndex >= totalEntityCount) {
        return;
    }
    
    // Get entity by pool index
    uint entityId = entity_get_by_index(entityIndex);
    
    if (entityId == 0xFFFFFFFF) {
        return;
    }
    
    // Get entity table
    uint tablePtr = entity_get_table_ptr(entityId);
    if (tablePtr == 0) {
        return;
    }
    
    XValue entityTable = XValue::table(tablePtr);
    DispatchConfig config = g_dispatchConfig[0];
    
    // Check filter
    if (config.requiredKeyCount > 0) {
        if (!dispatch_entity_matches(entityTable, config.requiredKeys, config.requiredKeyCount)) {
            return;
        }
    }
    
    // Add to filtered list
    uint listIndex;
    InterlockedAdd(g_filteredEntityCount[0], 1, listIndex);
    g_dispatchEntityList[listIndex] = entityId;
}

// =============================================================================
// Dispatch Statistics
// =============================================================================

// Get number of entities processed in last dispatch
uint dispatch_get_processed_count() {
    return g_dispatchState[0].processedCount;
}

// Get number of entities skipped in last dispatch
uint dispatch_get_skipped_count() {
    return g_dispatchState[0].skippedCount;
}

// Get dispatch status
uint dispatch_get_status() {
    return g_dispatchState[0].status;
}

