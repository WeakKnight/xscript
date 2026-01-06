// XScript Virtual Machine (32-bit)
// Core VM implementation

// =============================================================================
// VM Buffers
// =============================================================================

// Bytecode buffer
RWStructuredBuffer<uint> g_bytecode;

// Constant pool (XValue format)
RWStructuredBuffer<XValue> g_constants;

// Global variables
RWStructuredBuffer<XValue> g_globals;

// Function descriptors
RWStructuredBuffer<FunctionDescriptor> g_functions;

// VM states (one per thread)
RWStructuredBuffer<VMState> g_vmStates;

// Call stacks (one per thread)
RWStructuredBuffer<CallFrame> g_callStacks;

// Call depth per thread
RWStructuredBuffer<uint> g_callDepths;

// Host call requests
struct HostCallRequest {
    uint vmIndex;       // Which VM is requesting
    uint funcIndex;     // Host function index
    uint argCount;      // Number of arguments
    uint argOffset;     // Offset in arg buffer
    uint status;        // 0 = pending, 1 = complete
};
RWStructuredBuffer<HostCallRequest> g_hostCallRequests;
RWStructuredBuffer<uint> g_hostCallCount;

// Host call arguments and results
RWStructuredBuffer<XValue> g_hostCallArgs;
RWStructuredBuffer<XValue> g_hostCallResults;

// =============================================================================
// ECS Buffers (from entity.slang, spawn.slang, dispatch.slang)
// =============================================================================

// Entity dispatch list (entity IDs to process)
RWStructuredBuffer<uint> g_dispatchEntityList;

// Dispatch state (from dispatch.slang)
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
RWStructuredBuffer<DispatchState> g_dispatchState;

// =============================================================================
// Bytecode Reading
// =============================================================================

uint read_byte(uint pc) {
    uint wordIndex = pc / 4;
    uint byteIndex = pc % 4;
    uint word = g_bytecode[wordIndex];
    return (word >> (byteIndex * 8)) & 0xFF;
}

uint read_uint16(uint pc) {
    return read_byte(pc) | (read_byte(pc + 1) << 8);
}

int read_int16(uint pc) {
    uint u = read_uint16(pc);
    if (u >= 0x8000) {
        return int(u) - 0x10000;
    }
    return int(u);
}

uint read_uint24(uint pc) {
    return read_byte(pc) | (read_byte(pc + 1) << 8) | (read_byte(pc + 2) << 16);
}

// =============================================================================
// VM Execution Step
// =============================================================================

// Execute one instruction
void vm_step(inout VMState state, uint vmIndex) {
    if (state.status != VM_STATUS_RUNNING) {
        return;
    }
    
    uint opcode = read_byte(state.pc);
    state.pc++;
    
    XValue a, b, result;
    uint operand8, operand16;
    int offset;
    
    switch (opcode) {
        // ===== Stack Operations =====
        case OP_NOP:
            break;
            
        case OP_PUSH_NIL:
            vm_push(state, XValue::nil());
            break;
            
        case OP_PUSH_TRUE:
            vm_push(state, XValue::boolean(true));
            break;
            
        case OP_PUSH_FALSE:
            vm_push(state, XValue::boolean(false));
            break;
            
        case OP_PUSH_NUM:
            operand16 = read_uint16(state.pc);
            state.pc += 2;
            vm_push(state, g_constants[operand16]);
            break;
            
        case OP_PUSH_STR:
            operand16 = read_uint16(state.pc);
            state.pc += 2;
            vm_push(state, g_constants[operand16]);
            break;
            
        case OP_POP:
            vm_pop(state, a);
            xvalue_decref(a);
            break;
            
        case OP_DUP:
            a = vm_peek(state, 0);
            xvalue_incref(a);
            vm_push(state, a);
            break;
            
        case OP_SWAP:
            a = state.stack[state.sp - 1];
            state.stack[state.sp - 1] = state.stack[state.sp - 2];
            state.stack[state.sp - 2] = a;
            break;
            
        // ===== Local/Global Variables =====
        case OP_GET_LOCAL:
            operand8 = read_byte(state.pc);
            state.pc++;
            a = vm_get_local(state, operand8);
            xvalue_incref(a);
            vm_push(state, a);
            break;
            
        case OP_SET_LOCAL:
            operand8 = read_byte(state.pc);
            state.pc++;
            vm_pop(state, a);
            vm_set_local(state, operand8, a);
            break;
            
        case OP_GET_GLOBAL:
            operand16 = read_uint16(state.pc);
            state.pc += 2;
            a = g_globals[operand16];
            xvalue_incref(a);
            vm_push(state, a);
            break;
            
        case OP_SET_GLOBAL:
            operand16 = read_uint16(state.pc);
            state.pc += 2;
            vm_pop(state, a);
            xvalue_assign(g_globals[operand16], a);
            break;
            
        // ===== Arithmetic (32-bit float) =====
        case OP_ADD:
            vm_pop(state, b);
            vm_pop(state, a);
            result = vm_binop_add(state, a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_SUB:
            vm_pop(state, b);
            vm_pop(state, a);
            result = vm_binop_sub(state, a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_MUL:
            vm_pop(state, b);
            vm_pop(state, a);
            result = vm_binop_mul(state, a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_DIV:
            vm_pop(state, b);
            vm_pop(state, a);
            result = vm_binop_div(state, a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_MOD:
            vm_pop(state, b);
            vm_pop(state, a);
            result = vm_binop_mod(state, a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_NEG:
            vm_pop(state, a);
            result = xvalue_neg(a);
            vm_push(state, result);
            xvalue_decref(a);
            break;
            
        case OP_POW:
            vm_pop(state, b);
            vm_pop(state, a);
            result = vm_binop_pow(state, a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        // ===== Comparison =====
        case OP_EQ:
            vm_pop(state, b);
            vm_pop(state, a);
            result = xvalue_eq(a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_NE:
            vm_pop(state, b);
            vm_pop(state, a);
            result = xvalue_ne(a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_LT:
            vm_pop(state, b);
            vm_pop(state, a);
            result = xvalue_lt(a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_LE:
            vm_pop(state, b);
            vm_pop(state, a);
            result = xvalue_le(a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_GT:
            vm_pop(state, b);
            vm_pop(state, a);
            result = xvalue_gt(a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_GE:
            vm_pop(state, b);
            vm_pop(state, a);
            result = xvalue_ge(a, b);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_NOT:
            vm_pop(state, a);
            result = xvalue_not(a);
            vm_push(state, result);
            xvalue_decref(a);
            break;
            
        // ===== Control Flow =====
        case OP_JMP:
            offset = read_int16(state.pc);
            state.pc += 2;
            state.pc = uint(int(state.pc) + offset);
            break;
            
        case OP_JMP_IF:
            offset = read_int16(state.pc);
            state.pc += 2;
            vm_pop(state, a);
            if (a.isTruthy()) {
                state.pc = uint(int(state.pc) + offset);
            }
            xvalue_decref(a);
            break;
            
        case OP_JMP_IF_NOT:
            offset = read_int16(state.pc);
            state.pc += 2;
            vm_pop(state, a);
            if (a.isFalsy()) {
                state.pc = uint(int(state.pc) + offset);
            }
            xvalue_decref(a);
            break;
            
        case OP_LOOP:
            offset = read_int16(state.pc);
            state.pc += 2;
            state.pc = uint(int(state.pc) - offset);
            break;
            
        // ===== Function Calls =====
        case OP_CALL:
            operand8 = read_byte(state.pc);
            state.pc++;
            // Get function from stack
            a = vm_peek(state, operand8);  // Function is below args
            if (a.type == TYPE_FUNCTION) {
                vm_prepare_call(state, a.asUint(), operand8,
                               g_functions, g_callStacks,
                               g_callDepths[vmIndex]);
            } else if (a.type == TYPE_TABLE) {
                // Check for __call metamethod
                XValue callMeta = table_getmetamethod(a, META_CALL);
                if (!callMeta.isNil() && callMeta.type == TYPE_FUNCTION) {
                    vm_prepare_call(state, callMeta.asUint(), operand8 + 1,
                                   g_functions, g_callStacks,
                                   g_callDepths[vmIndex]);
                } else {
                    state.status = VM_STATUS_ERROR;
                    state.error = ERR_TYPE_ERROR;
                }
            } else {
                state.status = VM_STATUS_ERROR;
                state.error = ERR_TYPE_ERROR;
            }
            break;
            
        case OP_RETURN:
            operand8 = read_byte(state.pc);
            state.pc++;
            vm_return(state, operand8, g_callStacks, g_callDepths[vmIndex]);
            break;
            
        case OP_CALL_HOST:
            operand16 = read_uint16(state.pc);
            state.pc += 2;
            operand8 = read_byte(state.pc);
            state.pc++;
            // Request host call
            {
                uint reqIdx;
                InterlockedAdd(g_hostCallCount[0], 1, reqIdx);
                
                HostCallRequest req;
                req.vmIndex = vmIndex;
                req.funcIndex = operand16;
                req.argCount = operand8;
                req.argOffset = reqIdx * 8;  // Max 8 args per call
                req.status = 0;
                g_hostCallRequests[reqIdx] = req;
                
                // Copy arguments
                for (uint i = 0; i < operand8; i++) {
                    g_hostCallArgs[req.argOffset + i] = vm_peek(state, operand8 - 1 - i);
                }
                
                // Pop arguments
                state.sp -= operand8;
            }
            state.status = VM_STATUS_WAITING_HOST;
            break;
            
        // ===== Table Operations =====
        case OP_NEW_TABLE:
            operand8 = read_byte(state.pc);
            state.pc++;
            result = table_new(operand8 == 0 ? 8 : operand8 * 2);
            vm_push(state, result);
            break;
            
        case OP_GET_TABLE:
            vm_pop(state, b);  // key
            vm_pop(state, a);  // table
            result = vm_table_get(a, b);
            xvalue_incref(result);
            vm_push(state, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_SET_TABLE:
            vm_pop(state, result);  // value
            vm_pop(state, b);       // key
            vm_pop(state, a);       // table
            vm_table_set(a, b, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_GET_FIELD:
            operand16 = read_uint16(state.pc);
            state.pc += 2;
            vm_pop(state, a);  // table
            b = g_constants[operand16];  // key from constants
            result = vm_table_get(a, b);
            xvalue_incref(result);
            vm_push(state, result);
            xvalue_decref(a);
            break;
            
        case OP_SET_FIELD:
            operand16 = read_uint16(state.pc);
            state.pc += 2;
            vm_pop(state, result);  // value
            vm_pop(state, a);       // table
            b = g_constants[operand16];  // key from constants
            vm_table_set(a, b, result);
            xvalue_decref(a);
            break;
            
        // ===== Metatable Operations =====
        case OP_GET_META:
            a = vm_peek(state, 0);
            result = table_getmetatable(a);
            xvalue_incref(result);
            state.stack[state.sp - 1] = result;
            xvalue_decref(a);
            break;
            
        case OP_SET_META:
            vm_pop(state, b);  // metatable
            vm_pop(state, a);  // table
            table_setmetatable(a, b);
            vm_push(state, a);  // Push table back
            xvalue_decref(b);
            break;
            
        // ===== ECS Operations =====
        case OP_SPAWN_ENTITY:
            // Pop table, spawn entity via spawn buffer, push pending spawn ID
            vm_pop(state, a);  // table to spawn
            if (a.type == TYPE_TABLE) {
                uint spawnId = spawn_entity_gpu(a.asUint(), vm_get_current_entity_id(), vmIndex);
                vm_push(state, XValue::number(float(spawnId)));
            } else {
                state.status = VM_STATUS_ERROR;
                state.error = ERR_TYPE_ERROR;
            }
            break;
            
        case OP_DESTROY_ENTITY:
            // Pop entity ID, mark for destruction
            vm_pop(state, a);
            if (a.type == TYPE_NUMBER) {
                uint entityId = uint(a.asNumber());
                entity_destroy(entityId);
            }
            // Destroying current entity is also valid
            else if (a.type == TYPE_TABLE) {
                // If passed a table, destroy current entity
                uint currentId = vm_get_current_entity_id();
                if (currentId != 0xFFFFFFFF) {
                    entity_destroy(currentId);
                }
            }
            break;
            
        case OP_GET_ENTITY:
            // Push current dispatch entity table
            result = vm_get_current_entity();
            xvalue_incref(result);
            vm_push(state, result);
            break;
            
        case OP_GET_ENTITY_ID:
            // Push current dispatch entity ID as number
            {
                uint currentId = vm_get_current_entity_id();
                vm_push(state, XValue::number(float(currentId)));
            }
            break;
            
        case OP_HAS_COMPONENT:
            // Pop key, pop entity, push bool
            vm_pop(state, b);  // key
            vm_pop(state, a);  // entity table
            {
                bool hasComp = vm_has_component(a, b);
                vm_push(state, XValue::boolean(hasComp));
            }
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_ADD_COMPONENT:
            // Pop value, pop key, pop entity
            vm_pop(state, result);  // value
            vm_pop(state, b);       // key
            vm_pop(state, a);       // entity table
            vm_add_component(a, b, result);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        case OP_REMOVE_COMPONENT:
            // Pop key, pop entity
            vm_pop(state, b);  // key
            vm_pop(state, a);  // entity table
            vm_remove_component(a, b);
            xvalue_decref(a);
            xvalue_decref(b);
            break;
            
        // ===== Special =====
        case OP_HALT:
            state.status = VM_STATUS_COMPLETED;
            break;
            
        default:
            state.status = VM_STATUS_ERROR;
            state.error = ERR_INVALID_OPCODE;
            break;
    }
}

// =============================================================================
// VM Compute Shader Entry Points
// =============================================================================

// Execute VMs for multiple steps
[shader("compute")]
[numthreads(64, 1, 1)]
void vm_execute(uint3 threadId : SV_DispatchThreadID, 
                uniform uint vmCount,
                uniform uint maxSteps) {
    uint vmIndex = threadId.x;
    
    if (vmIndex >= vmCount) return;
    
    VMState state = g_vmStates[vmIndex];
    
    // Execute up to maxSteps instructions
    for (uint step = 0; step < maxSteps; step++) {
        if (state.status != VM_STATUS_RUNNING) {
            break;
        }
        
        vm_step(state, vmIndex);
    }
    
    // Write back state
    g_vmStates[vmIndex] = state;
}

// Initialize a VM
[shader("compute")]
[numthreads(1, 1, 1)]
void vm_init(uniform uint vmIndex, uniform uint entryPoint) {
    VMState state;
    
    // Initialize stack
    for (uint i = 0; i < VM_STACK_SIZE; i++) {
        state.stack[i] = XValue::nil();
    }
    
    state.sp = 0;
    state.fp = 0;
    state.pc = entryPoint;
    state.status = VM_STATUS_RUNNING;
    state.error = ERR_NONE;
    
    g_vmStates[vmIndex] = state;
    g_callDepths[vmIndex] = 0;
}

// Resume VM after host call
[shader("compute")]
[numthreads(64, 1, 1)]
void vm_resume(uint3 threadId : SV_DispatchThreadID,
               uniform uint requestCount) {
    uint reqIndex = threadId.x;
    
    if (reqIndex >= requestCount) return;
    
    HostCallRequest req = g_hostCallRequests[reqIndex];
    
    if (req.status != 1) return;  // Not completed
    
    VMState state = g_vmStates[req.vmIndex];
    
    // Push result onto stack
    XValue result = g_hostCallResults[reqIndex];
    vm_push(state, result);
    
    state.status = VM_STATUS_RUNNING;
    
    g_vmStates[req.vmIndex] = state;
}

// =============================================================================
// Initialization
// =============================================================================

[shader("compute")]
[numthreads(1, 1, 1)]
void xscript_init(uniform uint heapSize,
                  uniform uint stringPoolSize) {
    // Initialize subsystems
    heap_init(heapSize);
    string_pool_init(stringPoolSize);
    gc_init();
    
    // Initialize ECS subsystems
    entity_pool_init();
    spawn_buffer_init();
    
    // Clear host call count
    g_hostCallCount[0] = 0;
}

// =============================================================================
// ECS System Execution
// =============================================================================

// Execute VM for ECS system dispatch (entity is pre-set)
[shader("compute")]
[numthreads(64, 1, 1)]
void vm_execute_system(
    uint3 threadId : SV_DispatchThreadID,
    uniform uint entityCount,
    uniform uint entryPoint,
    uniform uint maxSteps,
    uniform float dt
) {
    uint entityIndex = threadId.x;
    
    if (entityIndex >= entityCount) return;
    
    // Get entity from dispatch list
    uint entityId = g_dispatchEntityList[entityIndex];
    uint tablePtr = entity_get_table_ptr(entityId);
    
    if (tablePtr == 0) return;  // Invalid entity
    
    // Set current entity for ECS operations
    vm_set_current_entity(entityId, tablePtr);
    
    // Initialize VM state for this thread
    VMState state;
    for (uint i = 0; i < VM_STACK_SIZE; i++) {
        state.stack[i] = XValue::nil();
    }
    
    state.sp = 0;
    state.fp = 0;
    state.pc = entryPoint;
    state.status = VM_STATUS_RUNNING;
    state.error = ERR_NONE;
    
    // Push entity table and dt as arguments
    vm_push(state, XValue::table(tablePtr));
    vm_push(state, XValue::number(dt));
    
    // Execute VM
    for (uint step = 0; step < maxSteps; step++) {
        if (state.status != VM_STATUS_RUNNING) {
            break;
        }
        vm_step(state, entityIndex);
    }
    
    // Clear current entity
    vm_clear_current_entity();
    
    // Update dispatch stats
    if (state.status == VM_STATUS_COMPLETED || state.status == VM_STATUS_RUNNING) {
        InterlockedAdd(g_dispatchState[0].processedCount, 1);
    } else if (state.status == VM_STATUS_ERROR) {
        InterlockedAdd(g_dispatchState[0].errorCount, 1);
    }
}
