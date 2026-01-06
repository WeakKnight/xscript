// XScript VM Operations (32-bit)
// Instruction implementations

import value;
import gc;
import table;

// =============================================================================
// Opcode Definitions
// =============================================================================

// Stack operations
static const uint OP_NOP        = 0x00;
static const uint OP_PUSH_NIL   = 0x01;
static const uint OP_PUSH_TRUE  = 0x02;
static const uint OP_PUSH_FALSE = 0x03;
static const uint OP_PUSH_NUM   = 0x04;
static const uint OP_PUSH_STR   = 0x05;
static const uint OP_POP        = 0x06;
static const uint OP_DUP        = 0x07;
static const uint OP_SWAP       = 0x08;

// Local/Global variables
static const uint OP_GET_LOCAL   = 0x10;
static const uint OP_SET_LOCAL   = 0x11;
static const uint OP_GET_GLOBAL  = 0x12;
static const uint OP_SET_GLOBAL  = 0x13;
static const uint OP_GET_UPVALUE = 0x14;
static const uint OP_SET_UPVALUE = 0x15;

// Arithmetic
static const uint OP_ADD = 0x20;
static const uint OP_SUB = 0x21;
static const uint OP_MUL = 0x22;
static const uint OP_DIV = 0x23;
static const uint OP_MOD = 0x24;
static const uint OP_NEG = 0x25;
static const uint OP_POW = 0x26;

// Comparison
static const uint OP_EQ  = 0x30;
static const uint OP_NE  = 0x31;
static const uint OP_LT  = 0x32;
static const uint OP_LE  = 0x33;
static const uint OP_GT  = 0x34;
static const uint OP_GE  = 0x35;
static const uint OP_NOT = 0x36;
static const uint OP_AND = 0x37;
static const uint OP_OR  = 0x38;

// Control flow
static const uint OP_JMP        = 0x40;
static const uint OP_JMP_IF     = 0x41;
static const uint OP_JMP_IF_NOT = 0x42;
static const uint OP_LOOP       = 0x43;

// Function calls
static const uint OP_CALL      = 0x50;
static const uint OP_RETURN    = 0x51;
static const uint OP_CALL_HOST = 0x52;

// Table operations
static const uint OP_NEW_TABLE = 0x60;
static const uint OP_GET_TABLE = 0x61;
static const uint OP_SET_TABLE = 0x62;
static const uint OP_GET_FIELD = 0x63;
static const uint OP_SET_FIELD = 0x64;

// Metatable operations
static const uint OP_GET_META    = 0x70;
static const uint OP_SET_META    = 0x71;
static const uint OP_INVOKE_META = 0x72;

// ECS operations
static const uint OP_SPAWN_ENTITY     = 0x80;  // Pop table, spawn entity, push entity ID
static const uint OP_DESTROY_ENTITY   = 0x81;  // Pop entity ID, mark for destruction
static const uint OP_GET_ENTITY       = 0x82;  // Push current dispatch entity table
static const uint OP_GET_ENTITY_ID    = 0x83;  // Push current dispatch entity ID
static const uint OP_HAS_COMPONENT    = 0x84;  // Pop key, pop entity, push bool
static const uint OP_ADD_COMPONENT    = 0x85;  // Pop value, pop key, pop entity
static const uint OP_REMOVE_COMPONENT = 0x86;  // Pop key, pop entity

// Special
static const uint OP_HALT  = 0xFF;

// =============================================================================
// VM State
// =============================================================================

static const uint VM_STATUS_RUNNING      = 0;
static const uint VM_STATUS_PAUSED       = 1;
static const uint VM_STATUS_WAITING_HOST = 2;
static const uint VM_STATUS_ERROR        = 3;
static const uint VM_STATUS_COMPLETED    = 4;

// Stack size reduced for GPU compatibility
// For GPU dispatch, ECS system functions typically need small stacks
// Stack: 32 XValues * 12 bytes = 384 bytes, leaving plenty of room
static const uint VM_STACK_SIZE = 32;
static const uint VM_CALL_STACK_SIZE = 8;

struct VMState {
    XValue stack[VM_STACK_SIZE];
    uint sp;        // Stack pointer
    uint fp;        // Frame pointer
    uint pc;        // Program counter
    uint status;    // Execution status
    uint error;     // Error code (if status == ERROR)
};

struct CallFrame {
    uint returnPC;   // Return address
    uint prevFP;     // Previous frame pointer
    uint localBase;  // Base of local variables
    uint argCount;   // Number of arguments
};

// =============================================================================
// Error Codes
// =============================================================================

static const uint ERR_NONE            = 0;
static const uint ERR_STACK_OVERFLOW  = 1;
static const uint ERR_STACK_UNDERFLOW = 2;
static const uint ERR_TYPE_ERROR      = 3;
static const uint ERR_DIV_BY_ZERO     = 4;
static const uint ERR_INVALID_OPCODE  = 5;
static const uint ERR_OUT_OF_MEMORY   = 6;
static const uint ERR_CALL_DEPTH      = 7;

// =============================================================================
// Stack Operations
// =============================================================================

// Push value onto stack
bool vm_push(inout VMState state, XValue v) {
    if (state.sp >= VM_STACK_SIZE) {
        state.status = VM_STATUS_ERROR;
        state.error = ERR_STACK_OVERFLOW;
        return false;
    }
    
    state.stack[state.sp] = v;
    state.sp++;
    return true;
}

// Pop value from stack
bool vm_pop(inout VMState state, out XValue v) {
    if (state.sp == 0) {
        state.status = VM_STATUS_ERROR;
        state.error = ERR_STACK_UNDERFLOW;
        v = XValue::nil();
        return false;
    }
    
    state.sp--;
    v = state.stack[state.sp];
    return true;
}

// Peek at stack top
XValue vm_peek(VMState state, uint offset) {
    if (state.sp <= offset) {
        return XValue::nil();
    }
    return state.stack[state.sp - 1 - offset];
}

// Get local variable
XValue vm_get_local(VMState state, uint slot) {
    uint index = state.fp + slot;
    if (index >= state.sp) {
        return XValue::nil();
    }
    return state.stack[index];
}

// Set local variable
void vm_set_local(inout VMState state, uint slot, XValue v) {
    uint index = state.fp + slot;
    if (index < VM_STACK_SIZE) {
        xvalue_assign(state.stack[index], v);
    }
}

// =============================================================================
// Binary Operations with Metamethod Support (32-bit float)
// =============================================================================

XValue vm_binop_add(VMState state, XValue a, XValue b) {
    // Check for numbers
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_add(a, b);
    }
    
    // Check for metamethod
    XValue meta = XValue::nil();
    if (a.type == TYPE_TABLE) {
        meta = table_getmetamethod(a, META_ADD);
    }
    if (meta.isNil() && b.type == TYPE_TABLE) {
        meta = table_getmetamethod(b, META_ADD);
    }
    
    // If metamethod found, caller should invoke it
    // For now, return nil for type error
    if (!meta.isNil()) {
        // Return special value indicating metamethod call needed
        // This is simplified - full implementation would handle this properly
    }
    
    return XValue::nil();
}

XValue vm_binop_sub(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_sub(a, b);
    }
    return XValue::nil();
}

XValue vm_binop_mul(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_mul(a, b);
    }
    return XValue::nil();
}

XValue vm_binop_div(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        float bNum = b.asNumber();
        if (bNum == 0.0f) {
            // Division by zero - return inf
            return XValue::number(1.0f / 0.0f);
        }
        return xvalue_div(a, b);
    }
    return XValue::nil();
}

XValue vm_binop_mod(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_mod(a, b);
    }
    return XValue::nil();
}

XValue vm_binop_pow(VMState state, XValue a, XValue b) {
    if (a.type == TYPE_NUMBER && b.type == TYPE_NUMBER) {
        return xvalue_pow(a, b);
    }
    return XValue::nil();
}

// =============================================================================
// Table Operations
// =============================================================================

XValue vm_table_get(XValue t, XValue key) {
    if (t.type != TYPE_TABLE) {
        return XValue::nil();
    }
    
    // Iterative lookup with metatable chain (max 8 levels to prevent infinite loops)
    XValue currentTable = t;
    for (uint depth = 0; depth < 8; depth++) {
        XValue result = table_get(currentTable, key);
        
        // Found value, return it
        if (!result.isNil()) {
            return result;
        }
        
        // Check for __index metamethod
        XValue indexMeta = table_getmetamethod(currentTable, META_INDEX);
        if (indexMeta.isNil()) {
            // No more tables to search
            return XValue::nil();
        }
        
        // If __index is a table, continue lookup in it
        if (indexMeta.type == TYPE_TABLE) {
            currentTable = indexMeta;
            continue;
        }
        
        // If __index is a function, caller should invoke it
        // For now, return nil
        return XValue::nil();
    }
    
    return XValue::nil();
}

bool vm_table_set(XValue t, XValue key, XValue value) {
    if (t.type != TYPE_TABLE) {
        return false;
    }
    
    // Check if key exists
    XValue existing = table_get(t, key);
    
    if (existing.isNil()) {
        // Key doesn't exist, check __newindex
        XValue newindexMeta = table_getmetamethod(t, META_NEWINDEX);
        if (!newindexMeta.isNil()) {
            // Metamethod exists, caller should invoke it
            // For now, just proceed with normal set
        }
    }
    
    // Write barrier for GC
    gc_write_barrier(t, existing, value);
    
    return table_set(t, key, value);
}

// =============================================================================
// Function Call Support
// =============================================================================

struct FunctionDescriptor {
    uint codeOffset;     // Offset in bytecode
    uint paramCount;     // Number of parameters
    uint localCount;     // Number of local variables
    uint upvalueCount;   // Number of upvalues
    uint nameIndex;      // Function name (string index)
};

// Prepare for function call
bool vm_prepare_call(inout VMState state, uint funcIndex, uint argCount,
                     RWStructuredBuffer<FunctionDescriptor> funcs,
                     RWStructuredBuffer<CallFrame> callStack,
                     inout uint callDepth) {
    if (callDepth >= VM_CALL_STACK_SIZE) {
        state.status = VM_STATUS_ERROR;
        state.error = ERR_CALL_DEPTH;
        return false;
    }
    
    FunctionDescriptor func = funcs[funcIndex];
    
    // Save current frame
    CallFrame frame;
    frame.returnPC = state.pc;
    frame.prevFP = state.fp;
    frame.localBase = state.sp - argCount;
    frame.argCount = argCount;
    callStack[callDepth] = frame;
    callDepth++;
    
    // Set up new frame
    state.fp = state.sp - argCount;
    state.pc = func.codeOffset;
    
    // Initialize local variables to nil
    for (uint i = argCount; i < func.localCount; i++) {
        vm_push(state, XValue::nil());
    }
    
    return true;
}

// Return from function call
bool vm_return(inout VMState state, uint returnCount,
               RWStructuredBuffer<CallFrame> callStack,
               inout uint callDepth) {
    if (callDepth == 0) {
        // Return from top-level, execution complete
        state.status = VM_STATUS_COMPLETED;
        return true;
    }
    
    callDepth--;
    CallFrame frame = callStack[callDepth];
    
    // Save return values
    XValue returnValues[8];  // Max 8 return values
    uint actualReturns = min(returnCount, 8u);
    
    for (uint i = 0; i < actualReturns; i++) {
        returnValues[i] = vm_peek(state, actualReturns - 1 - i);
    }
    
    // Pop everything including locals and arguments
    state.sp = frame.localBase;
    
    // Push return values
    for (uint i = 0; i < actualReturns; i++) {
        vm_push(state, returnValues[i]);
    }
    
    // Restore previous frame
    state.fp = frame.prevFP;
    state.pc = frame.returnPC;
    
    return true;
}

// =============================================================================
// ECS Operations
// =============================================================================

// Current dispatch entity (set by dispatch kernel before executing VM)
// These are global state used during ECS dispatch
static uint g_currentEntityId = 0xFFFFFFFF;
static uint g_currentEntityTablePtr = 0;

// Set current dispatch entity (called by dispatch kernel)
void vm_set_current_entity(uint entityId, uint tablePtr) {
    g_currentEntityId = entityId;
    g_currentEntityTablePtr = tablePtr;
}

// Clear current dispatch entity
void vm_clear_current_entity() {
    g_currentEntityId = 0xFFFFFFFF;
    g_currentEntityTablePtr = 0;
}

// Get current dispatch entity table
XValue vm_get_current_entity() {
    if (g_currentEntityTablePtr == 0) {
        return XValue::nil();
    }
    return XValue::table(g_currentEntityTablePtr);
}

// Get current dispatch entity ID
uint vm_get_current_entity_id() {
    return g_currentEntityId;
}

// Check if entity table has a component (key)
// entity: entity table XValue
// key: component name (string XValue)
// Returns true if the key exists and value is not nil
bool vm_has_component(XValue entity, XValue key) {
    if (entity.type != TYPE_TABLE) {
        return false;
    }
    
    XValue value = table_get(entity, key);
    return !value.isNil();
}

// Add component to entity
// entity: entity table XValue
// key: component name (string XValue)
// value: component value
bool vm_add_component(XValue entity, XValue key, XValue value) {
    if (entity.type != TYPE_TABLE || key.isNil()) {
        return false;
    }
    
    return table_set(entity, key, value);
}

// Remove component from entity (set to nil)
// entity: entity table XValue
// key: component name (string XValue)
bool vm_remove_component(XValue entity, XValue key) {
    if (entity.type != TYPE_TABLE) {
        return false;
    }
    
    return table_set(entity, key, XValue::nil());
}
