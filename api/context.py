"""
XScript Context

The main interface for compiling and executing XScript code.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import struct

from .types import XValue, XTable, XFunction, TYPE_FUNCTION, TYPE_TABLE

# Runtime directory for Slang modules
RUNTIME_DIR = Path(__file__).parent.parent / "runtime"

# GPU Buffer sizes
MAX_ENTITIES = 65536
MAX_BYTECODE_WORDS = 16384
MAX_CONSTANTS = 1024
MAX_GLOBALS = 256
MAX_FUNCTIONS = 256
VM_STACK_SIZE = 160
VM_CALL_STACK_SIZE = 32
HEAP_SIZE = 1024 * 1024  # 1MB
STRING_POOL_SIZE = 64 * 1024  # 64KB
DISPATCH_THREAD_GROUP_SIZE = 64

# XValue struct size (type: u32, flags: u32, data: u32) = 12 bytes
XVALUE_SIZE = 12

# Entity slot size (tablePtr: u32, generation: u32, flags: u32, reserved: u32) = 16 bytes
ENTITY_SLOT_SIZE = 16

# VMState struct size (stack + sp + fp + pc + status + error)
VM_STATE_SIZE = (VM_STACK_SIZE * XVALUE_SIZE) + 20

# CallFrame size (returnPC + prevFP + localBase + argCount) = 16 bytes
CALL_FRAME_SIZE = 16

# FunctionDescriptor size
FUNCTION_DESC_SIZE = 20  # name_idx, arity, local_count, code_offset, code_length

# DispatchConfig size (functionIndex, entityCount, requiredKeyCount, requiredKeys[8], dt, flags, padding[2])
DISPATCH_CONFIG_SIZE = 60  # 3*4 + 8*4 + 4 + 4 + 8 = 60 bytes

# DispatchState size 
DISPATCH_STATE_SIZE = 32  # 8 * 4 bytes


@dataclass
class Filter:
    """
    Entity filter for dispatch operations.
    
    Filters entities by required component keys.
    """
    keys: Tuple[str, ...]
    _context: 'Context' = field(repr=False)
    
    def matches(self, entity: Optional[Dict[str, Any]]) -> bool:
        """Check if an entity matches this filter."""
        if entity is None:
            return False
        for key in self.keys:
            if key not in entity:
                return False
        return True


@dataclass
class DispatchStats:
    """Statistics from a dispatch operation."""
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    spawned: int = 0
    destroyed: int = 0

# Import compiler
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from compiler import Lexer, Parser, CodeGenerator, Bytecode


@dataclass
class Script:
    """
    A compiled XScript script.
    
    Contains bytecode and metadata ready for execution.
    """
    
    source: str
    bytecode: Bytecode
    filename: Optional[str] = None
    
    def disassemble(self) -> str:
        """Get disassembly of the bytecode."""
        return self.bytecode.disassemble()
    
    def save(self, path: str) -> None:
        """Save compiled bytecode to file."""
        data = self.bytecode.serialize()
        with open(path, 'wb') as f:
            f.write(data)
    
    @classmethod
    def load(cls, path: str) -> 'Script':
        """Load compiled bytecode from file."""
        with open(path, 'rb') as f:
            data = f.read()
        bytecode = Bytecode.deserialize(data)
        return cls(source="", bytecode=bytecode, filename=path)


class Context:
    """
    XScript execution context.
    
    Manages script compilation, execution, and host function registration.
    This is the main entry point for using XScript from Python.
    
    Example:
        ctx = Context()
        script = ctx.compile('var x = 10; return x * 2;')
        result = ctx.execute(script)
    """
    
    def __init__(self, 
                 device: str = "cpu",
                 heap_size: int = 1024 * 1024,
                 string_pool_size: int = 64 * 1024,
                 debug: bool = False):
        """
        Create a new XScript context.
        
        Args:
            device: Execution device ("cpu" or "cuda")
            heap_size: Size of heap memory in bytes
            string_pool_size: Size of string pool in bytes
            debug: Enable debug mode
        """
        self.device = device
        self.heap_size = heap_size
        self.string_pool_size = string_pool_size
        self.debug = debug
        
        # Global variables
        self._globals: Dict[str, XValue] = {}
        
        # Registered host functions
        self._host_functions: Dict[str, XFunction] = {}
        self._host_function_list: List[XFunction] = []
        
        # String pool
        self._strings: List[str] = []
        
        # GPU resources (will be initialized on first use)
        self._initialized = False
        self._device = None
        self._session = None
        self._vm_module = None
        
        # GPU buffers
        self._gpu_bytecode = None
        self._gpu_constants = None
        self._gpu_globals = None
        self._gpu_functions = None
        self._gpu_vm_states = None
        self._gpu_call_stacks = None
        self._gpu_call_depths = None
        self._gpu_heap = None
        self._gpu_heap_state = None
        self._gpu_string_pool = None
        self._gpu_string_pool_state = None
        self._gpu_entity_pool = None
        self._gpu_entity_pool_state = None
        self._gpu_entity_free_list = None
        self._gpu_entity_destroy_list = None
        self._gpu_entity_destroy_count = None
        self._gpu_spawn_buffer = None
        self._gpu_spawn_count = None
        self._gpu_spawn_buffer_state = None
        self._gpu_dispatch_config = None
        self._gpu_dispatch_state = None
        self._gpu_dispatch_entity_list = None
        self._gpu_dispatch_results = None
        
        # GPU kernel cache
        self._dispatch_kernel = None
        self._dispatch_init_kernel = None
        self._dispatch_finalize_kernel = None
        
        # Track if bytecode needs re-upload
        self._last_uploaded_bytecode = None
        
        # ECS entity storage (CPU-side)
        self._entities: Dict[int, XTable] = {}
        self._next_entity_id: int = 0
        self._entity_generation: Dict[int, int] = {}  # index -> generation
        
        # Register built-in functions
        self._register_builtins()
    
    def _register_builtins(self) -> None:
        """Register built-in functions."""
        
        @self.register("print")
        def builtin_print(*args):
            print(*[str(a) for a in args])
        
        @self.register("type")
        def builtin_type(value):
            if isinstance(value, XValue):
                type_names = ['nil', 'boolean', 'number', 'string', 
                             'table', 'function', 'userdata', 'thread']
                return type_names[value.type]
            # Map Python types to XScript type names
            if value is None:
                return 'nil'
            elif isinstance(value, bool):
                return 'boolean'
            elif isinstance(value, (int, float)):
                return 'number'
            elif isinstance(value, str):
                return 'string'
            elif isinstance(value, dict):
                return 'table'
            elif callable(value):
                return 'function'
            return type(value).__name__
        
        @self.register("tostring")
        def builtin_tostring(value):
            if isinstance(value, XValue):
                return str(value.to_python())
            return str(value)
        
        @self.register("tonumber")
        def builtin_tonumber(value):
            try:
                if isinstance(value, XValue):
                    return float(value.to_python())
                return float(value)
            except (ValueError, TypeError):
                return None
    
    def compile(self, source: str, filename: Optional[str] = None) -> Script:
        """
        Compile XScript source code.
        
        Args:
            source: XScript source code string
            filename: Optional filename for error messages
            
        Returns:
            Compiled Script object
        """
        # Tokenize
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        # Parse
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Generate bytecode
        codegen = CodeGenerator()
        bytecode = codegen.generate(ast)
        
        return Script(source=source, bytecode=bytecode, filename=filename)
    
    def compile_file(self, path: str) -> Script:
        """
        Compile XScript source file.
        
        Args:
            path: Path to .xs source file
            
        Returns:
            Compiled Script object
        """
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
        return self.compile(source, filename=path)
    
    def execute(self, script: Script) -> XValue:
        """
        Execute a compiled script.
        
        Args:
            script: Compiled Script object
            
        Returns:
            Result value (or nil)
        """
        if self.device == "cpu":
            return self._execute_cpu(script)
        else:
            return self._execute_gpu(script)
    
    def _execute_cpu(self, script: Script) -> XValue:
        """Execute script on CPU (interpreter mode)."""
        from .interpreter import Interpreter
        
        interp = Interpreter(self)
        result = interp.run(script.bytecode)
        
        # Register all functions from bytecode after execution
        # (this overwrites the number indices with proper XFunction objects)
        self._register_script_functions(script.bytecode)
        
        return result
    
    def _register_script_functions(self, bytecode: 'Bytecode') -> None:
        """Register script functions from bytecode as globals."""
        from .types import XFunction
        
        for func_info in bytecode.functions:
            xfunc = XFunction(
                name=func_info.name,
                arity=func_info.arity,
                is_host=False,
                host_func=None,
                code_offset=func_info.code_offset
            )
            self._globals[func_info.name] = XValue(TYPE_FUNCTION, xfunc)
    
    def _execute_gpu(self, script: Script) -> XValue:
        """Execute script on GPU via SlangPy."""
        # GPU execution requires SlangPy
        try:
            import slangpy as spy
        except ImportError:
            raise RuntimeError("GPU execution requires slangpy. Install with: pip install slangpy")
        
        if not self._initialized:
            self._initialize_gpu()
        
        # TODO: Implement GPU execution path
        # For now, fall back to CPU
        return self._execute_cpu(script)
    
    def _initialize_gpu(self) -> None:
        """Initialize GPU resources."""
        import slangpy as spy
        
        # Create device
        self._device = spy.Device()
        
        # Create session with include paths for runtime modules
        opts = spy.SlangCompilerOptions()
        opts.include_paths = [str(RUNTIME_DIR), str(RUNTIME_DIR / "tests")]
        self._session = self._device.create_slang_session(compiler_options=opts)
        
        # Load the main dispatch module (which imports vm.slang and others)
        self._vm_module = self._session.load_module(str(RUNTIME_DIR / "dispatch"))
        
        # Create GPU buffers
        buffer_usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        
        # Bytecode buffer
        self._gpu_bytecode = self._device.create_buffer(
            size=MAX_BYTECODE_WORDS * 4,
            usage=buffer_usage
        )
        
        # Constants buffer (XValue array)
        self._gpu_constants = self._device.create_buffer(
            size=MAX_CONSTANTS * XVALUE_SIZE,
            usage=buffer_usage
        )
        
        # Globals buffer (XValue array)
        self._gpu_globals = self._device.create_buffer(
            size=MAX_GLOBALS * XVALUE_SIZE,
            usage=buffer_usage
        )
        
        # Functions buffer (FunctionDescriptor array)
        self._gpu_functions = self._device.create_buffer(
            size=MAX_FUNCTIONS * FUNCTION_DESC_SIZE,
            usage=buffer_usage
        )
        
        # VM states buffer (one per potential entity)
        self._gpu_vm_states = self._device.create_buffer(
            size=MAX_ENTITIES * VM_STATE_SIZE,
            usage=buffer_usage
        )
        
        # Call stacks buffer
        self._gpu_call_stacks = self._device.create_buffer(
            size=MAX_ENTITIES * VM_CALL_STACK_SIZE * CALL_FRAME_SIZE,
            usage=buffer_usage
        )
        
        # Call depths buffer
        self._gpu_call_depths = self._device.create_buffer(
            size=MAX_ENTITIES * 4,
            usage=buffer_usage
        )
        
        # Heap buffer
        self._gpu_heap = self._device.create_buffer(
            size=HEAP_SIZE,
            usage=buffer_usage
        )
        
        # Heap state (freePtr, size)
        self._gpu_heap_state = self._device.create_buffer(
            size=8,
            usage=buffer_usage
        )
        
        # String pool buffer
        self._gpu_string_pool = self._device.create_buffer(
            size=STRING_POOL_SIZE,
            usage=buffer_usage
        )
        
        # String pool state
        self._gpu_string_pool_state = self._device.create_buffer(
            size=32,
            usage=buffer_usage
        )
        
        # Entity pool buffer
        self._gpu_entity_pool = self._device.create_buffer(
            size=MAX_ENTITIES * ENTITY_SLOT_SIZE,
            usage=buffer_usage
        )
        
        # Entity pool state
        self._gpu_entity_pool_state = self._device.create_buffer(
            size=32,
            usage=buffer_usage
        )
        
        # Entity free list
        self._gpu_entity_free_list = self._device.create_buffer(
            size=MAX_ENTITIES * 4,
            usage=buffer_usage
        )
        
        # Entity destroy list
        self._gpu_entity_destroy_list = self._device.create_buffer(
            size=MAX_ENTITIES * 4,
            usage=buffer_usage
        )
        
        # Entity destroy count
        self._gpu_entity_destroy_count = self._device.create_buffer(
            size=4,
            usage=buffer_usage
        )
        
        # Spawn buffer (SpawnRequest = 16 bytes each)
        self._gpu_spawn_buffer = self._device.create_buffer(
            size=1024 * 16,  # Max 1024 spawns per dispatch
            usage=buffer_usage
        )
        
        # Spawn count
        self._gpu_spawn_count = self._device.create_buffer(
            size=4,
            usage=buffer_usage
        )
        
        # Spawn buffer state
        self._gpu_spawn_buffer_state = self._device.create_buffer(
            size=32,
            usage=buffer_usage
        )
        
        # Dispatch config
        self._gpu_dispatch_config = self._device.create_buffer(
            size=DISPATCH_CONFIG_SIZE,
            usage=buffer_usage
        )
        
        # Dispatch state
        self._gpu_dispatch_state = self._device.create_buffer(
            size=DISPATCH_STATE_SIZE,
            usage=buffer_usage
        )
        
        # Dispatch entity list
        self._gpu_dispatch_entity_list = self._device.create_buffer(
            size=MAX_ENTITIES * 4,
            usage=buffer_usage
        )
        
        # Dispatch results
        self._gpu_dispatch_results = self._device.create_buffer(
            size=MAX_ENTITIES * 4,
            usage=buffer_usage
        )
        
        # Initialize heap state (freePtr=0, size=HEAP_SIZE)
        self._gpu_heap_state.copy_from_numpy(np.array([0, HEAP_SIZE], dtype=np.uint32))
        
        # Initialize entity pool state
        self._gpu_entity_pool_state.copy_from_numpy(np.zeros(8, dtype=np.uint32))
        
        # Initialize entity destroy count
        self._gpu_entity_destroy_count.copy_from_numpy(np.array([0], dtype=np.uint32))
        
        # Initialize spawn count
        self._gpu_spawn_count.copy_from_numpy(np.array([0], dtype=np.uint32))
        
        # Initialize dispatch state
        self._gpu_dispatch_state.copy_from_numpy(np.zeros(8, dtype=np.uint32))
        
        # Load dispatch kernels
        self._load_dispatch_kernels()
        
        self._initialized = True
        
        if self.debug:
            print(f"GPU initialized: {self._device}")
    
    def _load_dispatch_kernels(self) -> None:
        """Load dispatch compute kernels."""
        import slangpy as spy
        
        # Load the dispatch init kernel
        init_program = self._session.load_program(
            str(RUNTIME_DIR / "dispatch"),
            ["dispatch_init_kernel"]
        )
        init_desc = spy.ComputeKernelDesc()
        init_desc.program = init_program
        self._dispatch_init_kernel = self._device.create_compute_kernel(init_desc)
        
        # Load the main dispatch kernel
        dispatch_program = self._session.load_program(
            str(RUNTIME_DIR / "dispatch"),
            ["system_dispatch"]
        )
        dispatch_desc = spy.ComputeKernelDesc()
        dispatch_desc.program = dispatch_program
        self._dispatch_kernel = self._device.create_compute_kernel(dispatch_desc)
        
        # Load the dispatch finalize kernel
        finalize_program = self._session.load_program(
            str(RUNTIME_DIR / "dispatch"),
            ["dispatch_finalize_kernel"]
        )
        finalize_desc = spy.ComputeKernelDesc()
        finalize_desc.program = finalize_program
        self._dispatch_finalize_kernel = self._device.create_compute_kernel(finalize_desc)
    
    # =========================================================================
    # GPU Data Upload Methods
    # =========================================================================
    
    def _upload_bytecode(self, bytecode: 'Bytecode') -> None:
        """Upload bytecode, constants, and functions to GPU."""
        # Check if already uploaded
        if self._last_uploaded_bytecode is bytecode:
            return
        
        # Pack bytecode into uint32 words
        code_words = self._pack_bytecode_to_words(bytecode.code)
        self._gpu_bytecode.copy_from_numpy(code_words)
        
        # Upload constants
        constants_data = self._pack_constants(bytecode.constants)
        self._gpu_constants.copy_from_numpy(constants_data)
        
        # Upload function descriptors
        funcs_data = self._pack_functions(bytecode.functions)
        self._gpu_functions.copy_from_numpy(funcs_data)
        
        # Upload globals (initialize to nil)
        globals_data = self._pack_globals(bytecode.globals)
        self._gpu_globals.copy_from_numpy(globals_data)
        
        self._last_uploaded_bytecode = bytecode
        
        if self.debug:
            print(f"Uploaded bytecode: {len(bytecode.code)} bytes, "
                  f"{len(bytecode.constants)} constants, "
                  f"{len(bytecode.functions)} functions")
    
    def _pack_bytecode_to_words(self, code: bytearray) -> np.ndarray:
        """Pack bytecode bytes into uint32 words."""
        # Pad to multiple of 4
        padded_len = ((len(code) + 3) // 4) * 4
        padded_code = bytes(code) + b'\x00' * (padded_len - len(code))
        
        # Convert to uint32 array
        words = np.frombuffer(padded_code, dtype=np.uint32).copy()
        
        # Pad to MAX_BYTECODE_WORDS
        if len(words) < MAX_BYTECODE_WORDS:
            words = np.pad(words, (0, MAX_BYTECODE_WORDS - len(words)))
        
        return words
    
    def _pack_constants(self, constants: List) -> np.ndarray:
        """Pack constants into XValue format for GPU."""
        # XValue: type (u32), flags (u32), data (u32)
        data = np.zeros(MAX_CONSTANTS * 3, dtype=np.uint32)
        
        for i, const in enumerate(constants):
            if i >= MAX_CONSTANTS:
                break
            
            base = i * 3
            
            if const.type == 0:  # NIL
                data[base] = 0  # TYPE_NIL
                data[base + 1] = 0  # flags
                data[base + 2] = 0  # data
            elif const.type == 1:  # BOOL
                data[base] = 1  # TYPE_BOOL
                data[base + 1] = 0
                data[base + 2] = 1 if const.value else 0
            elif const.type == 2:  # NUMBER
                data[base] = 2  # TYPE_NUMBER
                data[base + 1] = 0
                # Pack float as uint32
                data[base + 2] = struct.unpack('I', struct.pack('f', float(const.value)))[0]
            elif const.type == 3:  # STRING
                data[base] = 3  # TYPE_STRING
                data[base + 1] = 0
                # String index - intern the string
                str_idx = self.intern_string(const.value)
                data[base + 2] = str_idx
        
        return data
    
    def _pack_functions(self, functions: List) -> np.ndarray:
        """Pack function descriptors for GPU."""
        # FunctionDescriptor: name_idx (u32), arity (u32), local_count (u32), 
        #                     code_offset (u32), code_length (u32)
        data = np.zeros(MAX_FUNCTIONS * 5, dtype=np.uint32)
        
        for i, func in enumerate(functions):
            if i >= MAX_FUNCTIONS:
                break
            
            base = i * 5
            data[base] = self.intern_string(func.name)
            data[base + 1] = func.arity
            data[base + 2] = func.local_count
            data[base + 3] = func.code_offset
            data[base + 4] = func.code_length
        
        return data
    
    def _pack_globals(self, globals_dict: Dict[str, int]) -> np.ndarray:
        """Initialize globals buffer with nil values."""
        # XValue: type (u32), flags (u32), data (u32)
        data = np.zeros(MAX_GLOBALS * 3, dtype=np.uint32)
        # All initialized to TYPE_NIL (0)
        return data
    
    # =========================================================================
    # GPU Table Conversion
    # =========================================================================
    
    # Table structure constants (matching table.slang)
    TABLE_INITIAL_CAPACITY = 8
    TABLE_OFF_CAPACITY = 0
    TABLE_OFF_COUNT = 1
    TABLE_OFF_METATABLE = 2
    TABLE_OFF_ARRAY_SIZE = 3
    TABLE_OFF_ENTRIES = 4
    ENTRY_SIZE_WORDS = 7  # key(3) + value(3) + next(1)
    
    def _dict_to_gpu_table(self, data: Dict[str, Any], heap_data: bytearray, 
                           heap_ptr: int) -> Tuple[int, int]:
        """
        Convert Python dict to GPU heap table format.
        
        Args:
            data: Python dictionary to convert
            heap_data: Heap byte array to write to
            heap_ptr: Current heap allocation pointer
            
        Returns:
            Tuple of (table_pointer, new_heap_ptr)
        """
        capacity = max(self.TABLE_INITIAL_CAPACITY, len(data) * 2)
        
        # Calculate size: header (4 uints) + entries (capacity * 7 uints)
        header_size = self.TABLE_OFF_ENTRIES
        entries_size = capacity * self.ENTRY_SIZE_WORDS
        total_size_bytes = (header_size + entries_size) * 4
        
        # Align to 4 bytes
        table_ptr = heap_ptr
        new_heap_ptr = heap_ptr + total_size_bytes
        
        # Ensure heap_data is large enough
        while len(heap_data) < new_heap_ptr:
            heap_data.extend(b'\x00' * 4096)
        
        # Write header
        def write_uint(offset: int, value: int):
            struct.pack_into('<I', heap_data, table_ptr + offset * 4, value)
        
        write_uint(self.TABLE_OFF_CAPACITY, capacity)
        write_uint(self.TABLE_OFF_COUNT, len(data))
        write_uint(self.TABLE_OFF_METATABLE, 0)
        write_uint(self.TABLE_OFF_ARRAY_SIZE, 0)
        
        # Initialize all entries to nil
        for i in range(capacity):
            entry_offset = self.TABLE_OFF_ENTRIES + i * self.ENTRY_SIZE_WORDS
            # Key = nil (type=0, flags=0, data=0)
            write_uint(entry_offset + 0, 0)  # key.type
            write_uint(entry_offset + 1, 0)  # key.flags
            write_uint(entry_offset + 2, 0)  # key.data
            # Value = nil
            write_uint(entry_offset + 3, 0)  # value.type
            write_uint(entry_offset + 4, 0)  # value.flags
            write_uint(entry_offset + 5, 0)  # value.data
            # Next = 0
            write_uint(entry_offset + 6, 0)
        
        # Insert each key-value pair
        for key, value in data.items():
            # Convert key to XValue format
            key_type, key_data = self._python_to_xvalue_parts(key, heap_data, new_heap_ptr)
            if isinstance(key_data, tuple):  # Nested table returned new heap ptr
                key_data, new_heap_ptr = key_data
            
            # Convert value to XValue format (may allocate nested tables)
            val_type, val_data = self._python_to_xvalue_parts(value, heap_data, new_heap_ptr)
            if isinstance(val_data, tuple):
                val_data, new_heap_ptr = val_data
            
            # Hash the key to find bucket
            key_hash = self._xvalue_hash(key_type, key_data)
            bucket = key_hash % capacity
            
            # Find empty slot (linear probing for simplicity)
            for probe in range(capacity):
                slot = (bucket + probe) % capacity
                entry_offset = self.TABLE_OFF_ENTRIES + slot * self.ENTRY_SIZE_WORDS
                
                # Check if slot is empty (key.type == TYPE_NIL)
                slot_type = struct.unpack_from('<I', heap_data, table_ptr + entry_offset * 4)[0]
                if slot_type == 0:  # Empty slot
                    # Write key
                    write_uint(entry_offset + 0, key_type)
                    write_uint(entry_offset + 1, 0)  # flags
                    write_uint(entry_offset + 2, key_data)
                    # Write value
                    write_uint(entry_offset + 3, val_type)
                    write_uint(entry_offset + 4, 0)  # flags
                    write_uint(entry_offset + 5, val_data)
                    break
        
        return table_ptr, new_heap_ptr
    
    def _python_to_xvalue_parts(self, value: Any, heap_data: bytearray, 
                                 heap_ptr: int) -> Tuple[int, Any]:
        """
        Convert Python value to XValue type and data parts.
        
        Returns:
            Tuple of (type_id, data) where data may be (ptr, new_heap_ptr) for tables
        """
        if value is None:
            return (0, 0)  # TYPE_NIL
        elif isinstance(value, bool):
            return (1, 1 if value else 0)  # TYPE_BOOL
        elif isinstance(value, (int, float)):
            # Pack float as uint32
            float_bytes = struct.pack('f', float(value))
            uint_val = struct.unpack('I', float_bytes)[0]
            return (2, uint_val)  # TYPE_NUMBER
        elif isinstance(value, str):
            str_idx = self.intern_string(value)
            return (3, str_idx)  # TYPE_STRING
        elif isinstance(value, dict):
            # Recursively create nested table
            nested_ptr, new_heap_ptr = self._dict_to_gpu_table(value, heap_data, heap_ptr)
            return (4, (nested_ptr, new_heap_ptr))  # TYPE_TABLE
        else:
            return (0, 0)  # Unknown -> nil
    
    def _xvalue_hash(self, type_id: int, data: int) -> int:
        """FNV-1a hash matching the GPU implementation."""
        FNV_OFFSET = 2166136261
        FNV_PRIME = 16777619
        
        h = FNV_OFFSET
        h ^= type_id
        h = (h * FNV_PRIME) & 0xFFFFFFFF
        h ^= data
        h = (h * FNV_PRIME) & 0xFFFFFFFF
        
        return h
    
    # =========================================================================
    # GPU Entity Pool Sync
    # =========================================================================
    
    # Entity constants (matching entity.slang)
    ENTITY_FLAG_NONE = 0x00
    ENTITY_FLAG_ACTIVE = 0x01
    ENTITY_FLAG_DESTROYED = 0x02
    
    def _sync_entities_to_gpu(self) -> List[int]:
        """
        Upload all entities to GPU entity pool and heap.
        
        Returns:
            List of entity IDs that were synced
        """
        # Build heap data
        heap_data = bytearray(HEAP_SIZE)
        heap_ptr = 0
        
        # Entity pool data: EntitySlot = tablePtr(u32), generation(u32), flags(u32), reserved(u32)
        entity_pool = np.zeros(MAX_ENTITIES * 4, dtype=np.uint32)
        
        # Entity pool state: activeCount, highWaterMark, freeListHead, freeListCount, ...
        pool_state = np.zeros(8, dtype=np.uint32)
        
        entity_ids = []
        
        for entity_id, table in self._entities.items():
            if entity_id >= MAX_ENTITIES:
                continue
            
            # Convert table to GPU format
            table_dict = table.to_dict()
            table_ptr, heap_ptr = self._dict_to_gpu_table(table_dict, heap_data, heap_ptr)
            
            # Write entity slot
            slot_base = entity_id * 4
            entity_pool[slot_base] = table_ptr
            entity_pool[slot_base + 1] = self._entity_generation.get(entity_id, 0)
            entity_pool[slot_base + 2] = self.ENTITY_FLAG_ACTIVE
            entity_pool[slot_base + 3] = 0  # reserved
            
            entity_ids.append(entity_id)
        
        # Update pool state
        pool_state[0] = len(entity_ids)  # activeCount
        pool_state[1] = max(entity_ids) + 1 if entity_ids else 0  # highWaterMark
        
        # Upload to GPU
        heap_array = np.frombuffer(heap_data[:heap_ptr + 4096], dtype=np.uint32).copy()
        if len(heap_array) < HEAP_SIZE // 4:
            heap_array = np.pad(heap_array, (0, HEAP_SIZE // 4 - len(heap_array)))
        
        self._gpu_heap.copy_from_numpy(heap_array)
        self._gpu_heap_state.copy_from_numpy(np.array([heap_ptr, HEAP_SIZE], dtype=np.uint32))
        self._gpu_entity_pool.copy_from_numpy(entity_pool)
        self._gpu_entity_pool_state.copy_from_numpy(pool_state)
        
        # Upload strings
        self._upload_string_pool()
        
        if self.debug:
            print(f"Synced {len(entity_ids)} entities to GPU, heap used: {heap_ptr} bytes")
        
        return entity_ids
    
    def _upload_string_pool(self) -> None:
        """Upload string pool to GPU."""
        # String pool format: count (u32), then for each string:
        #   length (u32), chars (padded to 4 bytes)
        
        string_data = bytearray()
        
        # Write count
        string_data.extend(struct.pack('<I', len(self._strings)))
        
        # Write each string
        for s in self._strings:
            encoded = s.encode('utf-8')
            padded_len = ((len(encoded) + 3) // 4) * 4
            string_data.extend(struct.pack('<I', len(encoded)))
            string_data.extend(encoded)
            string_data.extend(b'\x00' * (padded_len - len(encoded)))
        
        # Pad to STRING_POOL_SIZE
        while len(string_data) < STRING_POOL_SIZE:
            string_data.extend(b'\x00' * min(4096, STRING_POOL_SIZE - len(string_data)))
        
        string_array = np.frombuffer(bytes(string_data[:STRING_POOL_SIZE]), dtype=np.uint32).copy()
        self._gpu_string_pool.copy_from_numpy(string_array)
        
        # Update string pool state
        state = np.array([len(self._strings), len(string_data), 0, 0, 0, 0, 0, 0], dtype=np.uint32)
        self._gpu_string_pool_state.copy_from_numpy(state)
    
    def _build_filtered_entity_list(self, entity_filter: Filter) -> np.ndarray:
        """Build list of entity IDs matching the filter."""
        matching_ids = []
        
        for entity_id, table in self._entities.items():
            entity_dict = table.to_dict()
            if entity_filter.matches(entity_dict):
                matching_ids.append(entity_id)
        
        # Convert to numpy array
        if matching_ids:
            return np.array(matching_ids, dtype=np.uint32)
        else:
            return np.array([0], dtype=np.uint32)  # Dummy array
    
    def _get_function_index(self, func_name: str, bytecode: 'Bytecode') -> int:
        """Get function index by name from bytecode."""
        for i, func in enumerate(bytecode.functions):
            if func.name == func_name:
                return i
        raise NameError(f"Function '{func_name}' not found in bytecode")
    
    def _intern_filter_keys(self, entity_filter: Filter) -> np.ndarray:
        """Intern filter keys and return as numpy array."""
        keys = np.zeros(8, dtype=np.uint32)  # Max 8 keys
        
        for i, key in enumerate(entity_filter.keys[:8]):
            keys[i] = self.intern_string(key)
        
        return keys
    
    # =========================================================================
    # GPU Entity Readback
    # =========================================================================
    
    def _sync_entities_from_gpu(self) -> None:
        """Read modified entity data back from GPU."""
        # Read heap data
        heap_array = self._gpu_heap.to_numpy()
        heap_data = heap_array.tobytes()
        
        # Read entity pool
        entity_pool = self._gpu_entity_pool.to_numpy()
        
        # Read entity pool state
        pool_state = self._gpu_entity_pool_state.to_numpy()
        high_water_mark = pool_state[1]
        
        # Read strings for key lookup
        string_data = self._gpu_string_pool.to_numpy().tobytes()
        
        # Update each entity
        for entity_id in list(self._entities.keys()):
            if entity_id >= high_water_mark:
                continue
            
            slot_base = entity_id * 4
            table_ptr = entity_pool[slot_base]
            flags = entity_pool[slot_base + 2]
            
            # Skip destroyed entities
            if flags & self.ENTITY_FLAG_DESTROYED:
                del self._entities[entity_id]
                continue
            
            # Skip inactive entities  
            if not (flags & self.ENTITY_FLAG_ACTIVE):
                continue
            
            # Read table from heap
            table_dict = self._gpu_table_to_dict(heap_data, table_ptr)
            self._entities[entity_id] = XTable.from_dict(table_dict)
        
        if self.debug:
            print(f"Synced {len(self._entities)} entities from GPU")
    
    def _gpu_table_to_dict(self, heap_data: bytes, table_ptr: int) -> Dict[str, Any]:
        """Convert GPU heap table to Python dict."""
        def read_uint(offset: int) -> int:
            return struct.unpack_from('<I', heap_data, table_ptr + offset * 4)[0]
        
        capacity = read_uint(self.TABLE_OFF_CAPACITY)
        count = read_uint(self.TABLE_OFF_COUNT)
        
        result = {}
        
        # Read each entry
        for i in range(capacity):
            entry_offset = self.TABLE_OFF_ENTRIES + i * self.ENTRY_SIZE_WORDS
            
            # Read key XValue
            key_type = read_uint(entry_offset + 0)
            key_flags = read_uint(entry_offset + 1)
            key_data = read_uint(entry_offset + 2)
            
            # Skip nil keys (empty slots)
            if key_type == 0:  # TYPE_NIL
                continue
            
            # Read value XValue
            val_type = read_uint(entry_offset + 3)
            val_flags = read_uint(entry_offset + 4)
            val_data = read_uint(entry_offset + 5)
            
            # Convert key to Python
            py_key = self._xvalue_parts_to_python(key_type, key_data, heap_data)
            
            # Convert value to Python
            py_val = self._xvalue_parts_to_python(val_type, val_data, heap_data)
            
            if py_key is not None:
                result[py_key] = py_val
        
        return result
    
    def _xvalue_parts_to_python(self, type_id: int, data: int, 
                                 heap_data: bytes) -> Any:
        """Convert XValue parts to Python value."""
        if type_id == 0:  # TYPE_NIL
            return None
        elif type_id == 1:  # TYPE_BOOL
            return data != 0
        elif type_id == 2:  # TYPE_NUMBER
            # Unpack float from uint32
            float_bytes = struct.pack('I', data)
            return struct.unpack('f', float_bytes)[0]
        elif type_id == 3:  # TYPE_STRING
            # Lookup string by index
            if data < len(self._strings):
                return self._strings[data]
            return f"string_{data}"
        elif type_id == 4:  # TYPE_TABLE
            # Recursively read nested table
            return self._gpu_table_to_dict(heap_data, data)
        else:
            return None
    
    def _process_spawn_buffer(self) -> None:
        """Process spawn requests from GPU spawn buffer."""
        # Read spawn count
        spawn_count = self._gpu_spawn_count.to_numpy()[0]
        
        if spawn_count == 0:
            return
        
        # Read heap for table data
        heap_array = self._gpu_heap.to_numpy()
        heap_data = heap_array.tobytes()
        
        # Read spawn buffer (SpawnRequest = tablePtr, sourceEntityId, sourceThreadId, status)
        spawn_buffer = self._gpu_spawn_buffer.to_numpy()
        
        for i in range(spawn_count):
            base = i * 4
            table_ptr = spawn_buffer[base]
            status = spawn_buffer[base + 3]
            
            # Only process pending spawns
            if status == 0:  # SPAWN_STATUS_PENDING
                # Read table data from heap
                table_dict = self._gpu_table_to_dict(heap_data, table_ptr)
                
                # Spawn new entity
                self.spawn(table_dict)
        
        # Reset spawn count
        self._gpu_spawn_count.copy_from_numpy(np.array([0], dtype=np.uint32))
        
        if self.debug:
            print(f"Processed {spawn_count} spawn requests")
    
    def call(self, func_name: str, *args) -> Any:
        """
        Call a script function by name.
        
        Args:
            func_name: Name of the function to call
            *args: Arguments to pass to the function
            
        Returns:
            Return value of the function
        """
        func = self.get_global(func_name)
        if func.is_nil():
            raise NameError(f"Function '{func_name}' not defined")
        
        if func.type != TYPE_FUNCTION:
            raise TypeError(f"'{func_name}' is not a function")
        
        # Convert args to XValues
        xargs = [XValue.from_python(arg) for arg in args]
        
        # TODO: Implement function calling
        # For now, return nil
        return None
    
    def call_batch(self, func_name: str, 
                   data: np.ndarray, 
                   **kwargs) -> np.ndarray:
        """
        Call a script function on batched data.
        
        This is optimized for GPU execution where the function
        is called in parallel on multiple data items.
        
        Args:
            func_name: Name of the function to call
            data: NumPy array of input data
            **kwargs: Additional arguments (broadcast to all calls)
            
        Returns:
            NumPy array of results
        """
        # TODO: Implement batch calling
        raise NotImplementedError("Batch calling not yet implemented")
    
    def get_global(self, name: str) -> XValue:
        """
        Get a global variable value.
        
        Args:
            name: Variable name
            
        Returns:
            Value (or nil if not defined)
        """
        return self._globals.get(name, XValue.nil())
    
    def set_global(self, name: str, value: Any) -> None:
        """
        Set a global variable.
        
        Args:
            name: Variable name
            value: Value to set (will be converted to XValue)
        """
        self._globals[name] = XValue.from_python(value)
    
    def register(self, name: str) -> Callable:
        """
        Decorator to register a Python function as a host function.
        
        Example:
            @ctx.register("my_func")
            def my_func(a, b):
                return a + b
        
        Args:
            name: Name to use in scripts
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            self.register_function(name, func)
            return func
        return decorator
    
    def register_function(self, name: str, func: Callable) -> None:
        """
        Register a Python function as a host function.
        
        Args:
            name: Name to use in scripts
            func: Python function to register
        """
        import inspect
        sig = inspect.signature(func)
        arity = len(sig.parameters)
        
        xfunc = XFunction(
            name=name,
            arity=arity,
            is_host=True,
            host_func=func
        )
        
        self._host_functions[name] = xfunc
        self._host_function_list.append(xfunc)
        
        # Also add to globals
        self._globals[name] = XValue(TYPE_FUNCTION, xfunc)
    
    def get_host_function(self, index: int) -> Optional[XFunction]:
        """Get host function by index."""
        if 0 <= index < len(self._host_function_list):
            return self._host_function_list[index]
        return None
    
    def get_host_function_by_name(self, name: str) -> Optional[XFunction]:
        """Get host function by name."""
        return self._host_functions.get(name)
    
    def intern_string(self, s: str) -> int:
        """
        Intern a string, returning its index.
        
        Args:
            s: String to intern
            
        Returns:
            String pool index
        """
        if s in self._strings:
            return self._strings.index(s)
        
        idx = len(self._strings)
        self._strings.append(s)
        return idx
    
    def get_string(self, index: int) -> str:
        """
        Get a string by index.
        
        Args:
            index: String pool index
            
        Returns:
            The string
        """
        if 0 <= index < len(self._strings):
            return self._strings[index]
        return ""
    
    def to_python(self, value: XValue) -> Any:
        """
        Convert an XValue to Python.
        
        Args:
            value: XValue to convert
            
        Returns:
            Python value
        """
        return value.to_python()
    
    def from_python(self, value: Any) -> XValue:
        """
        Convert a Python value to XValue.
        
        Args:
            value: Python value
            
        Returns:
            XValue
        """
        return XValue.from_python(value)
    
    # =========================================================================
    # ECS Methods
    # =========================================================================
    
    def spawn(self, data: Dict[str, Any]) -> int:
        """
        Spawn a new entity with the given component data.
        
        Args:
            data: Dictionary of component data
            
        Returns:
            Entity ID
        """
        # Create entity table from data
        table = XTable.from_dict(data)
        
        # Allocate entity ID
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        
        # Store entity
        self._entities[entity_id] = table
        self._entity_generation[entity_id] = 0
        
        return entity_id
    
    def get_entity(self, entity_id: int) -> Optional[Dict[str, Any]]:
        """
        Get entity data by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity data as dictionary, or None if not found
        """
        table = self._entities.get(entity_id)
        if table is None:
            return None
        return table.to_dict()
    
    def destroy(self, entity_id: int) -> None:
        """
        Destroy an entity.
        
        Args:
            entity_id: Entity ID to destroy
        """
        if entity_id in self._entities:
            del self._entities[entity_id]
            # Increment generation to invalidate old handles
            self._entity_generation[entity_id] = self._entity_generation.get(entity_id, 0) + 1
    
    def entity_count(self) -> int:
        """
        Get the number of active entities.
        
        Returns:
            Number of entities
        """
        return len(self._entities)
    
    def filter(self, *keys: str) -> Filter:
        """
        Create an entity filter for dispatch operations.
        
        Args:
            *keys: Component keys to filter by
            
        Returns:
            Filter object
        """
        return Filter(keys=keys, _context=self)
    
    def dispatch(self, script: Script, func_name: str, 
                 entity_filter: Filter, **kwargs) -> DispatchStats:
        """
        Dispatch a system function to all matching entities.
        
        Args:
            script: Compiled script containing the function
            func_name: Name of the function to call
            entity_filter: Filter specifying which entities to process
            **kwargs: Additional arguments (e.g., dt=0.016)
            
        Returns:
            DispatchStats with execution statistics
        """
        dt = kwargs.get('dt', 0.0)
        
        # Use GPU dispatch if device is cuda
        if self.device == "cuda":
            return self._dispatch_gpu(script, func_name, entity_filter, dt)
        
        # CPU dispatch path
        return self._dispatch_cpu(script, func_name, entity_filter, dt)
    
    def _dispatch_cpu(self, script: Script, func_name: str,
                      entity_filter: Filter, dt: float) -> DispatchStats:
        """Dispatch system function on CPU."""
        # Execute the script first to define functions
        self.execute(script)
        
        # Get the function from globals
        func = self.get_global(func_name)
        if func.is_nil():
            raise NameError(f"Function '{func_name}' not defined")
        
        if func.type != TYPE_FUNCTION:
            raise TypeError(f"'{func_name}' is not a function")
        
        stats = DispatchStats()
        
        # Process each entity
        for entity_id, table in list(self._entities.items()):
            entity_dict = table.to_dict()
            
            # Check filter
            if not entity_filter.matches(entity_dict):
                stats.skipped += 1
                continue
            
            # Execute function on entity
            try:
                self._execute_system_on_entity(script, func_name, table, dt)
                stats.processed += 1
            except Exception as e:
                stats.errors += 1
                if self.debug:
                    print(f"Error processing entity {entity_id}: {e}")
        
        return stats
    
    def _dispatch_gpu(self, script: Script, func_name: str,
                      entity_filter: Filter, dt: float) -> DispatchStats:
        """Dispatch system function on GPU."""
        import slangpy as spy
        
        # Initialize GPU if needed
        if not self._initialized:
            self._initialize_gpu()
        
        # Execute script to register functions
        self._execute_cpu(script)
        
        # Get function index
        func_index = self._get_function_index(func_name, script.bytecode)
        
        # Upload bytecode if changed
        self._upload_bytecode(script.bytecode)
        
        # Sync entities to GPU
        self._sync_entities_to_gpu()
        
        # Build filtered entity list
        entity_ids = self._build_filtered_entity_list(entity_filter)
        entity_count = len(entity_ids) if entity_ids[0] != 0 or len(self._entities) > 0 else 0
        
        if entity_count == 0:
            return DispatchStats(processed=0, skipped=len(self._entities))
        
        # Upload entity list
        padded_ids = np.zeros(MAX_ENTITIES, dtype=np.uint32)
        padded_ids[:len(entity_ids)] = entity_ids
        self._gpu_dispatch_entity_list.copy_from_numpy(padded_ids)
        
        # Build dispatch config
        # DispatchConfig: functionIndex(u32), entityCount(u32), requiredKeyCount(u32),
        #                 requiredKeys[8](8*u32), dt(f32), flags(u32), padding[2](2*u32)
        config = np.zeros(15, dtype=np.uint32)
        config[0] = func_index
        config[1] = entity_count
        config[2] = len(entity_filter.keys)
        
        # Required keys (string indices)
        filter_keys = self._intern_filter_keys(entity_filter)
        config[3:11] = filter_keys
        
        # dt as float
        dt_bytes = struct.pack('f', dt)
        config[11] = struct.unpack('I', dt_bytes)[0]
        config[12] = 0  # flags
        config[13] = 0  # padding
        config[14] = 0  # padding
        
        self._gpu_dispatch_config.copy_from_numpy(config)
        
        # Reset dispatch state
        self._gpu_dispatch_state.copy_from_numpy(np.zeros(8, dtype=np.uint32))
        
        # Reset spawn count
        self._gpu_spawn_count.copy_from_numpy(np.array([0], dtype=np.uint32))
        
        # Initialize dispatch
        self._dispatch_init_kernel.dispatch(
            thread_count=spy.uint3(1, 1, 1),
            vars={
                "g_dispatchConfig": self._gpu_dispatch_config,
                "g_dispatchState": self._gpu_dispatch_state,
            }
        )
        
        # Run main dispatch kernel
        thread_groups = (entity_count + DISPATCH_THREAD_GROUP_SIZE - 1) // DISPATCH_THREAD_GROUP_SIZE
        
        self._dispatch_kernel.dispatch(
            thread_count=spy.uint3(thread_groups * DISPATCH_THREAD_GROUP_SIZE, 1, 1),
            vars={
                "g_dispatchConfig": self._gpu_dispatch_config,
                "g_dispatchState": self._gpu_dispatch_state,
                "g_dispatchEntityList": self._gpu_dispatch_entity_list,
                "g_dispatchResults": self._gpu_dispatch_results,
                "g_bytecode": self._gpu_bytecode,
                "g_constants": self._gpu_constants,
                "g_globals": self._gpu_globals,
                "g_functions": self._gpu_functions,
                "g_vmStates": self._gpu_vm_states,
                "g_callStacks": self._gpu_call_stacks,
                "g_callDepths": self._gpu_call_depths,
                "g_heapMemory": self._gpu_heap,
                "g_heapState": self._gpu_heap_state,
                "g_entityPool": self._gpu_entity_pool,
                "g_entityPoolState": self._gpu_entity_pool_state,
                "g_spawnBuffer": self._gpu_spawn_buffer,
                "g_spawnCount": self._gpu_spawn_count,
            }
        )
        
        # Finalize dispatch
        self._dispatch_finalize_kernel.dispatch(
            thread_count=spy.uint3(1, 1, 1),
            vars={
                "g_dispatchState": self._gpu_dispatch_state,
            }
        )
        
        # Read dispatch state
        state = self._gpu_dispatch_state.to_numpy()
        
        # Sync entities back from GPU
        self._sync_entities_from_gpu()
        
        # Process any spawn requests
        self._process_spawn_buffer()
        
        if self.debug:
            print(f"GPU dispatch: processed={state[0]}, skipped={state[1]}, errors={state[2]}")
        
        return DispatchStats(
            processed=int(state[0]),
            skipped=int(state[1]),
            errors=int(state[2]),
            spawned=int(state[3]),
            destroyed=int(state[4])
        )
    
    def _execute_system_on_entity(self, script: Script, func_name: str, 
                                   entity: XTable, dt: float) -> None:
        """Execute a system function on a single entity."""
        from .interpreter import Interpreter
        
        # Create interpreter with entity context
        interp = Interpreter(self)
        interp.set_current_entity(entity)
        interp.bytecode = script.bytecode
        
        # Get the function - it should be in globals after compile
        func = self.get_global(func_name)
        if func.is_nil() or func.type != TYPE_FUNCTION:
            raise NameError(f"Function '{func_name}' not defined")
        
        func_obj = func.data
        
        # Call the function with entity and dt as arguments
        # The function modifies entity in-place
        entity_val = entity.to_xvalue()
        dt_val = XValue.number(dt)
        
        if func_obj.is_host:
            # Host function - call directly
            func_obj.host_func(entity.to_dict(), dt)
        else:
            # Script function - set up call and execute
            interp.stack = [entity_val, dt_val]
            interp.pc = func_obj.code_offset
            
            # Execute function body
            try:
                while interp.pc < len(script.bytecode.code):
                    result = interp.step()
                    if result is not None:
                        break
            except Exception as e:
                if self.debug:
                    print(f"Error in system function: {e}")


# Convenience functions
def create_context(**kwargs) -> Context:
    """Create a new XScript context."""
    return Context(**kwargs)


def run(source: str, **kwargs) -> Any:
    """
    Compile and run XScript code.
    
    Args:
        source: XScript source code
        **kwargs: Context options
        
    Returns:
        Result value
    """
    ctx = Context(**kwargs)
    script = ctx.compile(source)
    return ctx.execute(script).to_python()

