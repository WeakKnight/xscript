"""
XScript Context

The main interface for compiling and executing XScript code.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .types import XValue, XTable, XFunction, TYPE_FUNCTION

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
        
        # GPU buffers (will be initialized on first use)
        self._initialized = False
        self._slang_module = None
        
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
        return interp.run(script.bytecode)
    
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
        # TODO: Load Slang module and create buffers
        self._initialized = True
    
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

