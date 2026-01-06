"""
XScript - GPU-Accelerated Scripting Language

XScript is a scripting language designed for game development,
featuring a GPU-based virtual machine implemented in Slang.

Example:
    import xscript as xs
    
    ctx = xs.Context()
    script = ctx.compile('''
        var x = 10;
        var y = 20;
        return x + y;
    ''')
    result = ctx.execute(script)
    print(result)  # 30
"""

from api.context import Context, Script, create_context, run
from api.types import XValue, XTable, XFunction
from compiler import compile_source, compile_file, Bytecode

__version__ = "0.1.0"
__author__ = "XScript Team"

__all__ = [
    # Main API
    'Context',
    'Script',
    'create_context',
    'run',
    
    # Types
    'XValue',
    'XTable',
    'XFunction',
    
    # Compiler
    'compile_source',
    'compile_file',
    'Bytecode',
]


def version() -> str:
    """Get XScript version string."""
    return __version__

