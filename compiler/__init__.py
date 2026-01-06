"""
XScript Compiler Package

A Python-based compiler for the XScript scripting language.
Compiles XScript source code to bytecode for execution on the GPU VM.
"""

from .tokens import Token, TokenType
from .lexer import Lexer
from .ast import *
from .parser import Parser
from .bytecode import Bytecode, OpCode
from .codegen import CodeGenerator
from .errors import XScriptError, CompileError, SyntaxError

__version__ = "0.1.0"
__all__ = [
    "Token",
    "TokenType", 
    "Lexer",
    "Parser",
    "Bytecode",
    "OpCode",
    "CodeGenerator",
    "XScriptError",
    "CompileError",
    "SyntaxError",
]


def compile_source(source: str) -> Bytecode:
    """
    Compile XScript source code to bytecode.
    
    Args:
        source: XScript source code string
        
    Returns:
        Bytecode object ready for VM execution
        
    Raises:
        CompileError: If compilation fails
    """
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    ast = parser.parse()
    
    codegen = CodeGenerator()
    bytecode = codegen.generate(ast)
    
    return bytecode


def compile_file(filepath: str) -> Bytecode:
    """
    Compile XScript source file to bytecode.
    
    Args:
        filepath: Path to .xs source file
        
    Returns:
        Bytecode object ready for VM execution
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    return compile_source(source)

