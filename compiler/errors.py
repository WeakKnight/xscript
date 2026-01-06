"""
XScript Compiler Errors

Defines exception classes for compilation errors.
"""

from typing import Optional


class XScriptError(Exception):
    """Base exception for all XScript errors."""
    
    def __init__(self, message: str, line: Optional[int] = None, 
                 column: Optional[int] = None, filename: Optional[str] = None):
        self.message = message
        self.line = line
        self.column = column
        self.filename = filename
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with location information."""
        parts = []
        
        if self.filename:
            parts.append(self.filename)
        
        if self.line is not None:
            if parts:
                parts.append(f":{self.line}")
            else:
                parts.append(f"line {self.line}")
            
            if self.column is not None:
                parts.append(f":{self.column}")
        
        if parts:
            return f"{':'.join(parts)}: {self.message}"
        return self.message


class SyntaxError(XScriptError):
    """Raised for syntax errors during lexing or parsing."""
    pass


class CompileError(XScriptError):
    """Raised for semantic errors during compilation."""
    pass


class RuntimeError(XScriptError):
    """Raised for errors during script execution."""
    pass


class TypeError(CompileError):
    """Raised for type-related errors."""
    pass


class NameError(CompileError):
    """Raised for undefined name errors."""
    pass


class ArgumentError(CompileError):
    """Raised for function argument errors."""
    pass

