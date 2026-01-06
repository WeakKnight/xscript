"""
XScript Type Wrappers (32-bit version)

Python wrappers for XScript runtime types.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import numpy as np
import struct


# Type constants matching runtime/value.slang
TYPE_NIL = 0
TYPE_BOOL = 1
TYPE_NUMBER = 2
TYPE_STRING = 3
TYPE_TABLE = 4
TYPE_FUNCTION = 5
TYPE_USERDATA = 6
TYPE_THREAD = 7


@dataclass
class XValue:
    """
    Python representation of an XScript value (32-bit version).
    
    This class wraps XScript values for Python interop.
    Uses 32-bit float for numbers and 32-bit uint for data storage.
    """
    
    type: int
    data: Any
    
    @classmethod
    def nil(cls) -> 'XValue':
        """Create a nil value."""
        return cls(TYPE_NIL, None)
    
    @classmethod
    def boolean(cls, value: bool) -> 'XValue':
        """Create a boolean value."""
        return cls(TYPE_BOOL, value)
    
    @classmethod
    def number(cls, value: float) -> 'XValue':
        """Create a number value (32-bit float)."""
        # Clamp to float32 range
        return cls(TYPE_NUMBER, np.float32(value))
    
    @classmethod
    def string(cls, value: str) -> 'XValue':
        """Create a string value."""
        return cls(TYPE_STRING, value)
    
    @classmethod
    def from_python(cls, value: Any) -> 'XValue':
        """Convert a Python value to XValue."""
        if value is None:
            return cls.nil()
        elif isinstance(value, bool):
            return cls.boolean(value)
        elif isinstance(value, (int, float)):
            return cls.number(value)
        elif isinstance(value, str):
            return cls.string(value)
        elif isinstance(value, dict):
            return XTable.from_dict(value).to_xvalue()
        elif isinstance(value, (list, tuple)):
            return XTable.from_list(value).to_xvalue()
        elif isinstance(value, XValue):
            return value
        else:
            raise TypeError(f"Cannot convert {type(value)} to XValue")
    
    def to_python(self) -> Any:
        """Convert XValue to Python value."""
        if self.type == TYPE_NIL:
            return None
        elif self.type == TYPE_BOOL:
            return bool(self.data)
        elif self.type == TYPE_NUMBER:
            return float(self.data)
        elif self.type == TYPE_STRING:
            return str(self.data)
        elif self.type == TYPE_TABLE:
            if isinstance(self.data, XTable):
                return self.data.to_dict()
            return self.data
        elif self.type == TYPE_FUNCTION:
            return self.data
        else:
            return self.data
    
    def is_nil(self) -> bool:
        """Check if value is nil."""
        return self.type == TYPE_NIL
    
    def is_truthy(self) -> bool:
        """Check if value is truthy (not nil and not false)."""
        if self.type == TYPE_NIL:
            return False
        if self.type == TYPE_BOOL:
            return bool(self.data)
        return True
    
    def __repr__(self) -> str:
        type_names = ['nil', 'bool', 'number', 'string', 'table', 'function', 'userdata', 'thread']
        type_name = type_names[self.type] if self.type < len(type_names) else f'type{self.type}'
        return f"XValue({type_name}, {self.data!r})"
    
    @staticmethod
    def numpy_dtype() -> np.dtype:
        """Get the numpy dtype for XValue (32-bit version)."""
        return np.dtype([
            ('type', np.uint32),
            ('flags', np.uint32),
            ('data', np.uint32),
        ])


class XTable:
    """
    Python representation of an XScript table.
    
    Tables are the primary data structure in XScript,
    functioning as both arrays and dictionaries.
    """
    
    def __init__(self):
        self._entries: Dict[Any, XValue] = {}
        self._metatable: Optional['XTable'] = None
        self._array_part: List[XValue] = []
    
    @classmethod
    def from_dict(cls, d: Dict[Any, Any]) -> 'XTable':
        """Create a table from a Python dictionary."""
        table = cls()
        for key, value in d.items():
            table.set(key, XValue.from_python(value))
        return table
    
    @classmethod
    def from_list(cls, lst: List[Any]) -> 'XTable':
        """Create a table from a Python list (1-indexed)."""
        table = cls()
        for i, value in enumerate(lst):
            table.set(i, XValue.from_python(value))
        return table
    
    def get(self, key: Any) -> XValue:
        """Get a value from the table."""
        # Try array part first for integer keys
        if isinstance(key, int) and 0 <= key < len(self._array_part):
            return self._array_part[key]
        
        # Try hash part
        if key in self._entries:
            return self._entries[key]
        
        # Try metatable __index
        if self._metatable is not None:
            index_method = self._metatable.get('__index')
            if not index_method.is_nil():
                if index_method.type == TYPE_TABLE:
                    return index_method.data.get(key)
                # If it's a function, would need to call it
        
        return XValue.nil()
    
    def set(self, key: Any, value: XValue) -> None:
        """Set a value in the table."""
        # Use array part for sequential integer keys
        if isinstance(key, int) and key >= 0:
            while len(self._array_part) <= key:
                self._array_part.append(XValue.nil())
            self._array_part[key] = value
        else:
            self._entries[key] = value
    
    def get_metatable(self) -> Optional['XTable']:
        """Get the metatable."""
        return self._metatable
    
    def set_metatable(self, mt: Optional['XTable']) -> None:
        """Set the metatable."""
        self._metatable = mt
    
    def to_dict(self) -> Dict[Any, Any]:
        """Convert table to Python dictionary."""
        result = {}
        
        # Add array part
        for i, value in enumerate(self._array_part):
            if not value.is_nil():
                result[i] = value.to_python()
        
        # Add hash part
        for key, value in self._entries.items():
            result[key] = value.to_python()
        
        return result
    
    def to_list(self) -> List[Any]:
        """Convert array part to Python list."""
        return [v.to_python() for v in self._array_part]
    
    def to_xvalue(self) -> XValue:
        """Wrap table in XValue."""
        return XValue(TYPE_TABLE, self)
    
    def __len__(self) -> int:
        """Get the length of the array part."""
        return len(self._array_part)
    
    def __getitem__(self, key: Any) -> Any:
        """Get item (Python style)."""
        return self.get(key).to_python()
    
    def __setitem__(self, key: Any, value: Any) -> None:
        """Set item (Python style)."""
        self.set(key, XValue.from_python(value))
    
    def __repr__(self) -> str:
        return f"XTable({self.to_dict()!r})"


class XFunction:
    """
    Python representation of an XScript function.
    
    Can be either a script function or a host (Python) function.
    """
    
    def __init__(self, 
                 name: str,
                 arity: int,
                 is_host: bool = False,
                 host_func: Optional[Callable] = None,
                 code_offset: int = 0):
        self.name = name
        self.arity = arity
        self.is_host = is_host
        self.host_func = host_func
        self.code_offset = code_offset
    
    def __call__(self, *args) -> Any:
        """Call the function (only works for host functions)."""
        if self.is_host and self.host_func:
            return self.host_func(*args)
        raise RuntimeError("Cannot directly call script functions from Python")
    
    def __repr__(self) -> str:
        if self.is_host:
            return f"XFunction(host:{self.name}, arity={self.arity})"
        return f"XFunction({self.name}, arity={self.arity}, offset={self.code_offset})"


def float32_to_uint32(f: float) -> int:
    """Convert float32 to its bit pattern as uint32."""
    return struct.unpack('I', struct.pack('f', f))[0]


def uint32_to_float32(u: int) -> float:
    """Convert uint32 bit pattern to float32."""
    return struct.unpack('f', struct.pack('I', u))[0]


def numpy_to_xvalue_buffer(values: List[XValue]) -> np.ndarray:
    """Convert list of XValues to numpy structured array (32-bit version)."""
    dtype = XValue.numpy_dtype()
    arr = np.zeros(len(values), dtype=dtype)
    
    for i, v in enumerate(values):
        arr[i]['type'] = v.type
        arr[i]['flags'] = 0
        
        if v.type == TYPE_NIL:
            arr[i]['data'] = 0
        elif v.type == TYPE_BOOL:
            arr[i]['data'] = 1 if v.data else 0
        elif v.type == TYPE_NUMBER:
            # Pack float32 as uint32 bit pattern
            arr[i]['data'] = float32_to_uint32(float(v.data))
        elif v.type == TYPE_STRING:
            # String index
            arr[i]['data'] = v.data if isinstance(v.data, int) else 0
        else:
            # Pointer/index types
            arr[i]['data'] = v.data if isinstance(v.data, int) else 0
    
    return arr


def xvalue_buffer_to_list(arr: np.ndarray) -> List[XValue]:
    """Convert numpy structured array back to list of XValues (32-bit version)."""
    values = []
    
    for i in range(len(arr)):
        type_id = int(arr[i]['type'])
        data = int(arr[i]['data'])
        
        if type_id == TYPE_NIL:
            values.append(XValue.nil())
        elif type_id == TYPE_BOOL:
            values.append(XValue.boolean(data != 0))
        elif type_id == TYPE_NUMBER:
            # Unpack float32 from uint32 bit pattern
            value = uint32_to_float32(data)
            values.append(XValue.number(value))
        elif type_id == TYPE_STRING:
            values.append(XValue(TYPE_STRING, data))
        else:
            values.append(XValue(type_id, data))
    
    return values
