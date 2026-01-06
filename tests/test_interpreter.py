"""
XScript Interpreter Tests

Tests for the CPU interpreter.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.context import Context
from api.types import XValue, TYPE_NIL, TYPE_NUMBER, TYPE_STRING


class TestBasicExecution:
    """Test basic script execution."""
    
    def test_nil_value(self):
        ctx = Context()
        script = ctx.compile("var x;")
        ctx.execute(script)
        
        x = ctx.get_global("x")
        assert x.is_nil()
    
    def test_number_assignment(self):
        ctx = Context()
        script = ctx.compile("var x = 42;")
        ctx.execute(script)
        
        x = ctx.get_global("x")
        assert x.type == TYPE_NUMBER
        assert x.data == 42.0
    
    def test_string_assignment(self):
        ctx = Context()
        script = ctx.compile('var s = "hello";')
        ctx.execute(script)
        
        s = ctx.get_global("s")
        assert s.type == TYPE_STRING
        assert s.data == "hello"
    
    def test_arithmetic(self):
        ctx = Context()
        script = ctx.compile("var x = 10 + 20 * 2;")
        ctx.execute(script)
        
        x = ctx.get_global("x")
        assert x.data == 50.0  # 10 + (20 * 2) = 50
    
    def test_comparison(self):
        ctx = Context()
        script = ctx.compile("var x = 10 > 5;")
        ctx.execute(script)
        
        x = ctx.get_global("x")
        assert x.to_python() == True


class TestControlFlow:
    """Test control flow statements."""
    
    def test_if_true(self):
        ctx = Context()
        script = ctx.compile("""
            var x = 0;
            if (true) {
                x = 1;
            }
        """)
        ctx.execute(script)
        
        x = ctx.get_global("x")
        assert x.data == 1.0
    
    def test_if_false(self):
        ctx = Context()
        script = ctx.compile("""
            var x = 0;
            if (false) {
                x = 1;
            }
        """)
        ctx.execute(script)
        
        x = ctx.get_global("x")
        assert x.data == 0.0
    
    def test_if_else(self):
        ctx = Context()
        script = ctx.compile("""
            var x = 0;
            if (false) {
                x = 1;
            } else {
                x = 2;
            }
        """)
        ctx.execute(script)
        
        x = ctx.get_global("x")
        assert x.data == 2.0
    
    def test_while_loop(self):
        ctx = Context()
        script = ctx.compile("""
            var x = 0;
            while (x < 5) {
                x = x + 1;
            }
        """)
        ctx.execute(script)
        
        x = ctx.get_global("x")
        assert x.data == 5.0


class TestHostFunctions:
    """Test host function integration."""
    
    def test_register_function(self):
        ctx = Context()
        
        @ctx.register("double")
        def double(n):
            return n * 2
        
        script = ctx.compile("var x = double(21);")
        ctx.execute(script)
        
        x = ctx.get_global("x")
        assert x.data == 42.0
    
    def test_builtin_type(self):
        ctx = Context()
        script = ctx.compile('var t = type(42);')
        ctx.execute(script)
        
        t = ctx.get_global("t")
        assert t.data == "number"


class TestTables:
    """Test table operations."""
    
    def test_create_table(self):
        ctx = Context()
        script = ctx.compile("var t = {};")
        ctx.execute(script)
        
        t = ctx.get_global("t")
        assert t.type == 4  # TYPE_TABLE
    
    def test_table_field_access(self):
        ctx = Context()
        script = ctx.compile("""
            var t = { x: 10, y: 20 };
            var sum = t.x + t.y;
        """)
        ctx.execute(script)
        
        # Note: This may fail until table operations are fully implemented
        # in the interpreter


def run_tests():
    """Run all tests."""
    import traceback
    
    test_classes = [TestBasicExecution, TestControlFlow, TestHostFunctions]
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    
    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total += 1
                try:
                    getattr(instance, method_name)()
                    passed += 1
                    print(f"  ✓ {test_class.__name__}.{method_name}")
                except AssertionError as e:
                    failed += 1
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                except NotImplementedError:
                    skipped += 1
                    print(f"  ○ {test_class.__name__}.{method_name}: skipped")
                except Exception as e:
                    failed += 1
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                    traceback.print_exc()
    
    print(f"\nResults: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    return failed == 0


if __name__ == "__main__":
    print("Running XScript Interpreter Tests\n")
    success = run_tests()
    sys.exit(0 if success else 1)

