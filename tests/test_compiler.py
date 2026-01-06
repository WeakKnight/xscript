"""
XScript Compiler Tests

Basic tests for the lexer, parser, and code generator.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler.lexer import Lexer
from compiler.tokens import TokenType
from compiler.parser import Parser
from compiler.codegen import CodeGenerator
from compiler.bytecode import OpCode


class TestLexer:
    """Test the lexer."""
    
    def test_empty_source(self):
        lexer = Lexer("")
        tokens = lexer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_numbers(self):
        lexer = Lexer("123 45.67 0xFF 0b1010")
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 123.0
        
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == 45.67
        
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == 255.0  # 0xFF
        
        assert tokens[3].type == TokenType.NUMBER
        assert tokens[3].value == 10.0  # 0b1010
    
    def test_strings(self):
        lexer = Lexer('"hello" \'world\'')
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"
        
        assert tokens[1].type == TokenType.STRING
        assert tokens[1].value == "world"
    
    def test_keywords(self):
        lexer = Lexer("var func if else while for return true false nil")
        tokens = lexer.tokenize()
        
        expected = [
            TokenType.VAR, TokenType.FUNC, TokenType.IF, TokenType.ELSE,
            TokenType.WHILE, TokenType.FOR, TokenType.RETURN,
            TokenType.TRUE, TokenType.FALSE, TokenType.NIL, TokenType.EOF
        ]
        
        for token, expected_type in zip(tokens, expected):
            assert token.type == expected_type
    
    def test_operators(self):
        lexer = Lexer("+ - * / % ^ == != < <= > >= = +=")
        tokens = lexer.tokenize()
        
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.PERCENT, TokenType.CARET, TokenType.EQ, TokenType.NE,
            TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE,
            TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.EOF
        ]
        
        for token, expected_type in zip(tokens, expected):
            assert token.type == expected_type
    
    def test_comments(self):
        lexer = Lexer("""
            // Line comment
            var x = 10; // Inline comment
            /* Block
               comment */
            var y = 20;
        """)
        tokens = lexer.tokenize()
        
        # Should only have: var, x, =, 10, ;, var, y, =, 20, ;, EOF
        identifiers = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert len(identifiers) == 2
        assert identifiers[0].lexeme == "x"
        assert identifiers[1].lexeme == "y"


class TestParser:
    """Test the parser."""
    
    def test_var_declaration(self):
        tokens = Lexer("var x = 10;").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        assert len(ast.statements) == 1
        stmt = ast.statements[0]
        assert stmt.name.lexeme == "x"
        assert stmt.initializer is not None
    
    def test_function_declaration(self):
        tokens = Lexer("func add(a, b) { return a + b; }").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        assert len(ast.statements) == 1
        func = ast.statements[0]
        assert func.name.lexeme == "add"
        assert len(func.params) == 2
    
    def test_if_statement(self):
        tokens = Lexer("if (x > 0) { print(x); }").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        assert len(ast.statements) == 1
    
    def test_for_loop(self):
        tokens = Lexer("for (var i = 0; i < 10; i += 1) { print(i); }").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        assert len(ast.statements) == 1
    
    def test_table_literal(self):
        tokens = Lexer("var t = { x: 1, y: 2 };").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        assert len(ast.statements) == 1
    
    def test_binary_expression(self):
        tokens = Lexer("var x = 1 + 2 * 3;").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        assert len(ast.statements) == 1


class TestCodegen:
    """Test the code generator."""
    
    def test_simple_expression(self):
        tokens = Lexer("var x = 10;").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        codegen = CodeGenerator()
        bytecode = codegen.generate(ast)
        
        assert len(bytecode.code) > 0
        assert len(bytecode.constants) > 0
    
    def test_arithmetic(self):
        tokens = Lexer("var x = 1 + 2;").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        codegen = CodeGenerator()
        bytecode = codegen.generate(ast)
        
        # Should contain ADD opcode
        assert OpCode.ADD in bytecode.code
    
    def test_function_call(self):
        tokens = Lexer("print(42);").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        codegen = CodeGenerator()
        bytecode = codegen.generate(ast)
        
        # Should contain CALL opcode
        assert OpCode.CALL in bytecode.code
    
    def test_disassemble(self):
        tokens = Lexer("var x = 10; var y = x + 5;").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        codegen = CodeGenerator()
        bytecode = codegen.generate(ast)
        
        disasm = bytecode.disassemble()
        assert "XScript Bytecode" in disasm
        assert "Constants" in disasm
        assert "Code" in disasm


class TestBytecode:
    """Test bytecode serialization."""
    
    def test_serialize_deserialize(self):
        tokens = Lexer("var x = 10; var y = 20; var z = x + y;").tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        
        codegen = CodeGenerator()
        original = codegen.generate(ast)
        
        # Serialize
        data = original.serialize()
        
        # Deserialize
        from compiler.bytecode import Bytecode
        restored = Bytecode.deserialize(data)
        
        # Compare
        assert len(restored.code) == len(original.code)
        assert len(restored.constants) == len(original.constants)
        assert restored.code == original.code


def run_tests():
    """Run all tests."""
    import traceback
    
    test_classes = [TestLexer, TestParser, TestCodegen, TestBytecode]
    total = 0
    passed = 0
    failed = 0
    
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
                except Exception as e:
                    failed += 1
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                    traceback.print_exc()
    
    print(f"\nResults: {passed}/{total} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("Running XScript Compiler Tests\n")
    success = run_tests()
    sys.exit(0 if success else 1)

