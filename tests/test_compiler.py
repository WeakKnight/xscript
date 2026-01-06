"""
XScript Compiler Tests

Tests for the XScript compiler: lexer, parser, and code generator.
"""

import pytest
from compiler import compile_source, Lexer, Parser, CodeGenerator, Bytecode, OpCode
from compiler.tokens import Token, TokenType
from compiler.errors import CompileError, SyntaxError


# =============================================================================
# Lexer Tests
# =============================================================================

class TestLexerBasics:
    """Basic lexer functionality tests."""
    
    def test_empty_source(self):
        lexer = Lexer("")
        tokens = lexer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_whitespace_only(self):
        lexer = Lexer("   \t\n  ")
        tokens = lexer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF


class TestLexerNumbers:
    """Number literal tokenization tests."""
    
    def test_integer(self):
        lexer = Lexer("42")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 42.0
    
    def test_float(self):
        lexer = Lexer("3.14")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 3.14
    
    def test_hex(self):
        lexer = Lexer("0xFF")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 255.0
    
    def test_binary(self):
        lexer = Lexer("0b101")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 5.0
    
    def test_exponent(self):
        lexer = Lexer("1e10")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == 1e10


class TestLexerStrings:
    """String literal tokenization tests."""
    
    def test_double_quote_string(self):
        lexer = Lexer('"hello"')
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"
    
    def test_single_quote_string(self):
        lexer = Lexer("'hello'")
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"
    
    def test_escape_sequences(self):
        lexer = Lexer(r'"hello\nworld\t!"')
        tokens = lexer.tokenize()
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello\nworld\t!"


class TestLexerKeywords:
    """Keyword tokenization tests."""
    
    @pytest.mark.parametrize("keyword,expected_type", [
        ("var", TokenType.VAR),
        ("func", TokenType.FUNC),
        ("if", TokenType.IF),
        ("else", TokenType.ELSE),
        ("for", TokenType.FOR),
        ("while", TokenType.WHILE),
        ("return", TokenType.RETURN),
        ("nil", TokenType.NIL),
        ("true", TokenType.TRUE),
        ("false", TokenType.FALSE),
        ("and", TokenType.AND),
        ("or", TokenType.OR),
        ("not", TokenType.NOT),
    ])
    def test_keywords(self, keyword, expected_type):
        lexer = Lexer(keyword)
        tokens = lexer.tokenize()
        assert tokens[0].type == expected_type


class TestLexerOperators:
    """Operator tokenization tests."""
    
    @pytest.mark.parametrize("op,expected_type", [
        ("+", TokenType.PLUS),
        ("-", TokenType.MINUS),
        ("*", TokenType.STAR),
        ("/", TokenType.SLASH),
        ("%", TokenType.PERCENT),
        ("^", TokenType.CARET),
        ("==", TokenType.EQ),
        ("!=", TokenType.NE),
        ("<", TokenType.LT),
        ("<=", TokenType.LE),
        (">", TokenType.GT),
        (">=", TokenType.GE),
        ("=", TokenType.ASSIGN),
        ("+=", TokenType.PLUS_ASSIGN),
        ("-=", TokenType.MINUS_ASSIGN),
    ])
    def test_operators(self, op, expected_type):
        lexer = Lexer(op)
        tokens = lexer.tokenize()
        assert tokens[0].type == expected_type


class TestLexerComments:
    """Comment handling tests."""
    
    def test_line_comment(self):
        lexer = Lexer("42 // this is a comment")
        tokens = lexer.tokenize()
        assert len(tokens) == 2  # NUMBER, EOF
        assert tokens[0].type == TokenType.NUMBER
    
    def test_block_comment(self):
        lexer = Lexer("42 /* block */ 10")
        tokens = lexer.tokenize()
        assert len(tokens) == 3  # NUMBER, NUMBER, EOF
        assert tokens[0].value == 42.0
        assert tokens[1].value == 10.0


# =============================================================================
# Parser Tests
# =============================================================================

class TestParserExpressions:
    """Expression parsing tests."""
    
    def test_literal_number(self):
        bytecode = compile_source("42;")
        assert OpCode.PUSH_NUM in bytecode.code
    
    def test_literal_string(self):
        bytecode = compile_source('"hello";')
        assert OpCode.PUSH_STR in bytecode.code
    
    def test_literal_true(self):
        bytecode = compile_source("true;")
        assert OpCode.PUSH_TRUE in bytecode.code
    
    def test_literal_false(self):
        bytecode = compile_source("false;")
        assert OpCode.PUSH_FALSE in bytecode.code
    
    def test_literal_nil(self):
        bytecode = compile_source("nil;")
        assert OpCode.PUSH_NIL in bytecode.code
    
    def test_binary_add(self):
        bytecode = compile_source("1 + 2;")
        assert OpCode.ADD in bytecode.code
    
    def test_binary_sub(self):
        bytecode = compile_source("3 - 1;")
        assert OpCode.SUB in bytecode.code
    
    def test_binary_mul(self):
        bytecode = compile_source("2 * 3;")
        assert OpCode.MUL in bytecode.code
    
    def test_binary_div(self):
        bytecode = compile_source("6 / 2;")
        assert OpCode.DIV in bytecode.code
    
    def test_unary_neg(self):
        bytecode = compile_source("-42;")
        assert OpCode.NEG in bytecode.code
    
    def test_comparison_eq(self):
        bytecode = compile_source("1 == 1;")
        assert OpCode.EQ in bytecode.code
    
    def test_comparison_lt(self):
        bytecode = compile_source("1 < 2;")
        assert OpCode.LT in bytecode.code


class TestParserStatements:
    """Statement parsing tests."""
    
    def test_var_declaration(self):
        bytecode = compile_source("var x = 10;")
        assert OpCode.PUSH_NUM in bytecode.code
        # Top-level vars are globals
        assert OpCode.SET_GLOBAL in bytecode.code
    
    def test_if_statement(self):
        bytecode = compile_source("if (true) { 1; }")
        assert OpCode.JMP_IF_NOT in bytecode.code
    
    def test_while_statement(self):
        bytecode = compile_source("while (true) { 1; }")
        assert OpCode.JMP_IF_NOT in bytecode.code
        assert OpCode.LOOP in bytecode.code
    
    def test_function_declaration(self):
        bytecode = compile_source("func f() { return 1; }")
        assert OpCode.RETURN in bytecode.code


# =============================================================================
# ECS Codegen Tests
# =============================================================================

class TestECSCodegen:
    """ECS builtin function code generation tests."""
    
    def test_spawn_generates_opcode(self):
        """Test that spawn() generates SPAWN_ENTITY opcode."""
        bytecode = compile_source("spawn({});")
        assert OpCode.SPAWN_ENTITY in bytecode.code
    
    def test_destroy_generates_opcode(self):
        """Test that destroy() generates DESTROY_ENTITY opcode."""
        bytecode = compile_source("var e = 1; destroy(e);")
        assert OpCode.DESTROY_ENTITY in bytecode.code
    
    def test_get_entity_generates_opcode(self):
        """Test that get_entity() generates GET_ENTITY opcode."""
        bytecode = compile_source("var e = get_entity();")
        assert OpCode.GET_ENTITY in bytecode.code
    
    def test_get_entity_id_generates_opcode(self):
        """Test that get_entity_id() generates GET_ENTITY_ID opcode."""
        bytecode = compile_source("var id = get_entity_id();")
        assert OpCode.GET_ENTITY_ID in bytecode.code
    
    def test_has_component_generates_opcode(self):
        """Test that has_component() generates HAS_COMPONENT opcode."""
        bytecode = compile_source('var e = get_entity(); has_component(e, "health");')
        assert OpCode.HAS_COMPONENT in bytecode.code
    
    def test_add_component_generates_opcode(self):
        """Test that add_component() generates ADD_COMPONENT opcode."""
        bytecode = compile_source('var e = get_entity(); add_component(e, "health", 100);')
        assert OpCode.ADD_COMPONENT in bytecode.code
    
    def test_remove_component_generates_opcode(self):
        """Test that remove_component() generates REMOVE_COMPONENT opcode."""
        bytecode = compile_source('var e = get_entity(); remove_component(e, "health");')
        assert OpCode.REMOVE_COMPONENT in bytecode.code
    
    def test_spawn_with_table(self):
        """Test spawn with table literal argument."""
        bytecode = compile_source("var e = spawn({x: 0, y: 0, health: 100});")
        assert OpCode.NEW_TABLE in bytecode.code
        assert OpCode.SPAWN_ENTITY in bytecode.code
        # Top-level vars are globals
        assert OpCode.SET_GLOBAL in bytecode.code
    
    def test_ecs_in_if_statement(self):
        """Test ECS operations in control flow."""
        source = '''
        var e = get_entity();
        if (has_component(e, "health")) {
            remove_component(e, "health");
        }
        '''
        bytecode = compile_source(source)
        assert OpCode.GET_ENTITY in bytecode.code
        assert OpCode.HAS_COMPONENT in bytecode.code
        assert OpCode.REMOVE_COMPONENT in bytecode.code
        assert OpCode.JMP_IF_NOT in bytecode.code


class TestECSCodegenErrors:
    """ECS builtin error handling tests."""
    
    def test_spawn_wrong_arg_count(self):
        """Test that spawn() with wrong arg count raises error."""
        with pytest.raises(CompileError):
            compile_source("spawn();")  # Missing argument
    
    def test_spawn_too_many_args(self):
        """Test that spawn() with too many args raises error."""
        with pytest.raises(CompileError):
            compile_source("spawn({}, 1);")  # Extra argument
    
    def test_destroy_wrong_arg_count(self):
        """Test that destroy() with wrong arg count raises error."""
        with pytest.raises(CompileError):
            compile_source("destroy();")  # Missing argument
    
    def test_get_entity_with_args(self):
        """Test that get_entity() with args raises error."""
        with pytest.raises(CompileError):
            compile_source("get_entity(1);")  # Unexpected argument
    
    def test_has_component_wrong_args(self):
        """Test that has_component() with wrong arg count raises error."""
        with pytest.raises(CompileError):
            compile_source("has_component(e);")  # Missing key argument
    
    def test_add_component_wrong_args(self):
        """Test that add_component() with wrong arg count raises error."""
        with pytest.raises(CompileError):
            compile_source('add_component(e, "key");')  # Missing value argument
    
    def test_remove_component_wrong_args(self):
        """Test that remove_component() with wrong arg count raises error."""
        with pytest.raises(CompileError):
            compile_source("remove_component(e);")  # Missing key argument


# =============================================================================
# Integration Tests
# =============================================================================

class TestCompileSource:
    """End-to-end compilation tests."""
    
    def test_simple_arithmetic(self):
        """Test compiling simple arithmetic."""
        bytecode = compile_source("var x = 10 + 20;")
        assert bytecode is not None
        assert len(bytecode.code) > 0
    
    def test_function_definition(self):
        """Test compiling function definition."""
        source = '''
        func add(a, b) {
            return a + b;
        }
        '''
        bytecode = compile_source(source)
        assert len(bytecode.functions) > 0
    
    def test_control_flow(self):
        """Test compiling control flow."""
        source = '''
        var x = 0;
        if (x < 10) {
            x = x + 1;
        }
        '''
        bytecode = compile_source(source)
        assert OpCode.JMP_IF_NOT in bytecode.code
    
    def test_table_literal(self):
        """Test compiling table literal."""
        bytecode = compile_source("var t = {x: 10, y: 20};")
        assert OpCode.NEW_TABLE in bytecode.code
        # Table literals use SET_TABLE for key-value pairs
        assert OpCode.SET_TABLE in bytecode.code


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

