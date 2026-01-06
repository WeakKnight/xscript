# XScript AI Agent Development Guide

本文档为AI辅助开发提供项目上下文和开发规范。

## 项目概述

XScript是一门面向游戏脚本的编程语言，核心特点：
- **Stack VM架构**: 基于栈的虚拟机，运行在GPU上(Slang/HLSL)
- **32位值系统**: 所有值使用32位存储 (float32, uint32)
- **动态类型**: 类似Lua的类型系统 (nil, bool, number, string, table, function)
- **Meta Table**: 支持元表和元方法，实现运算符重载
- **引用计数GC**: 基于引用计数的垃圾回收
- **C风格语法**: 熟悉的C/JavaScript风格语法

## 目录结构

```
xscript/
├── runtime/           # Slang GPU运行时
│   ├── value.slang    # XValue 32位值类型
│   ├── heap.slang     # 堆内存分配器
│   ├── table.slang    # 哈希表 + 元表
│   ├── string.slang   # 字符串池
│   ├── gc.slang       # 引用计数GC
│   ├── ops.slang      # 操作码实现
│   ├── vm.slang       # VM主循环
│   └── tests/         # Slang单元测试
│       ├── test_value.slang
│       ├── test_arithmetic.slang
│       ├── test_stack.slang
│       └── test_vm.slang
│
├── compiler/          # Python编译器
│   ├── tokens.py      # Token定义
│   ├── lexer.py       # 词法分析器
│   ├── ast.py         # AST节点定义
│   ├── parser.py      # 语法解析器
│   ├── bytecode.py    # 字节码定义
│   ├── codegen.py     # 代码生成器
│   └── errors.py      # 错误处理
│
├── api/               # Python API
│   ├── types.py       # XValue/XTable Python包装
│   ├── context.py     # 脚本执行上下文
│   └── interpreter.py # CPU解释器(调试用)
│
├── tests/             # Python测试
│   ├── test_compiler.py
│   ├── test_interpreter.py
│   └── test_vm_standalone.py  # VM CPU模拟测试
│
└── examples/          # 示例脚本
    ├── hello_world.xs
    ├── game_npc.xs
    └── vector_math.xs
```

## 核心数据结构

### XValue (32位)

```slang
struct XValue {
    uint type;    // 类型标签 (0-7)
    uint flags;   // 标志位 (GC标记等)
    uint data;    // 32位数据 (值或指针)
};
```

类型存储方式:
| 类型 | type值 | data存储 |
|------|--------|----------|
| nil | 0 | 0 |
| bool | 1 | 0或1 |
| number | 2 | float32位模式 (asuint/asfloat) |
| string | 3 | 字符串池索引 |
| table | 4 | 堆偏移量 |
| function | 5 | 函数索引 |

### 操作码 (Opcodes)

主要操作码分组:
- `0x00-0x0F`: 栈操作 (NOP, PUSH_NIL, PUSH_TRUE, POP, DUP, SWAP)
- `0x10-0x1F`: 变量 (GET_LOCAL, SET_LOCAL, GET_GLOBAL, SET_GLOBAL)
- `0x20-0x2F`: 算术 (ADD, SUB, MUL, DIV, MOD, NEG, POW)
- `0x30-0x3F`: 比较 (EQ, NE, LT, LE, GT, GE, NOT)
- `0x40-0x4F`: 控制流 (JMP, JMP_IF, JMP_IF_NOT, LOOP)
- `0x50-0x5F`: 函数 (CALL, RETURN, CALL_HOST)
- `0x60-0x6F`: 表操作 (NEW_TABLE, GET_TABLE, SET_TABLE)
- `0x70-0x7F`: 元表 (GET_META, SET_META)
- `0xFF`: HALT

## 开发规范

### Slang代码规范

1. **文件组织**: 每个模块一个文件，使用 `// =====` 分隔区块
2. **命名规范**:
   - 常量: `TYPE_NIL`, `OP_ADD` (大写下划线)
   - 结构体: `XValue`, `VMState` (PascalCase)
   - 函数: `xvalue_add`, `vm_push` (小写下划线)
3. **注释**: 使用 `//` 单行注释，重要函数前加说明
4. **32位约束**: 所有数值使用 `uint` 和 `float`，不使用64位类型

### Python代码规范

1. **类型注解**: 所有函数参数和返回值使用类型注解
2. **Docstrings**: 公开API使用Google风格docstring
3. **测试**: 每个模块对应 `test_*.py` 文件

### 测试规范

1. **Slang测试**: 
   - 使用 `RWStructuredBuffer<TestResult>` 收集结果
   - 每个测试有唯一 `testId`
   - 提供 `run_all_*_tests` 入口点

2. **Python测试**:
   - 使用pytest框架
   - 测试类名: `Test*`
   - 测试方法名: `test_*`

## 常见任务

### 添加新操作码

1. 在 `runtime/ops.slang` 添加常量定义
2. 在 `runtime/vm.slang` 的 `vm_step` 中添加case
3. 在 `compiler/bytecode.py` 的 `OpCode` 枚举中添加
4. 在 `compiler/codegen.py` 中生成对应字节码
5. 在 `api/interpreter.py` 中添加CPU实现
6. 添加测试用例

### 添加新的内置函数

1. 在 `api/context.py` 的 `_register_builtins` 中添加
2. 使用 `@self.register("name")` 装饰器
3. 添加测试用例

### 添加新的元方法

1. 在 `runtime/table.slang` 中添加 `META_*` 常量
2. 在 `runtime/ops.slang` 中处理元方法调用
3. 在Python端的 `XTable` 类中添加支持

## 调试技巧

### CPU模拟器调试

运行VM独立测试:
```bash
python -m tests.test_vm_standalone
```

### 查看生成的字节码

```python
from compiler.codegen import CodeGenerator
from compiler.parser import Parser
from compiler.lexer import Lexer

source = "var x = 10 + 20;"
lexer = Lexer(source)
parser = Parser(lexer.tokenize())
ast = parser.parse()
codegen = CodeGenerator()
bytecode = codegen.generate(ast)
print(bytecode.disassemble())
```

### 运行完整测试套件

```bash
python -m pytest tests/ -v
```

## 待实现功能

- [ ] 闭包和upvalue支持
- [ ] 字符串连接操作
- [ ] 表迭代器 (pairs/ipairs)
- [ ] 协程支持
- [ ] SlangPy GPU集成
- [ ] 更完整的标准库

## 注意事项

1. **32位浮点精度**: 使用float32，大数值可能有精度损失
2. **GPU内存**: 堆内存固定大小，需要预分配
3. **引用计数**: 注意循环引用问题
4. **字节序**: 字节码使用小端序

## 相关资源

- Slang语言: https://shader-slang.org/
- SlangPy: https://github.com/shader-slang/slangpy
- Lua参考手册: https://www.lua.org/manual/5.4/

