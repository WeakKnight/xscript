# XScript 编程语言设计提案

> XScript - 基于GPU的游戏脚本语言

## 概述

XScript 是一门面向游戏脚本的编程语言，核心创新在于将虚拟机运行在GPU上（通过Slang/HLSL），同时保持Lua般的灵活性和C风格的熟悉语法。

### 核心特性

- **基于Slang的Stack VM**: 虚拟机运行在GPU compute shader中
- **动态类型系统**: 类似Lua的灵活类型
- **Meta Table机制**: 支持运算符重载和元方法
- **引用计数GC**: GPU友好的确定性内存管理
- **C风格语法**: 熟悉且易于上手
- **Python API**: 通过SlangPy提供Host侧接口

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Layer (Python)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Script    │  │   XScript   │  │     Bytecode        │  │
│  │   Context   │──│   Compiler  │──│     Generator       │  │
│  │     API     │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Device Layer (Slang/GPU)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Stack-based│  │  Reference  │  │    Meta Table       │  │
│  │     VM      │──│  Counting   │──│      System         │  │
│  │             │  │     GC      │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                         │                                    │
│                         ▼                                    │
│              ┌─────────────────────┐                        │
│              │  Dynamic Value      │                        │
│              │      System         │                        │
│              └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 核心架构

### 1.1 Stack VM 设计

VM基于Slang实现，运行在GPU compute shader中。采用基于栈的虚拟机架构，类似于Lua VM但针对GPU并行执行优化。

#### 核心数据结构

```slang
// 值类型 - 所有XScript值的统一表示
struct XValue {
    uint type;           // 类型标签: nil, bool, number, string, table, function
    uint refCount;       // 引用计数 (仅堆分配类型使用)
    uint64_t data;       // 值数据或堆指针
};

// VM状态 - 每个执行上下文的状态
struct VMState {
    XValue stack[256];   // 操作数栈
    uint sp;             // 栈指针 (Stack Pointer)
    uint fp;             // 帧指针 (Frame Pointer)
    uint pc;             // 程序计数器 (Program Counter)
    uint status;         // 执行状态: running, paused, error, completed
};

// 调用帧 - 函数调用信息
struct CallFrame {
    uint returnPC;       // 返回地址
    uint prevFP;         // 上一帧指针
    uint localBase;      // 局部变量基址
    uint argCount;       // 参数数量
};
```

### 1.2 指令集设计

基于Lua字节码简化设计，适配GPU执行特性：

#### 栈操作指令

| 操作码 | 编码 | 描述 | 操作数 | 栈效果 |
|--------|------|------|--------|--------|
| NOP | 0x00 | 空操作 | - | 0 |
| PUSH_NIL | 0x01 | 压入nil值 | - | +1 |
| PUSH_TRUE | 0x02 | 压入true | - | +1 |
| PUSH_FALSE | 0x03 | 压入false | - | +1 |
| PUSH_NUM | 0x04 | 压入数字常量 | idx:u16 | +1 |
| PUSH_STR | 0x05 | 压入字符串常量 | idx:u16 | +1 |
| POP | 0x06 | 弹出栈顶 | - | -1 |
| DUP | 0x07 | 复制栈顶 | - | +1 |
| SWAP | 0x08 | 交换栈顶两元素 | - | 0 |

#### 局部变量指令

| 操作码 | 编码 | 描述 | 操作数 | 栈效果 |
|--------|------|------|--------|--------|
| GET_LOCAL | 0x10 | 读取局部变量 | slot:u8 | +1 |
| SET_LOCAL | 0x11 | 设置局部变量 | slot:u8 | -1 |
| GET_GLOBAL | 0x12 | 读取全局变量 | idx:u16 | +1 |
| SET_GLOBAL | 0x13 | 设置全局变量 | idx:u16 | -1 |
| GET_UPVALUE | 0x14 | 读取上值 | idx:u8 | +1 |
| SET_UPVALUE | 0x15 | 设置上值 | idx:u8 | -1 |

#### 算术运算指令

| 操作码 | 编码 | 描述 | 操作数 | 栈效果 |
|--------|------|------|--------|--------|
| ADD | 0x20 | 加法 | - | -1 |
| SUB | 0x21 | 减法 | - | -1 |
| MUL | 0x22 | 乘法 | - | -1 |
| DIV | 0x23 | 除法 | - | -1 |
| MOD | 0x24 | 取模 | - | -1 |
| NEG | 0x25 | 取负 | - | 0 |
| POW | 0x26 | 幂运算 | - | -1 |

#### 比较与逻辑指令

| 操作码 | 编码 | 描述 | 操作数 | 栈效果 |
|--------|------|------|--------|--------|
| EQ | 0x30 | 相等比较 | - | -1 |
| NE | 0x31 | 不等比较 | - | -1 |
| LT | 0x32 | 小于 | - | -1 |
| LE | 0x33 | 小于等于 | - | -1 |
| GT | 0x34 | 大于 | - | -1 |
| GE | 0x35 | 大于等于 | - | -1 |
| NOT | 0x36 | 逻辑非 | - | 0 |
| AND | 0x37 | 逻辑与 | - | -1 |
| OR | 0x38 | 逻辑或 | - | -1 |

#### 控制流指令

| 操作码 | 编码 | 描述 | 操作数 | 栈效果 |
|--------|------|------|--------|--------|
| JMP | 0x40 | 无条件跳转 | offset:i16 | 0 |
| JMP_IF | 0x41 | 条件为真跳转 | offset:i16 | -1 |
| JMP_IF_NOT | 0x42 | 条件为假跳转 | offset:i16 | -1 |
| LOOP | 0x43 | 循环跳转(向后) | offset:i16 | 0 |

#### 函数调用指令

| 操作码 | 编码 | 描述 | 操作数 | 栈效果 |
|--------|------|------|--------|--------|
| CALL | 0x50 | 函数调用 | argc:u8 | 变化 |
| RETURN | 0x51 | 函数返回 | retc:u8 | 变化 |
| CALL_HOST | 0x52 | 调用Host函数 | idx:u16, argc:u8 | 变化 |

#### 表操作指令

| 操作码 | 编码 | 描述 | 操作数 | 栈效果 |
|--------|------|------|--------|--------|
| NEW_TABLE | 0x60 | 创建新表 | capacity:u8 | +1 |
| GET_TABLE | 0x61 | 表索引读取 | - | -1 |
| SET_TABLE | 0x62 | 表索引写入 | - | -3 |
| GET_FIELD | 0x63 | 读取字段(常量key) | idx:u16 | 0 |
| SET_FIELD | 0x64 | 设置字段(常量key) | idx:u16 | -1 |

#### 元表指令

| 操作码 | 编码 | 描述 | 操作数 | 栈效果 |
|--------|------|------|--------|--------|
| GET_META | 0x70 | 获取元表 | - | 0 |
| SET_META | 0x71 | 设置元表 | - | -1 |
| INVOKE_META | 0x72 | 调用元方法 | meta:u8, argc:u8 | 变化 |

### 1.3 类型系统

动态类型系统，所有类型检查在运行时进行：

| 类型 | 类型ID | 存储方式 | 描述 |
|------|--------|----------|------|
| nil | 0 | 无数据 | 空值 |
| bool | 1 | data低1位 | 布尔值 |
| number | 2 | data作为float64位模式 | 数值(双精度浮点) |
| string | 3 | data为字符串池索引 | 字符串(不可变) |
| table | 4 | data为堆对象偏移 | 表(关联数组) |
| function | 5 | data为函数描述符索引 | 函数 |
| userdata | 6 | data为外部数据索引 | 用户数据 |
| thread | 7 | data为协程状态索引 | 协程(预留) |

### 1.4 Meta Table 机制

支持Lua风格的元表，实现运算符重载和自定义行为：

```slang
struct XTable {
    uint capacity;       // 哈希表容量
    uint count;          // 当前元素数
    uint metaTableRef;   // 元表引用 (0 = 无元表)
    uint entriesOffset;  // 指向 XTableEntry[] 的偏移
    uint arrayPart;      // 数组部分偏移 (优化整数索引)
    uint arraySize;      // 数组部分大小
};

struct XTableEntry {
    XValue key;
    XValue value;
    uint next;           // 哈希冲突链表
};
```

#### 元方法索引

| 元方法 | 索引 | 触发场景 |
|--------|------|----------|
| __add | 0 | a + b |
| __sub | 1 | a - b |
| __mul | 2 | a * b |
| __div | 3 | a / b |
| __mod | 4 | a % b |
| __pow | 5 | a ^ b |
| __neg | 6 | -a |
| __eq | 7 | a == b |
| __lt | 8 | a < b |
| __le | 9 | a <= b |
| __index | 10 | t[k] 读取不存在的键 |
| __newindex | 11 | t[k] = v 写入不存在的键 |
| __call | 12 | t(...) 将表作为函数调用 |
| __tostring | 13 | tostring(t) |
| __len | 14 | #t |
| __gc | 15 | 垃圾回收时调用 |

---

## 2. 内存管理与GC

### 2.1 引用计数策略

采用原子引用计数，适合GPU并行环境：

```slang
// 类型常量
static const uint TYPE_NIL = 0;
static const uint TYPE_BOOL = 1;
static const uint TYPE_NUMBER = 2;
static const uint TYPE_STRING = 3;
static const uint TYPE_TABLE = 4;
static const uint TYPE_FUNCTION = 5;
static const uint TYPE_USERDATA = 6;

// 引用计数操作
void xvalue_incref(inout XValue v) {
    // 只有堆分配类型需要引用计数
    if (v.type >= TYPE_STRING) {
        uint oldCount;
        InterlockedAdd(heap_refcount(v.data), 1, oldCount);
    }
}

void xvalue_decref(inout XValue v) {
    if (v.type >= TYPE_STRING) {
        uint oldCount;
        InterlockedAdd(heap_refcount(v.data), -1, oldCount);
        if (oldCount == 1) {
            // 引用计数归零，释放内存
            xvalue_free(v);
        }
    }
}

void xvalue_assign(inout XValue dst, XValue src) {
    if (dst.data != src.data || dst.type != src.type) {
        xvalue_incref(src);
        xvalue_decref(dst);
        dst = src;
    }
}
```

### 2.2 堆内存管理

GPU侧采用池分配器，预分配固定大小的内存池：

```slang
struct HeapBlock {
    uint size;           // 块大小
    uint refCount;       // 引用计数
    uint next;           // 空闲链表下一个
    uint flags;          // 标志位
    // data follows...
};

struct HeapAllocator {
    RWStructuredBuffer<uint> memory;
    uint freeListSmall;  // 小对象空闲链表 (<=64B)
    uint freeListMedium; // 中对象空闲链表 (<=256B)
    uint freeListLarge;  // 大对象空闲链表 (<=1KB)
    uint totalSize;
    uint usedSize;
};

// 分配内存
uint heap_alloc(uint size) {
    uint blockSize = align_up(size + sizeof(HeapBlock), 16);
    uint freeList = select_free_list(blockSize);
    
    uint block;
    InterlockedExchange(freeList, memory[freeList], block);
    
    if (block == 0) {
        // 空闲链表为空，从池尾分配
        InterlockedAdd(usedSize, blockSize, block);
    }
    
    memory[block] = blockSize;
    memory[block + 1] = 1;  // refCount = 1
    return block + sizeof(HeapBlock);
}

// 释放内存
void heap_free(uint ptr) {
    uint block = ptr - sizeof(HeapBlock);
    uint blockSize = memory[block];
    uint freeList = select_free_list(blockSize);
    
    uint oldHead;
    InterlockedExchange(freeList, block, oldHead);
    memory[block + 2] = oldHead;
}
```

### 2.3 字符串池

字符串采用池化存储，相同内容共享存储：

```slang
struct StringPool {
    RWStructuredBuffer<uint> data;      // 字符数据
    RWStructuredBuffer<uint> hashTable; // 哈希表
    uint capacity;
    uint count;
};

struct StringHeader {
    uint hash;           // 字符串哈希值
    uint length;         // 字符串长度
    uint refCount;       // 引用计数
    uint next;           // 哈希冲突链表
    // chars follow...
};
```

### 2.4 循环引用处理

对于游戏脚本场景，循环引用较少见。采用以下策略：

1. **弱引用**: 提供 `weak()` 函数创建弱引用
2. **手动打破**: 提供 `break_cycle()` 辅助函数
3. **作用域释放**: 函数返回时自动清理局部表

---

## 3. 语法设计

### 3.1 词法规范

#### 关键字

```
var func if else for while do break continue return
nil true false and or not
setmetatable getmetatable
```

#### 运算符

```
算术:  + - * / % ^
比较:  == != < <= > >=
逻辑:  and or not
赋值:  = += -= *= /= %=
其他:  . [] () {} , ; :
```

#### 字面量

```
数字:   123, 3.14, 0xFF, 1e10, 0b1010
字符串: "hello", 'world', `template ${expr}`
布尔:   true, false
空值:   nil
```

### 3.2 语法规范 (EBNF)

```ebnf
(* 程序结构 *)
program        = { statement } ;

(* 语句 *)
statement      = var_decl 
               | func_decl 
               | if_stmt 
               | for_stmt 
               | while_stmt 
               | do_while_stmt
               | break_stmt
               | continue_stmt
               | return_stmt 
               | expr_stmt 
               | block ;

(* 变量声明 *)
var_decl       = "var" IDENT [ "=" expr ] ";" ;

(* 函数声明 *)
func_decl      = "func" IDENT "(" [ param_list ] ")" block ;
param_list     = IDENT { "," IDENT } ;

(* 控制流语句 *)
if_stmt        = "if" "(" expr ")" block { "else" "if" "(" expr ")" block } [ "else" block ] ;
for_stmt       = "for" "(" [ for_init ] ";" [ expr ] ";" [ expr ] ")" block ;
for_init       = var_decl | expr ;
while_stmt     = "while" "(" expr ")" block ;
do_while_stmt  = "do" block "while" "(" expr ")" ";" ;
break_stmt     = "break" ";" ;
continue_stmt  = "continue" ";" ;
return_stmt    = "return" [ expr { "," expr } ] ";" ;

(* 块 *)
block          = "{" { statement } "}" ;

(* 表达式语句 *)
expr_stmt      = expr ";" ;

(* 表达式 *)
expr           = assignment ;
assignment     = [ call "." ] IDENT assign_op assignment
               | ternary ;
assign_op      = "=" | "+=" | "-=" | "*=" | "/=" | "%=" ;

ternary        = logic_or [ "?" expr ":" ternary ] ;

logic_or       = logic_and { "or" logic_and } ;
logic_and      = equality { "and" equality } ;

equality       = comparison { ( "==" | "!=" ) comparison } ;
comparison     = term { ( "<" | "<=" | ">" | ">=" ) term } ;

term           = factor { ( "+" | "-" ) factor } ;
factor         = power { ( "*" | "/" | "%" ) power } ;
power          = unary { "^" unary } ;

unary          = ( "-" | "not" | "#" ) unary | call ;

call           = primary { "(" [ arg_list ] ")" | "." IDENT | "[" expr "]" } ;
arg_list       = expr { "," expr } ;

primary        = NUMBER 
               | STRING 
               | "true" | "false" | "nil"
               | IDENT 
               | "(" expr ")"
               | table_literal
               | func_expr ;

(* 表字面量 *)
table_literal  = "{" [ table_fields ] "}" ;
table_fields   = table_field { ( "," | ";" ) table_field } [ "," | ";" ] ;
table_field    = "[" expr "]" ":" expr
               | IDENT ":" expr
               | expr ;

(* 匿名函数 *)
func_expr      = "func" "(" [ param_list ] ")" block ;

(* 词法元素 *)
IDENT          = ALPHA { ALPHA | DIGIT | "_" } ;
NUMBER         = DIGIT { DIGIT } [ "." DIGIT { DIGIT } ] [ ( "e" | "E" ) [ "+" | "-" ] DIGIT { DIGIT } ]
               | "0x" HEX { HEX }
               | "0b" BIN { BIN } ;
STRING         = '"' { CHAR } '"' | "'" { CHAR } "'" | "`" { CHAR | "${" expr "}" } "`" ;
ALPHA          = "a".."z" | "A".."Z" | "_" ;
DIGIT          = "0".."9" ;
HEX            = DIGIT | "a".."f" | "A".."F" ;
BIN            = "0" | "1" ;
```

### 3.3 语法示例

#### 基本示例

```c
// 变量声明
var x = 10;
var name = "player";
var pi = 3.14159;

// 表创建
var player = {
    name: "Hero",
    hp: 100,
    mp: 50,
    position: { x: 0, y: 0, z: 0 }
};

// 函数定义
func add(a, b) {
    return a + b;
}

// 匿名函数
var multiply = func(a, b) {
    return a * b;
};

// 控制流
if (x > 5) {
    print("large");
} else if (x > 0) {
    print("small");
} else {
    print("zero or negative");
}

// 循环
for (var i = 0; i < 10; i += 1) {
    print(i);
}

while (x > 0) {
    x = x - 1;
}

do {
    x = x + 1;
} while (x < 10);

// 表操作
player.name = "Warrior";
player["level"] = 10;
var hp = player.hp;
```

#### 元表示例

```c
// 创建向量类型
var Vector = {};

// 元表定义
var VectorMeta = {
    __add: func(a, b) {
        return Vector.new(a.x + b.x, a.y + b.y, a.z + b.z);
    },
    __sub: func(a, b) {
        return Vector.new(a.x - b.x, a.y - b.y, a.z - b.z);
    },
    __mul: func(a, b) {
        if (type(b) == "number") {
            return Vector.new(a.x * b, a.y * b, a.z * b);
        }
        return a.x * b.x + a.y * b.y + a.z * b.z;  // 点积
    },
    __tostring: func(v) {
        return "(" + v.x + ", " + v.y + ", " + v.z + ")";
    },
    __index: VectorMeta
};

// 构造函数
Vector.new = func(x, y, z) {
    var v = { x: x, y: y, z: z };
    setmetatable(v, VectorMeta);
    return v;
};

// 使用
var v1 = Vector.new(1, 2, 3);
var v2 = Vector.new(4, 5, 6);
var v3 = v1 + v2;  // 调用 __add
var dot = v1 * v2; // 调用 __mul
print(v3);         // 调用 __tostring
```

#### 游戏脚本示例

```c
// NPC 行为脚本
var NPC = {
    state: "idle",
    target: nil,
    speed: 5.0
};

func NPC.update(self, dt) {
    if (self.state == "idle") {
        self.idle_behavior(dt);
    } else if (self.state == "patrol") {
        self.patrol_behavior(dt);
    } else if (self.state == "chase") {
        self.chase_behavior(dt);
    }
}

func NPC.idle_behavior(self, dt) {
    // 检测玩家
    var player = find_player();
    if (player and distance(self.position, player.position) < 10) {
        self.target = player;
        self.state = "chase";
    }
}

func NPC.chase_behavior(self, dt) {
    if (self.target == nil) {
        self.state = "idle";
        return;
    }
    
    var dir = normalize(self.target.position - self.position);
    self.position = self.position + dir * self.speed * dt;
    
    if (distance(self.position, self.target.position) < 1) {
        attack(self.target);
    }
}
```

---

## 4. 编译管线

### 4.1 编译流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Source    │───▶│    Lexer    │───▶│   Tokens    │
│   (.xs)     │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Bytecode   │◀───│   Codegen   │◀───│   Parser    │
│   (.xsc)    │    │             │    │     AST     │
└─────────────┘    └─────────────┘    └─────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                  GPU VM (Slang)                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────────┐   │
│  │ Bytecode  │  │   VM      │  │    Result     │   │
│  │  Buffer   │─▶│ Execution │─▶│    Buffer     │   │
│  └───────────┘  └───────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 4.2 字节码格式

```
XScript Bytecode Format (.xsc)

Header (16 bytes):
┌────────────────────────────────────────┐
│ Magic: "XSC\0" (4 bytes)               │
│ Version: u16                           │
│ Flags: u16                             │
│ Constant Pool Offset: u32              │
│ Code Offset: u32                       │
└────────────────────────────────────────┘

Constant Pool:
┌────────────────────────────────────────┐
│ Count: u32                             │
│ Entry[0]: Type(u8) + Data(variable)    │
│ Entry[1]: ...                          │
│ ...                                    │
└────────────────────────────────────────┘

Function Table:
┌────────────────────────────────────────┐
│ Count: u32                             │
│ Func[0]: Name(u16) + PC(u32) + Args(u8)│
│ Func[1]: ...                           │
│ ...                                    │
└────────────────────────────────────────┘

Code Section:
┌────────────────────────────────────────┐
│ Size: u32                              │
│ Instructions: [Opcode + Operands]...   │
└────────────────────────────────────────┘
```

### 4.3 指令编码

每条指令1-4字节：

```
单字节指令:   [opcode:8]
双字节指令:   [opcode:8][operand:8]
三字节指令:   [opcode:8][operand:16]
四字节指令:   [opcode:8][operand:24] 或 [opcode:8][op1:8][op2:16]
```

---

## 5. Python API 设计

### 5.1 核心API

```python
import xscript as xs

# 创建脚本上下文
ctx = xs.Context(device="cuda")  # 或 "cpu" 用于调试

# 编译脚本
script = ctx.compile("""
    var counter = 0;
    func increment(n) {
        counter = counter + n;
        return counter;
    }
""")

# 或从文件加载
script = ctx.compile_file("game_logic.xs")

# 执行全局代码
ctx.execute(script)

# 调用脚本函数
result = ctx.call("increment", 5)
print(result)  # 5

result = ctx.call("increment", 3)
print(result)  # 8

# 获取/设置全局变量
counter = ctx.get_global("counter")
ctx.set_global("counter", 100)
```

### 5.2 Host函数注册

```python
# 使用装饰器注册Host函数
@ctx.register("spawn_enemy")
def spawn_enemy(enemy_type: str, x: float, y: float, z: float) -> int:
    """在指定位置生成敌人，返回敌人ID"""
    enemy = game.spawn(enemy_type, x, y, z)
    return enemy.id

@ctx.register("play_sound")
def play_sound(sound_name: str, volume: float = 1.0):
    """播放音效"""
    audio.play(sound_name, volume)

# 脚本中调用
# spawn_enemy("goblin", 10, 0, 20);
# play_sound("explosion", 0.8);
```

### 5.3 批量执行

```python
import numpy as np

# 批量处理NPC更新
npc_data = np.array([
    {"id": 1, "x": 10, "y": 20, "state": 0},
    {"id": 2, "x": 30, "y": 40, "state": 1},
    # ... 1000个NPC
], dtype=xs.npc_dtype)

# 批量调用update函数
results = ctx.call_batch("npc_update", npc_data, delta_time=0.016)

# 结果是NumPy数组
print(results["new_state"])
```

### 5.4 表与Python对象互转

```python
# Python字典 -> XScript表
py_dict = {
    "name": "Player1",
    "stats": {
        "hp": 100,
        "mp": 50,
        "level": 10
    },
    "inventory": [1, 2, 3, 4, 5]
}
xs_table = ctx.from_python(py_dict)
ctx.set_global("player", xs_table)

# XScript表 -> Python字典
xs_table = ctx.get_global("player")
py_dict = ctx.to_python(xs_table)
```

### 5.5 调试支持

```python
# 启用调试模式
ctx = xs.Context(device="cpu", debug=True)

# 设置断点
ctx.set_breakpoint("game_logic.xs", line=42)

# 注册调试回调
@ctx.on_breakpoint
def on_break(location, locals, globals):
    print(f"Break at {location}")
    print(f"Local variables: {locals}")
    
# 单步执行
ctx.step()

# 继续执行
ctx.continue_()
```

---

## 6. CPU/GPU 执行模型

### 6.1 执行流程

```
Python Host                         GPU Device
     │                                   │
     │  1. compile("script.xs")          │
     ├──────────────────────────────────▶│
     │                                   │
     │  2. upload(bytecode, globals)     │
     ├──────────────────────────────────▶│
     │                                   │
     │  3. execute(entry_point)          │
     ├──────────────────────────────────▶│
     │                                   │
     │        ┌──────────────────────┐   │
     │        │ VM Interpretation    │   │
     │        │ - fetch instruction  │   │
     │        │ - decode             │   │
     │        │ - execute            │   │
     │        └──────────────────────┘   │
     │                                   │
     │◀─────────── host_call_request ────│
     │                                   │
     │  4. handle host call              │
     │  5. return result                 │
     ├──────────────────────────────────▶│
     │                                   │
     │◀──────────── execution done ──────│
     │                                   │
     │  6. read results                  │
     │◀──────────────────────────────────│
     ▼                                   ▼
```

### 6.2 并行执行模型

每个GPU线程运行一个独立的VM实例，适合批量处理：

```slang
[shader("compute")]
[numthreads(64, 1, 1)]
void vm_execute(uint3 threadId : SV_DispatchThreadID)
{
    uint vmIndex = threadId.x;
    
    if (vmIndex >= vmCount) return;
    
    // 每个线程有独立的VM状态
    VMState state = vmStates[vmIndex];
    
    // 执行直到完成或需要Host调用
    while (state.status == VM_RUNNING) {
        uint opcode = bytecode[state.pc++];
        
        switch (opcode) {
            case OP_PUSH_NUM:
                // ... 执行指令
                break;
            case OP_CALL_HOST:
                // 标记需要Host处理
                hostCallRequests[vmIndex] = make_request(...);
                state.status = VM_WAITING_HOST;
                break;
            // ...
        }
    }
    
    vmStates[vmIndex] = state;
}
```

### 6.3 Host回调机制

```python
# Host侧处理回调
def execute_with_callbacks(ctx, script, entry):
    ctx.upload(script)
    
    while True:
        # 执行一批
        ctx.dispatch()
        
        # 检查Host调用请求
        requests = ctx.get_host_call_requests()
        if len(requests) == 0:
            break
            
        # 处理每个请求
        results = []
        for req in requests:
            func = ctx.host_functions[req.func_id]
            result = func(*req.args)
            results.append(result)
        
        # 返回结果给GPU
        ctx.set_host_call_results(results)
    
    return ctx.get_results()
```

---

## 7. 项目结构

```
xscript/
├── compiler/                 # Python编译器
│   ├── __init__.py
│   ├── lexer.py             # 词法分析器
│   ├── tokens.py            # Token定义
│   ├── parser.py            # 语法分析器
│   ├── ast.py               # AST节点定义
│   ├── codegen.py           # 字节码生成
│   ├── bytecode.py          # 字节码格式
│   └── errors.py            # 编译错误
│
├── runtime/                  # Slang VM运行时
│   ├── vm.slang             # 核心VM实现
│   ├── value.slang          # 值类型系统
│   ├── table.slang          # 表和元表
│   ├── string.slang         # 字符串池
│   ├── gc.slang             # 引用计数GC
│   ├── heap.slang           # 堆内存管理
│   ├── builtins.slang       # 内置函数
│   └── ops.slang            # 指令实现
│
├── api/                      # Python API (SlangPy集成)
│   ├── __init__.py
│   ├── context.py           # 脚本上下文
│   ├── types.py             # 类型转换
│   ├── bindings.py          # SlangPy绑定
│   └── debug.py             # 调试支持
│
├── stdlib/                   # 标准库
│   ├── math.xs              # 数学函数
│   ├── string.xs            # 字符串函数
│   ├── table.xs             # 表操作
│   └── io.xs                # I/O函数
│
├── tests/                    # 测试
│   ├── test_lexer.py
│   ├── test_parser.py
│   ├── test_codegen.py
│   ├── test_vm.py
│   └── scripts/             # 测试脚本
│
├── examples/                 # 示例
│   ├── hello_world.xs
│   ├── game_npc.xs
│   └── vector_math.xs
│
├── docs/                     # 文档
│   ├── language_guide.md
│   ├── api_reference.md
│   └── tutorials/
│
├── setup.py
├── requirements.txt
└── README.md
```

---

## 8. 实现里程碑

| 阶段 | 目标 | 关键交付 | 预计时间 |
|------|------|----------|----------|
| M1 | 核心VM原型 | 基本指令集、栈操作、数值运算 | 2周 |
| M2 | 类型系统 | nil/bool/number/string/table实现 | 2周 |
| M3 | Meta Table | 元方法调用、运算符重载 | 2周 |
| M4 | 编译器前端 | Lexer + Parser + AST | 2周 |
| M5 | 代码生成 | 字节码生成、常量池 | 1周 |
| M6 | Python API | Context、执行、数据交互 | 2周 |
| M7 | GC完善 | 引用计数优化、内存池 | 1周 |
| M8 | 标准库 | math/string/table函数 | 1周 |
| M9 | 优化 | 性能调优、批量执行 | 2周 |
| M10 | 文档与示例 | 语言指南、API文档、教程 | 1周 |

---

## 9. 技术挑战与对策

| 挑战 | 描述 | 对策 |
|------|------|------|
| GPU动态内存分配 | GPU不支持传统malloc | 预分配内存池 + 池分配器 |
| 字符串处理 | GPU不擅长变长数据 | 字符串池 + 哈希索引 + 长度前缀 |
| 控制流 | GPU偏好统一控制流 | 展开为线性字节码 + 跳转，避免深度分支 |
| Host回调延迟 | GPU-CPU同步开销大 | 批量收集回调请求，减少同步次数 |
| 调试困难 | GPU调试工具有限 | CPU fallback模式用于开发调试 |
| 循环引用 | 引用计数无法处理 | 弱引用 + 手动打破 + 作用域释放 |
| 并发安全 | 多线程访问共享数据 | 原子操作 + 每线程独立VM状态 |

---

## 附录 A: 内置函数

| 函数 | 描述 | 示例 |
|------|------|------|
| print(...) | 打印值 | print("hello", 123) |
| type(v) | 返回类型名 | type(123) == "number" |
| tostring(v) | 转换为字符串 | tostring(123) |
| tonumber(s) | 转换为数字 | tonumber("123") |
| setmetatable(t, mt) | 设置元表 | setmetatable(obj, Meta) |
| getmetatable(t) | 获取元表 | getmetatable(obj) |
| pairs(t) | 表迭代器 | for (k, v in pairs(t)) |
| ipairs(t) | 数组迭代器 | for (i, v in ipairs(arr)) |
| len(v) | 获取长度 | len("hello") == 5 |
| error(msg) | 抛出错误 | error("invalid argument") |
| pcall(f, ...) | 保护调用 | pcall(risky_func, arg) |

## 附录 B: 标准库

### math

| 函数 | 描述 |
|------|------|
| math.abs(x) | 绝对值 |
| math.floor(x) | 向下取整 |
| math.ceil(x) | 向上取整 |
| math.round(x) | 四舍五入 |
| math.sin(x) | 正弦 |
| math.cos(x) | 余弦 |
| math.tan(x) | 正切 |
| math.sqrt(x) | 平方根 |
| math.pow(x, y) | 幂运算 |
| math.min(...) | 最小值 |
| math.max(...) | 最大值 |
| math.random() | 随机数 [0,1) |
| math.PI | 圆周率常量 |

### string

| 函数 | 描述 |
|------|------|
| string.len(s) | 字符串长度 |
| string.sub(s, i, j) | 子字符串 |
| string.upper(s) | 转大写 |
| string.lower(s) | 转小写 |
| string.find(s, pattern) | 查找模式 |
| string.format(fmt, ...) | 格式化 |
| string.split(s, sep) | 分割 |
| string.join(arr, sep) | 连接 |

### table

| 函数 | 描述 |
|------|------|
| table.insert(t, v) | 插入元素 |
| table.remove(t, i) | 移除元素 |
| table.sort(t, cmp) | 排序 |
| table.concat(t, sep) | 连接为字符串 |
| table.keys(t) | 获取所有键 |
| table.values(t) | 获取所有值 |
| table.clone(t) | 浅复制 |
| table.merge(t1, t2) | 合并表 |

