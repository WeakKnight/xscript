# XScript AI Agent Development Guide

Quick reference for AI-assisted development. See `docs/` for detailed documentation.

## Quick Links

| Document | Content |
|----------|---------|
| `docs/DesignProposal.md` | Architecture, XValue, VM, ECS, syntax |
| `docs/SIMTProgrammingModel.md` | GPU SIMT/ECS execution model |
| `docs/SlangGuide.md` | Slang language reference |
| `docs/SlangPyGuide.md` | SlangPy Python bindings |

## Project Structure

```
xscript/
├── runtime/          # Slang GPU Runtime (.slang files)
├── compiler/         # Python Compiler (lexer, parser, codegen)
├── api/              # Python API (context.py = main entry)
├── tests/            # Python tests (pytest)
├── docs/             # Documentation
├── demo.py           # Working GPU dispatch demo
└── examples/         # Example scripts (.xs)
```

## Key Files

| File | Purpose |
|------|---------|
| `api/context.py` | Main Context class, GPU dispatch, buffer management |
| `runtime/dispatch.slang` | GPU dispatch kernel |
| `runtime/ops.slang` | Opcodes + VMState definition |
| `runtime/vm.slang` | VM main loop |
| `tests/test_gpu_dispatch.py` | GPU dispatch integration tests |

## Common Tasks

### Add New Opcode

1. Add constant in `runtime/ops.slang`
2. Add case in `runtime/vm.slang` `vm_step()`
3. Add to `compiler/bytecode.py` `OpCode` enum
4. Generate bytecode in `compiler/codegen.py`
5. Add CPU implementation in `api/interpreter.py`
6. Add tests

### Run Tests

```bash
python -m pytest tests/ -v          # All tests
python -m pytest tests/test_gpu_dispatch.py -v  # GPU tests
python demo.py                      # Quick demo
```

## Implementation Notes

Critical details discovered during development:

1. **Heap Pointer**: XValue.data stores word offset (not byte offset)
2. **Entity 0**: tablePtr=0 is treated as invalid entity
3. **FunctionDescriptor**: Field order is [codeOffset, paramCount, localCount, upvalueCount, nameIndex]
4. **Buffer Sizes**: HeapAllocator = 32 bytes, EntityPoolState = 32 bytes
5. **VM_STACK_SIZE**: 32 (reduced for GPU resource limits)
6. **32-bit Only**: No 64-bit types in Slang code

## TODO

- [ ] String concatenation
- [ ] Table iterators (pairs/ipairs)
