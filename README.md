# simple-astlm

## Iteration 1: ast.dump (baseline)
- Dataset: BigCode The Stack (Python) extracting 20k function-level AST modules to train +2k to test`
- Representation: `ast.dump()` string serialization
- Tokenization: Qwen2.5 tokenizer, packed into 2048-token blocks (avg ~660 tokens)
- Model: Qwen2.5-0.5B + LoRA (r=16, alpha=16, dropout=0.05)
- Training: 1 epoch, batch=2, grad_accum=8

## Iteration 2: JSON AST (structured representation)

- Representation: AST converted to JSON (explicit hierarchical structure)
- Training: Same setup as Iteration 1
- Generation: Model outputs JSON, followed by JSON repair before parsing