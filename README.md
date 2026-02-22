# simple-astlm

- Dataset: BigCode The Stack (Python) extracting 20k function-level AST modules to train +2k to test
- Tokenized using Qwen2.5 tokenizer and used packing into 2048-token blocks (avg AST length ~660 tokens)
- Fine-tuned Qwen2.5-0.5B with LoRA (r=16, alpha=16, dropout=0.05) for 1 epoch, batch=2, grad_accum=8