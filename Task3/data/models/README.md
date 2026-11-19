# Model Files

The `.gguf` model weight files are **not stored in this repository** because they are large binary files (1.27 GB and 0.49 GB respectively).

## Download the models

```powershell
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-fp16.gguf    --local-dir .
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir .
```

Official model page: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF

| File | Quantisation | Size |
|---|---|---|
| `qwen2.5-0.5b-instruct-fp16.gguf` | FP16 (baseline) | ~1.27 GB |
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Q4_K_M (4-bit) | ~0.49 GB |
