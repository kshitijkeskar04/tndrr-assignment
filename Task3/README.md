# Task 3 — Running a Model Locally & Optimisation

Benchmark comparing **FP16** vs **Q4\_K\_M (4-bit)** quantisation of `Qwen2.5-0.5B-Instruct`
using **llama.cpp** (`llama-server`) in CPU-only mode on an HP Pavilion laptop.

---

## 📂 Repository Structure

```
task3/
│
├── docs/
│   └── REPORT.md              <- Full written report (findings, theory, 20-user plan)
│
├── code/
│   ├── run_benchmark.py       <- Benchmark harness (runs llama-server & collects metrics)
│   └── requirements.txt       <- Python dependencies (requests, psutil)
│
├── data/
│   ├── models/
│   │   ├── README.md          <- Download instructions for model files
│   │   ├── qwen2.5-0.5b-instruct-fp16.gguf       (git-ignored, 1.27 GB)
│   │   └── qwen2.5-0.5b-instruct-q4_k_m.gguf    (git-ignored, 0.49 GB)
│   ├── prompts/
│   │   └── prompts.json       <- 10 standardised benchmark prompts
│   └── results/
│       ├── FP16_results.json  <- Raw metrics: tokens/s, RSS, per-prompt output (FP16)
│       ├── FP16_results.csv
│       ├── Q4_K_M_results.json
│       ├── Q4_K_M_results.csv
│       └── quality_comparison.txt <- Side-by-side model outputs for all 10 prompts
│
├── .gitignore
└── README.md
```

---

## ⚡ Key Results (TL;DR)

| Metric | FP16 | Q4\_K\_M | Delta |
|---|---:|---:|---|
| Model file size | 1.266 GB | 0.491 GB | **-61.2%** |
| Peak server RAM (RSS) | 1.146 GB | 0.673 GB | **-41.3%** |
| Mean generation speed | 18.3 tok/s | 41.6 tok/s | **+2.27x** |
| Avg quality score (/5) | 2.7 | 2.8 | approx same |

📄 **Full analysis → [`docs/REPORT.md`](docs/REPORT.md)**

---

## 🚀 How to Reproduce

### 1. Install llama.cpp
```powershell
winget install llama.cpp
# or download from https://github.com/ggml-org/llama.cpp/releases
```

### 2. Download model files
```powershell
py -m pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-fp16.gguf    --local-dir data/models
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir data/models
```

### 3. Install Python dependencies
```powershell
py -m pip install -r code/requirements.txt
```

### 4. Run the benchmark
```powershell
# FP16
py code/run_benchmark.py --model data/models/qwen2.5-0.5b-instruct-fp16.gguf --label FP16

# 4-bit
py code/run_benchmark.py --model data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf --label Q4_K_M
```

Results are written to `data/results/`.

---

## 🛠 Runtime Settings

| Parameter | Value |
|---|---|
| Runtime | llama.cpp (CPU-only, `-ngl 0`) |
| Threads | 6 |
| Context size | 2,048 tokens |
| Max generated tokens | 160 |
| Temperature | 0 (greedy) |
| Seed | 42 |

---

## 📋 Requirements

- Windows 10 / 11 (or Linux / macOS)
- Python 3.9+
- llama.cpp (`llama-server` on PATH)
- `requests`, `psutil` (see `code/requirements.txt`)
