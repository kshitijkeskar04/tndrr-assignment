# Task 3 — Running a Model Locally & Optimisation: Benchmark Report

## 1. Executive Summary & Objective

The objective of this benchmark is to evaluate the operational and qualitative trade-offs of deploying a local Large Language Model (LLM) under two precision levels: **Full Precision / Half Precision (FP16)** versus **4-bit Quantisation (Q4_K_M)**. 

The evaluation was performed under strictly controlled, identical conditions using `llama.cpp` (`llama-server`) on an x86-64 host machine. Both configurations were evaluated across ten standardised prompts covering technical explanation, code generation, SQL synthesis, summarisation, mathematical reasoning, and instruction following.

### Key Benchmark Findings

| Metric | FP16 (Baseline) | Q4_K_M (Quantised) | Delta / Impact |
|---|---|---|---|
| **Model On-Disk Size** | 1.266 GB (1,266,425,696 B) | 0.491 GB (491,400,032 B) | **-61.2%** storage reduction |
| **Peak Server Resident Memory (RSS)** | 1.146 GB | 0.673 GB | **-41.3%** memory savings |
| **Mean Generation Throughput** | 18.314 tokens/sec | 41.612 tokens/sec | **+127.2% (+2.27x)** speedup |
| **Average Qualitative Score (out of 5)** | 2.7 / 5.0 | 2.8 / 5.0 | Comparable output quality (within margin) |
| **Instruction Adherence (Bullet / Word count)** | Poor (failed P01/P08) | Poor (failed P01/P08) | Architecture limitation of 0.5B scale |

---

## 2. Hardware and Software Environment

To ensure strict reproducibility and avoid non-deterministic GPU memory offload behaviour across different runs, both model variants were executed in **CPU-only execution mode (`-ngl 0`)** with fixed thread pooling and locked random seeds.

| Parameter | Specification | Note / Rationale |
|---|---|---|
| **Host System** | HP Pavilion Laptop | Windows 10 Home (Build 26300) |
| **Processor (CPU)** | Intel Core (Family 6, Model 140) | 8 logical cores |
| **Total System Memory (RAM)** | 8.32 GB | Constrained consumer memory envelope |
| **Discrete GPU** | NVIDIA GeForce GTX 1650 (4 GB VRAM) | Documented via `nvidia-smi`; bypassed via `-ngl 0` |
| **Inference Runtime** | `llama.cpp` (`llama-server.exe`) | Official GGML/GGUF high-performance C++ engine |
| **Model Evaluated** | `Qwen2.5-0.5B-Instruct` GGUF | Official Qwen team GGUF export |
| **Execution Threads (`-t`)** | 6 threads | Tuned to physical core balance, avoiding OS stalls |
| **Context Size (`-c`)** | 2,048 tokens | Uniform slot context window |
| **Maximum Generation (`n_predict`)** | 160 tokens | Fixed generation budget per prompt |
| **Sampling Parameters** | `temperature = 0.0`, `seed = 42` | Strictly deterministic greedy decoding |

---

## 3. Model Configurations & File Profiles

| Configuration | Model Filename | Quantisation Type | Precision | File Size |
|---|---|---|---|---|
| **Configuration A** | `qwen2.5-0.5b-instruct-fp16.gguf` | Unquantised baseline | 16-bit Float (FP16) | 1.266 GB |
| **Configuration B** | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | Medium k-quant | 4-bit (Q4_K_M) | 0.491 GB |

*Note on Q4_K_M:* The `Q4_K_M` format applies 4-bit quantization with variable block scales, keeping sensitive attention and feed-forward matrices at higher precision where needed to minimise perplexity degradation compared to legacy naive Q4_0 formats.

---

## 4. Empirical Performance Results

Data extracted directly from automated server logging (`FP16_results.json` and `Q4_K_M_results.json`):

### Overall Resource and Speed Comparison

```
Model Size (GB):
FP16:    [==================================] 1.266 GB
Q4_K_M:  [==============] 0.491 GB (-61.2%)

Peak Server Memory RSS (GB):
FP16:    [==============================] 1.146 GB
Q4_K_M:  [==================] 0.673 GB (-41.3%)

Mean Generation Throughput (tok/s):
FP16:    [==================] 18.31 tok/s
Q4_K_M:  [==========================================] 41.61 tok/s (+2.27x)
```

### Per-Prompt Throughput Breakdown

| Prompt ID | Domain / Task | FP16 Speed (tok/s) | Q4_K_M Speed (tok/s) | Q4 Speedup Factor |
|---|---|---:|---:|---:|
| **P01** | ML Basics (General) | 17.19 | 47.52 | **2.76x** |
| **P02** | Python Factorial (Code) | 15.46 | 50.35 | **3.26x** |
| **P03** | Supervised vs Unsupervised (ML) | 17.31 | 49.46 | **2.86x** |
| **P04** | Overfitting Remediation (ML) | 17.50 | 49.50 | **2.83x** |
| **P05** | Neural Net Learning (Theory) | 18.36 | 26.56 | **1.45x** |
| **P06** | SQL Duplicate Email (Database) | 20.37 | 36.62 | **1.80x** |
| **P07** | Python Decorator (Code) | 18.93 | 39.31 | **2.08x** |
| **P08** | Transit Text Summary (NLP) | 20.72 | 36.24 | **1.75x** |
| **P09** | Profit Calculation (Math) | 19.09 | 39.36 | **2.06x** |
| **P10** | Transformer Mechanics (Arch) | 18.20 | 41.18 | **2.26x** |
| **Average** | **All 10 Prompts** | **18.31** | **41.61** | **2.27x** |

### Performance Observations

1. **Bandwidth-Bound Speedup:** Q4_K_M demonstrated an average generation speedup of **2.27x** (peaking at **3.26x** on P02). In autoregressive language generation on CPUs, execution is predominantly **memory-bandwidth bound**. Because each forward pass requires reading all model weights from main RAM into the CPU cache to predict a single token, reducing the weight size by ~61% directly decreases memory bus congestion and enables much faster token emission.
2. **Memory Footprint Reduction:** Resident set size (RSS) dropped by 473 MB (41.3%). The reduction does not strictly match the 61% file size reduction because both configurations allocate identical KV caches, context buffers, and thread-pool execution stacks.

---

## 5. Ten-Prompt Qualitative Evaluation

Responses were evaluated against a 5-point rubric measuring:
- **Correctness:** Factual and syntactical accuracy.
- **Completeness:** Addressing all sub-questions within prompt boundaries.
- **Clarity:** Structural coherence and readability.
- **Constraint Following:** Adherence to negative constraints, formatting rules, and token limits.

### Qualitative Scorecard

| ID | Category | Prompt Summary | FP16 Score | Q4_K_M Score | Comparative Qualitative Observation |
|---|---|---|---:|---:|---|
| **P01** | Explanation | ML beginner explanation (120 words) + example | 4/5 | 4/5 | Both gave accurate, clear overviews using image recognition as an example. Neither hit the exact 120-word target (~80 and ~76 words respectively), but both were coherent and concise. |
| **P02** | Python | Factorial with validation & example | 2/5 | 2/5 | **FP16** wrote clean recursive code with proper validation (`isinstance(n, int) and n >= 0`), but truncated at `# Example usage:\ntry`. **Q4_K_M** wrote iterative code with doctests, but **failed validation** (only checked `n < 0`, omitting non-integer validation) and truncated at the loop. |
| **P03** | ML Theory | Supervised vs unsupervised + 2 examples each | 2/5 | 2/5 | Both models provided high-quality definitions of supervised learning, but both hit the 160-token budget and truncated before listing unsupervised learning or full examples. |
| **P04** | ML Theory | Overfitting definition + 3 ways to reduce | 2/5 | 3/5 | **FP16 hallucinated/misunderstood the prompt**, listing three *causes* of overfitting instead of remedies. **Q4_K_M** correctly identified *Data Augmentation* and *Regularization*, though it truncated before completing the 3rd remedy. |
| **P05** | Deep Learning | Neural net learning: weights, loss, gradient descent | 3/5 | 2/5 | **FP16** correctly touched on weights, loss, and gradient descent, but had a severe conceptual slip: *"gradient descent calculates the direction of steepest ascent"*. **Q4_K_M** deviated into describing network layers (Input, Hidden, Output) rather than the mathematical learning loop. |
| **P06** | Database SQL | Find duplicate email in `users` table | 3/5 | 3/5 | Both wrote the exact identical valid query (`SELECT email_address FROM users GROUP BY email_address HAVING COUNT(email_address) > 1;`). Both truncated mid-explanation. |
| **P07** | Python | Explain decorators + working example | 4/5 | 4/5 | Both generated identical working decorator patterns (`my_decorator` wrapping `say_hello`). Clear and idiomatic, with slight truncation in the closing text. |
| **P08** | Instruction Following | Summarise transit text in **exactly 3 bullet points** | 2/5 | 2/5 | **Both models failed the negative constraint**: both generated 5 bullet points by splitting each original sentence into a bullet. Indicates common constraint-following weakness in sub-1B parameter models. |
| **P09** | Math / Reasoning | Buy 800, sell 1000: profit and profit % | 3/5 | 3/5 | Both correctly computed Profit = ₹200. Both started the correct percentage formula ($\frac{200}{800} \times 100 = 25\%$), but both hit the token cap before outputting the final "25%" value. |
| **P10** | Architecture | Explain Transformer: tokens, attention, positional info | 2/5 | 3/5 | **FP16 made a major factual error**, asserting that the Transformer is a *"recurrent neural network (RNN)"*. **Q4_K_M** correctly classified it as a deep learning architecture and defined tokens and attention clearly, though truncated before positional encodings. |
| **AVG** | **Overall** | **Mean Score across all 10 Prompts** | **2.7 / 5** | **2.8 / 5** | **Negligible difference in reasoning ability.** Quantisation did not degrade capability. |

### Qualitative Takeaways

1. **Quantisation Degradation is Minimal:** 4-bit quantisation did not systematically degrade reasoning or syntax. In fact, on P04 and P10, Q4_K_M produced factually superior responses compared to FP16.
2. **Impact of Context / Generation Ceiling:** In 7 out of 10 prompts, responses were truncated by the 160-token limit. This proves that observed quality issues were heavily influenced by the decoding cutoff rather than model comprehension alone.
3. **Model Scale Bottlenecks:** At 0.5B parameters, instruction following (e.g., "exactly 3 bullets" in P08, strict word count in P01) and factual grounding (FP16 calling Transformers RNNs in P10) represent inherent capacity limitations rather than quantization artifacts.

---

## 6. Theoretical Foundation: What is the KV Cache?

### Conceptual Definition

In autoregressive Transformer models, text generation proceeds **token-by-token**. When generating token $t_{n+1}$, the self-attention mechanism must compute attention scores between the new token and all preceding tokens $t_1, t_2, \dots, t_n$.

In the self-attention equation:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

- **Query ($Q$):** Generated only for the current incoming token being processed.
- **Key ($K$) and Value ($V$):** Representations of all historical context tokens.

Without a KV cache, the model would be forced to recompute the Key and Value projection matrices for all historical tokens at every single generation step, turning the sequence generation complexity into an intractable **$O(N^2)$ operation per sequence**.

The **KV Cache** is an active memory buffer that stores the precomputed Key and Value tensor representations of all past tokens in GPU or system memory. At each subsequent step, the model computes $K$ and $V$ **only for the single newly generated token**, appends them to the cache, and computes attention via a quick lookup, restoring per-token generation to **$O(1)$ new projections**.

### Mathematical Memory Footprint of the KV Cache

The memory consumed by the KV cache scales linearly with sequence length, batch size, and network depth:

$$\text{Memory}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L_{\text{seq}} \times b_{\text{bytes}} \times B_{\text{batch}}$$

Where:
- $2$ accounts for the two distinct matrices stored: Keys ($K$) and Values ($V$).
- $n_{\text{layers}}$: Number of Transformer decoder layers.
- $n_{\text{kv\_heads}}$: Number of key/value attention heads (in Multi-Query or Grouped-Query Attention, this is significantly smaller than query heads).
- $d_{\text{head}}$: Dimension per attention head ($d_{\text{model}} / n_{\text{heads}}$).
- $L_{\text{seq}}$: Cumulative sequence length (prompt tokens + generated tokens).
- $b_{\text{bytes}}$: Bytes per floating point element (2 for FP16/BF16, 1 for FP8, 0.5 for FP4).
- $B_{\text{batch}}$: Number of concurrent sequences being served.

---

## 7. Theoretical Foundation: Why Quantisation Saves Memory & Accelerates Inference

### 1. Weight Precision Reduction

Standard floating-point numbers represent values using IEEE 754 formats:
- **FP32 (Single Precision):** 32 bits = 4 bytes per parameter.
- **FP16 / BF16 (Half Precision):** 16 bits = 2 bytes per parameter.
- **4-Bit Quantisation (Q4 / INT4):** 4 bits = 0.5 bytes per parameter.

Moving from FP16 to 4-bit quantisation achieves a **$4\times$ theoretical reduction (75%)** in the storage requirement of the weight matrices:
$$\text{Memory}_{\text{weights}} \approx \text{Parameters} \times \text{Bytes per Parameter}$$

*Why observed savings are ~61.2% rather than 75%:*
In modern formats such as GGUF `Q4_K_M` or AWQ, real-world compression is slightly lower (~60–65%) because:
1. **Block Scales and Biases:** Quantisation formats divide weights into blocks (e.g., 32 or 256 weights) and store scale factors and minimums in higher precision (FP16 or FP8) to prevent accuracy degradation.
2. **Critical Layer Preservation:** Sensitive tensors (embedding tables, token output heads, layer norms) are often retained in FP16 or 8-bit precision to safeguard model stability.

### 2. Why Quantisation Accelerates Token Generation (The Memory-Bandwidth Wall)

A common misconception is that quantisation only speeds up models by reducing arithmetic complexity. On modern hardware, autoregressive generation (batch size 1 or small batches) has an **extremely low arithmetic intensity** (FLOPs per byte transferred):
- To generate a single token, **every single model weight must be read from memory into cache/registers**.
- The hardware's compute units (ALUs/Tensor Cores) sit idle waiting for memory retrieval. The inference latency is governed by **memory bandwidth**:

$$\text{Generation Time per Token} \approx \frac{\text{Bytes of Weights Loaded}}{\text{Memory Bandwidth (GB/s)}}$$

By compressing weights from 16 bits to 4 bits, the total data transferred across the memory bus per token drops by up to ~65%. This eliminates memory bus saturation, explaining why our empirical run witnessed a **2.27x throughput jump** on the CPU.

---

## 8. Back-of-the-Envelope Serving Plan: 20 Concurrent Users on a Single 24 GB GPU

### 1. System & Architecture Assumptions

To serve **20 concurrent active users** simultaneously on a single consumer or enterprise 24 GB GPU (e.g., NVIDIA RTX 3090, RTX 4090, or A10G), we establish realistic production assumptions:

| Assumption | Value / Choice | Rationale |
|---|---|---|
| **Target Hardware** | 1x NVIDIA RTX 4090 (24 GB VRAM) | Memory bandwidth: 1,008 GB/s; Tensor Cores: 4th Gen |
| **Model Selection** | `Llama-3.1-8B-Instruct` | 8.03B parameters, state-of-the-art open weights |
| **Weight Quantisation** | 4-bit AWQ / GPTQ (`INT4`) | Near-zero perplexity loss, fast GEMM dequantisation |
| **Attention Mechanism** | **Grouped-Query Attention (GQA)** | $n_{\text{heads}} = 32$, $n_{\text{kv\_heads}} = 8$ (Crucial for memory scalability) |
| **Target Concurrency** | 20 simultaneous active decoding streams | Continuous active generation |
| **Average Context Length** | 2,048 tokens per user | Combined prompt (prefill) and generation window |
| **Inference Engine** | **vLLM** (or TensorRT-LLM) | PagedAttention, continuous batching, chunked prefill |

---

### 2. Rigorous Memory Budget Calculation

#### Step A: Quantised Model Weight Footprint
For an 8-billion parameter model quantized to 4 bits:
- Baseline weight size: $8.03 \times 10^9 \times 0.5 \text{ bytes} \approx 4.02 \text{ GB}$.
- Quantisation metadata, block scales, and unquantised embedding/head layers ($\approx 1.2 \text{ GB}$):
$$\text{Weight VRAM} \approx \mathbf{5.2\text{ GB}}$$

#### Step B: KV Cache Footprint (The Decisive GQA Factor)
Llama-3.1-8B utilizes Grouped Query Attention:
- Number of layers ($n_{\text{layers}}$) = 32
- Number of KV heads ($n_{\text{kv\_heads}}$) = 8
- Head dimension ($d_{\text{head}}$) = 128
- Precision = 16-bit FP16 ($b_{\text{bytes}} = 2$)

**Per-token KV cache memory:**
$$\text{KV per token} = 2 \times 32 \times 8 \times 128 \times 2 \text{ bytes} = 131,072 \text{ bytes} = \mathbf{128\text{ KB / token}}$$

**Per-user KV cache allocation at 2,048 tokens:**
$$\text{KV per user} = 2,048 \times 128\text{ KB} = 262,144\text{ KB} = \mathbf{256\text{ MB / user}}$$

**Total KV cache for 20 concurrent users:**
$$\text{Total KV}_{\text{FP16}} = 20 \times 256\text{ MB} = 5,120\text{ MB} \approx \mathbf{5.12\text{ GB}}$$

*(Architectural Note: If the model used legacy Multi-Head Attention with 32 KV heads, the KV cache would be $4\times$ larger: $20.48\text{ GB}$, making 20 concurrent users impossible on a 24 GB GPU! GQA is what makes this deployment viable).*

*Further Optimization (FP8 KV Cache):* Using `--kv-cache-dtype fp8` reduces this to **2.56 GB**, freeing an additional 2.5 GB of VRAM.

#### Step C: CUDA Runtime, PagedActivation, and Compute Buffers
- PyTorch CUDA context and driver runtime: $\approx 1.2 \text{ GB}$
- Temporary activation scratchpads and cuBLAS buffers for batch size 20: $\approx 1.8 \text{ GB}$
$$\text{Runtime & Activations} \approx \mathbf{3.0\text{ GB}}$$

---

### 3. Comprehensive 24 GB VRAM Allocation Breakdown

| Memory Segment | Allocated VRAM | Share of 24 GB | Status / Description |
|---|---:|---:|---|
| **Model Weights (4-bit AWQ)** | 5.20 GB | 21.7% | Static footprint, locked in VRAM |
| **KV Cache (FP16, 20 users @ 2K tokens)** | 5.12 GB | 21.3% | Dynamic block-allocated pool (PagedAttention) |
| **CUDA Runtime & Activations** | 3.00 GB | 12.5% | Driver context, scratchpads, GEMM buffers |
| **Dynamic Headroom & Safety Buffer** | 10.68 GB | 44.5% | Burst protection, longer contexts, or scaling up to ~50 users |
| **Total VRAM Capacity** | **24.00 GB** | **100.0%** | **Completely safe from OOM crashes** |

```
[===== Weights: 5.2 GB =====][== KV Cache: 5.1 GB ==][= Activations: 3.0 GB =][========= Headroom: 10.7 GB =========]
0 GB                                                                                                        24 GB
```

---

### 4. Serving Architecture & Operational Strategy

To ensure sub-second response times and predictable latency under concurrent load, the deployment employs the following strategies:

1. **vLLM with PagedAttention:**
   Traditional serving allocates contiguous memory per user, causing up to 60–80% internal and external memory fragmentation. PagedAttention divides the KV cache into small virtual memory blocks (e.g., 16 tokens per block), allocating memory dynamically on-demand only as tokens are generated.
2. **Continuous (Iteration-Level) Batching:**
   Instead of static batching (where all 20 requests must wait for the longest generation to finish), continuous batching inserts newly arriving requests into the next forward pass iteration immediately, maximizing GPU core occupancy.
3. **Chunked Prefill (`max_num_batched_tokens = 512`):**
   Long prompts require massive compute (prefill phase). Without chunking, a new user entering a 1,500-token prompt would cause latency spikes (jitter) for the other 19 users currently receiving streaming tokens. Chunking interleaves prefill compute with active token generation.
4. **Throughput and Latency Estimation:**
   - RTX 4090 memory bandwidth: $\approx 1,008 \text{ GB/s}$.
   - Memory read per step (weights + KV cache): $\approx 5.2 \text{ GB} + 5.1 \text{ GB} \approx 10.3 \text{ GB}$.
   - Time per decode iteration: $\frac{10.3 \text{ GB}}{1008 \text{ GB/s}} \approx 10.2 \text{ ms}$.
   - For batch size 20: 20 tokens emitted every $\approx 15 \text{ ms}$ (including attention compute overhead).
   - **Per-user generation rate:** $\approx 25–35 \text{ tokens/sec}$ simultaneously!
   - This comfortably exceeds human reading speeds (~5–8 tokens/sec).

---

## 9. Conclusion

This benchmark provides rigorous empirical confirmation of the advantages of low-bit quantisation for local LLM deployment:
1. **Efficiency:** Quantising `Qwen2.5-0.5B-Instruct` to 4-bit (`Q4_K_M`) delivered a **2.27x throughput increase** and cut process memory consumption by **41.3%** on consumer hardware.
2. **Fidelity:** Downstream output quality remained virtually indistinguishable across 10 diverse prompts, with average rubric scores of **2.7/5 (FP16)** and **2.8/5 (Q4_K_M)**.
3. **Serving Scalability:** Thanks to modern innovations in model architecture (Grouped-Query Attention) and serving engines (PagedAttention in vLLM), serving 20 concurrent users with 2,048 tokens of context on a single 24 GB GPU is not only mathematically feasible, but leaves **over 10 GB of safety headroom**.
