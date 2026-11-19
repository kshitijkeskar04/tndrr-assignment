# TASK 2: Retrieval-Augmented Generation (RAG) from Scratch
## Comprehensive Technical & Empirical Submission Report

**Author / Candidate**: Kshitij Keskar  
**Task Definition**: Build a Question-Answering system over any public document set of at least 30 pages from scratch, without using LangChain, LlamaIndex, Haystack, or any external RAG framework. Use only an embedding model + NumPy/FAISS (and an optional LLM for final document-grounded answers).  
**Source Dataset**: World Travel & Tourism Council (WTTC), *Introduction to Artificial Intelligence (AI) Technology: Guide for Travel & Tourism Leaders* (January 2024).  
**Document Scale**: 44 physical PDF pages (Main body, technical annexes, and AI glossary span pages 3–43).

---

## Executive Summary

This report presents a first-principles implementation and rigorous empirical evaluation of a Retrieval-Augmented Generation (RAG) system. The pipeline ingests the 44-page WTTC AI report, cleans and segments it into two distinct chunk configurations (100 words and 300 words with 20-word sliding overlap), encodes each chunk into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`, and performs vector similarity search via pure NumPy dot product (with optional FAISS `IndexFlatIP`).

A benchmark of **15 hand-authored test questions** was evaluated across both chunk configurations:
- **100-word chunks**: Achieved **93.33% Recall@5** (14/15 hits) with a **Mean Reciprocal Rank (MRR@5) of 0.9000** and an **86.7% Rank-1 hit rate** (13/15 at Rank 1). Average matched cosine similarity was **0.6302**.
- **300-word chunks**: Achieved **93.33% Recall@5** (14/15 hits) with an **MRR@5 of 0.7444** and a **66.7% Rank-1 hit rate** (10/15 at Rank 1). Average matched cosine similarity was **0.5701**.
- **Empirical Winner**: **100-word chunks** significantly outperformed 300-word chunks in retrieval precision (+15.56 percentage points in MRR, +3 Rank-1 hits, and +0.06 higher average cosine similarity) because 100-word chunks maintain high semantic density and avoid the **vector dilution** that degrades larger chunks.

Two failure modes were observed and analyzed:
1. **Q1 on 100-word chunks**: Missed top-5 due to chunk boundary splitting and dense model preference for generic introductory definitions over specific structural entities ("three components"). Fixed via **BM25 Hybrid Retrieval** or 300-word chunks (Rank 3).
2. **Q10 on 300-word chunks**: Missed top-5 due to **semantic dilution** of the concise ANI/AGI/ASI taxonomy inside a larger paragraph. Fixed via **100-word chunks (Rank 1)** or **Cross-Encoder Re-Ranking**.

---

## 1. Document Set Suitability & Ingestion

### Document Metadata
- **Title**: *Introduction to Artificial Intelligence (AI) Technology: Guide for Travel & Tourism Leaders*
- **Publisher**: World Travel & Tourism Council (WTTC)
- **Publication Date**: January 2024
- **Physical Pages**: 44 pages
- **Extracted Content Range**: Physical PDF pages 3 through 43 (41 extracted content pages, strictly exceeding the required 30-page threshold).
- **Text Extraction Statistics**: 99,594 characters, 14,845 total words.

### Document Structure & Thematic Depth
The document is a comprehensive industry primer covering:
- **Historical Foundations**: Alan Turing (1950) and computing evolution.
- **The Core AI Triad**: Algorithms, Data, and Compute.
- **Machine Learning vs. Deep Learning**: Artificial Neural Networks (ANN), training paradigms (Supervised, Unsupervised, Reinforcement Learning).
- **Big Data Infrastructure**: The 4 Vs (Volume, Velocity, Variety, Veracity), Data Warehouses vs. Data Lakes.
- **Hardware Acceleration**: CPUs, GPUs (parallel processing), and TPUs (matrix calculations, Moore's Law).
- **AI Taxonomy**: Artificial Narrow Intelligence (ANI), Artificial General Intelligence (AGI), and Artificial Super Intelligence (ASI).
- **Generative AI & LLMs**: Multimodal Foundation Models vs. LLMs, Prompt Engineering, Hallucinations.
- **Societal Impact**: The Digital Divide (ITU internet access statistics), Workforce Upskilling (Randstad survey data).

### Text Preprocessing & Page Alignment
The PDF text was extracted using `pypdf.PdfReader`. A key engineering finding during extraction was that **physical PDF pages correspond 1:1 with printed page numbers** (e.g., physical page 8 contains `< Contents | 8`). Consecutive whitespace and newline artifacts were normalized into clean paragraph blocks while preserving punctuation and numerical entities.

---

## 2. System Architecture & Mathematical Formulation

The entire pipeline was built from scratch without high-level abstractions:

```
[44-Page WTTC PDF]
        │
        ▼  pypdf extraction (pages 3–43)
[Clean Extracted Text (41 pages, 14,845 words)]
        │
        ▼  Sliding-window segmentation (step = chunk_size - overlap)
┌─────────────────────────────────┬─────────────────────────────────┐
│       100-Word Chunks           │       300-Word Chunks           │
│   (20w overlap, 198 chunks)     │   (20w overlap, 73 chunks)      │
└────────────────┬────────────────┴────────────────┬────────────────┘
                 │                                 │
                 ▼  SentenceTransformer            ▼
           all-MiniLM-L6-v2                  all-MiniLM-L6-v2
                 │                                 │
                 ▼  L2-Normalize                   ▼  L2-Normalize
┌─────────────────────────────────┬─────────────────────────────────┐
│     Matrix D_100 (198 x 384)    │     Matrix D_300 (73 x 384)     │
└────────────────┬────────────────┴────────────────┬────────────────┘
                 │                                 │
                 ├────────── User Query ───────────┤
                 │  q_vec = model.encode(q)        │
                 ▼  L2-Normalize: ||q_vec||_2 = 1  ▼
┌─────────────────────────────────┬─────────────────────────────────┐
│  NumPy Dot Product: S = D @ q   │  NumPy Dot Product: S = D @ q   │
│  Top-k Argsort: idx = argsort(S)│  Top-k Argsort: idx = argsort(S)│
└────────────────┬────────────────┴────────────────┬────────────────┘
                 │                                 │
                 ▼                                 ▼
      [Top-5 Retrieved Passages]        [Top-5 Retrieved Passages]
                 │
                 ▼  (Optional) OpenAI-compatible client
      [Document-Grounded Answer with Page Citations]
```

### Sliding-Window Chunking Formulation
Given an ordered sequence of words $W = [w_1, w_2, \dots, w_M]$ on page $P$, a sliding window of size $C$ with overlap $O$ produces chunks at step $S = C - O$:
$$\text{Chunk}_k = [w_{k \cdot S}, w_{k \cdot S + 1}, \dots, w_{k \cdot S + C - 1}]$$
- For $C = 100$ and $O = 20$: Step $S = 80$ words. Total chunks = **198 chunks**.
- For $C = 300$ and $O = 20$: Step $S = 280$ words. Total chunks = **73 chunks**.

### Dense Vector Embedding & Normalization
Each chunk text $T_i$ is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`:
$$v_i = \text{Encoder}(T_i) \in \mathbb{R}^{384}$$
All chunk vectors and the query vector are $L_2$-normalized to unit length:
$$\hat{v}_i = \frac{v_i}{\|v_i\|_2}, \quad \hat{q} = \frac{q}{\|q\|_2}$$

### Cosine Similarity via NumPy Matrix Multiplication
Because all vectors have unit norm ($\|\hat{v}_i\|_2 = 1$ and $\|\hat{q}\|_2 = 1$), the cosine similarity between the query and every document chunk simplifies to the dot product:
$$\cos(\theta_i) = \frac{v_i \cdot q}{\|v_i\|_2 \|q\|_2} = \hat{v}_i \cdot \hat{q}$$
For the entire document collection matrix $\mathbf{D} \in \mathbb{R}^{N \times 384}$, all similarity scores are computed in a single vectorized BLAS operation:
$$\mathbf{S} = \mathbf{D} \hat{q} \in \mathbb{R}^N$$
The top-$k$ most relevant chunks are retrieved via descending index sorting:
$$\text{Top-}k = \text{argsort}(\mathbf{S})[::-1][:k]$$

---

## 3. The 15 Authored Test Questions and Gold Passages

The 15 evaluation questions were authored to test diverse retrieval challenges: direct factual definitions, comparative differences, multi-class taxonomies, technical hardware comparisons, and specific statistical citations.

| ID | Test Question | Gold Page | Expected Factual Answer | Exact Supporting Source Passage from PDF |
|:---:|:---|:---:|:---|:---|
| **Q1** | What three components explain how modern AI systems can exhibit intelligent behaviour? | **Page 8** | Advanced algorithms, data, and computing power. | *"It is these three components of advanced algorithms, data and computing power, that explain how machines can exhibit intelligent behaviour and why AI has suddenly exploded into our everyday lives."* |
| **Q2** | How does traditional programming differ from AI systems according to the report? | **Page 9** | Traditional programming follows explicitly pre-coded step-by-step rules; AI learns how to solve problems by analyzing training data. | *"Unlike traditional programming, AI systems are not explicitly coded in a series of precise steps, but instead ‘learn’ how to solve a problem by examining training data."* |
| **Q3** | What is Explainable AI trying to achieve? | **Page 10** | To understand and articulate how AI systems reach outputs, improving transparency and trust in AI systems. | *"‘Explainable AI’ is therefore an active and ongoing field in which researchers are trying to understand and articulate how AI systems reach an output. This is crucial to improve trust in artificial intelligence systems."* |
| **Q4** | What is the difference between machine learning and deep learning? | **Page 10** | Machine learning learns patterns from data; deep learning is a sophisticated subset that uses Artificial Neural Networks (ANNs) inspired by the human brain. | *"Deep Learning is a more sophisticated subset of Machine Learning and uses complex processes inspired by the human brain called Artificial Neural Networks (ANN)."* |
| **Q5** | How are supervised, unsupervised, and reinforcement learning distinguished? | **Page 11** | Supervised uses labeled data; unsupervised finds patterns in unlabeled data; reinforcement learns through trial-and-error feedback loops. | *"Supervised Learning: AI models are trained on data that has been correctly labelled. Unsupervised Learning: AI models are trained on data that has not been labelled. Reinforcement Learning: AI models ‘learn by doing’ through a trial-and-error feedback loop."* |
| **Q6** | What are the four Vs used to characterise Big Data? | **Page 13** | Volume, velocity, variety, and veracity. | *"These extremely large and very complex datasets are often called ‘Big Data’ and are characterised by their volume, velocity, variety and veracity, commonly referred to as the 4 V ’s."* |
| **Q7** | What is the difference between a data warehouse and a data lake? | **Page 15** | A data warehouse stores structured, organized, and optimized data; a data lake stores raw, unstructured data (images, videos, reviews, surveys). | *"A data warehouse is a repository that stores structured data, in a highly organised and optimised way. Alternatively a data lake is a repository that can store all types of unstructured data, such as social media posts, images, videos, customer feedback surveys, or online reviews."* |
| **Q8** | Why are GPUs useful for AI and deep learning? | **Page 16** | GPUs process tasks in parallel rather than sequentially, which is ideal for powering AI applications and deep learning workloads. | *"A key innovation was their ability to process tasks in parallel (rather than sequentially as in normal computers) and it turns out the accelerated parallel processing required to create lifelike graphics in computer games and movies is ideal for powering AI applications and especially deep learning systems."* |
| **Q9** | How does the report distinguish a TPU from a GPU? | **Page 18** | A TPU performs matrix calculations that significantly accelerate machine learning; analogous to a GPU printing a line of text at a time while a TPU prints an entire page at once. | *"A TPU chip takes this a step further and can perform ‘matrix calculations’ which significantly accelerates machine learning tasks... One analogy is that a GPU chip can be thought of as a printer that prints a line of text at a time, whereas a TPU chip is a printer that can print an entire page of text at a time."* |
| **Q10** | What are the three capability-based categories of AI? | **Page 21** | Artificial Narrow Intelligence (ANI), Artificial General Intelligence (AGI), and Artificial Super Intelligence (ASI). | *"An AI system delivers either artificial narrow intelligence (ANI), artificial general intelligence (AGI), or artificial super intelligence (ASI)."* |
| **Q11** | What is generative AI and what kinds of content can it produce? | **Page 24** | AI technology that creates new content (text, imagery, video, audio) based on user prompts and patterns learned from large training datasets. | *"Generative AI is a type of AI technology that can produce new and creative content such as text, imagery, video and audio, from a user ‘prompt’."* |
| **Q12** | What is the difference between a large language model and a foundational model? | **Page 37** | LLMs specialize in language/text; foundational models are a broader category trained on multimodal data (text, images, video, audio), encompassing LLMs. | *"Foundational Models are a broader category of underlying AI architecture, which includes LLM’s, but also AI models trained with more than just text, such as large datasets of images, video or audio."* |
| **Q13** | What does the report mean by an AI hallucination? | **Page 38** | Outputs generated by AI systems that appear plausible and real but are factually incorrect or unrealistic. | *"Hallucinations are outputs from AI systems that may seem real, but are in fact incorrect, or unrealistic."* |
| **Q14** | According to the report, what proportion of the world had never been connected to the internet in 2022? | **Page 31** | 34% of the global population, representing approximately 2.6 billion people. | *"The International Telecommunications Union (ITU) estimates that 34% of the world has never been connected to the internet (in 2022), representing 2.6 billion people."* |
| **Q15** | What did the 2023 Randstad survey report about workers receiving AI training? | **Page 33** | Only 13% of surveyed workers had been offered AI training in the preceding 12 months. | *"Randstad surveyed 7,000 employees in companies of all sizes, across 34 countries and found that only 13% of workers had been offered any form of AI training in the last 12 months."* |

---

## 4. Empirical Retrieval Accuracy Benchmark (Recall@5 & MRR@5)

### Evaluation Metrics
- **Recall@5**: The fraction of questions for which at least one chunk from the gold target page appears in the top-5 retrieved results ($k=5$):
  $$\text{Recall@5} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Gold Page}_i \in \text{Top-5}_i)$$
- **Mean Reciprocal Rank (MRR@5)**: Evaluates the ranking quality of retrieved results, giving higher credit when the relevant chunk appears earlier:
  $$\text{MRR@5} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{Rank}_i} \quad (\text{where } \text{Rank}_i \in \{1, 2, 3, 4, 5\}; \text{0 if not retrieved})$$

### Benchmark Comparison

| Evaluation Metric | 100-Word Chunks (20-word overlap) | 300-Word Chunks (20-word overlap) | Delta ($\Delta$) | Winner |
|:---|:---:|:---:|:---:|:---:|
| **Total Chunks in Corpus** | 198 chunks | 73 chunks | +125 chunks | 100w (Finer Granularity) |
| **Successful Hits in Top 5** | **14 / 15** | **14 / 15** | 0 | Tied |
| **Recall@5 Accuracy** | **93.33%** | **93.33%** | 0.00% | Tied |
| **Mean Reciprocal Rank (MRR@5)** | **0.9000** | **0.7444** | **+0.1556** | **100w (+15.56 pp)** |
| **Rank 1 Hits** | **13 / 15 (86.7%)** | **10 / 15 (66.7%)** | **+3 questions** | **100w (+20.0 pp)** |
| **Rank 2 Hits** | 1 / 15 (Q13) | 3 / 15 (Q2, Q13, Q15) | -2 questions | — |
| **Rank 3 Hits** | 0 / 15 | 1 / 15 (Q1) | -1 question | — |
| **Average Matched Cosine Score** | **0.6302** | **0.5701** | **+0.0601** | **100w (+6.01 points)** |
| **Retrieval Failures (Misses)** | 1 / 15 (Q1) | 1 / 15 (Q10) | 0 | Distinct failure modes |

### Complete Per-Query Empirical Results

The table below shows the exact measured matched ranks and cosine similarity scores for both chunk sizes:

| Question ID | Target Page | 100-Word Rank | 100-Word Cosine Score | 300-Word Rank | 300-Word Cosine Score | Score Delta ($\Delta_{\text{100w} - \text{300w}}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | Page 8 | **Miss** | — | **Rank 3** | 0.5494 | — |
| **Q2** | Page 9 | **Rank 1** | 0.6035 | **Rank 2** | 0.4818 | **+0.1217** |
| **Q3** | Page 10 | **Rank 1** | 0.6443 | **Rank 3** | 0.5924 | **+0.0519** |
| **Q4** | Page 10 | **Rank 1** | 0.5962 | **Rank 1** | 0.5328 | **+0.0634** |
| **Q5** | Page 11 | **Rank 1** | 0.4968 | **Rank 1** | 0.5440 | -0.0472 |
| **Q6** | Page 13 | **Rank 1** | 0.6283 | **Rank 1** | 0.4637 | **+0.1646** |
| **Q7** | Page 15 | **Rank 1** | 0.7716 | **Rank 1** | 0.6065 | **+0.1651** |
| **Q8** | Page 16 | **Rank 1** | 0.6546 | **Rank 1** | 0.6022 | **+0.0524** |
| **Q9** | Page 18 | **Rank 1** | 0.5258 | **Rank 1** | 0.5377 | -0.0119 |
| **Q10** | Page 21 | **Rank 1** | 0.5786 | **Miss** | — | — |
| **Q11** | Page 24 | **Rank 1** | 0.7432 | **Rank 1** | 0.7148 | **+0.0284** |
| **Q12** | Page 37 | **Rank 1** | 0.6057 | **Rank 1** | 0.5574 | **+0.0483** |
| **Q13** | Page 38 | **Rank 2** | 0.6541 | **Rank 2** | 0.5901 | **+0.0640** |
| **Q14** | Page 31 | **Rank 1** | 0.6551 | **Rank 1** | 0.6306 | **+0.0245** |
| **Q15** | Page 33 | **Rank 1** | 0.6652 | **Rank 2** | 0.5780 | **+0.0872** |
| **Average of Hits** | — | **Rank 1.07** | **0.6302** | **Rank 1.43** | **0.5701** | **+0.0601** |

---

### In-Depth Analysis: Which Worked Better and Why?

**The 100-word chunk size worked decisively better than the 300-word chunk size.**

While both configurations matched an identical Recall@5 of 93.33% (14 out of 15 questions retrieved), the 100-word chunks delivered superior retrieval quality across three critical dimensions:

1. **Superior Precision and MRR (+15.56%)**:
   - For 100-word chunks, **13 of the 14 retrieved questions appeared at Rank 1** (86.7% of all questions, 92.9% of hits).
   - For 300-word chunks, only **10 appeared at Rank 1** (66.7% of all questions), with 3 pushed to Rank 2 and 1 to Rank 3.
   - In production RAG, higher rank directly translates to higher generation accuracy because LLMs suffer from "lost-in-the-middle" attention degradation when the true evidence is buried below distractors.

2. **Semantic Density vs. Vector Dilution**:
   - Dense bi-encoders compress an entire text segment into a fixed 384-dimensional vector by mean-pooling the contextualized token embeddings.
   - In **100-word chunks**, the passage focus is sharp and dense. Sentences describing specific concepts (e.g., Big Data 4 Vs on Page 13, Data Warehouse vs. Lake on Page 15) occupy a large fraction of the chunk. Consequently, the cosine similarity between the query and the chunk is high (averaging **0.6302**).
   - In **300-word chunks**, the same factual sentence is surrounded by 200+ words of extraneous narrative, commentary, or adjacent topics. The embedding vector averages all these signals together. This **dilutes** the specific factual representation, causing the cosine similarity to drop by an average of **0.0601** (e.g., Q6 dropped from 0.6283 to 0.4637, Q7 dropped from 0.7716 to 0.6065). In tight rank competitions, this dilution allows generic topical passages to outscore the true source.

3. **Granularity Trade-off**:
   - The drawback of 100-word chunks is vulnerability to boundary cuts: if a complex answer requires sentences separated by 100+ words, a small chunk cannot encompass the full answer (as seen in Q1).
   - However, for factual QA retrieval, 100-word chunks provide significantly higher signal-to-noise ratio and retrieval accuracy.

---

## 5. Detailed Breakdown of the Two Retrieval Failures

### Failure Case 1: Question 1 under 100-word chunks
- **Question**: *"What three components explain how modern AI systems can exhibit intelligent behaviour?"*
- **Gold Source**: Page 8 (*"It is these three components of advanced algorithms, data and computing power, that explain how machines can exhibit intelligent behaviour and why AI has suddenly exploded into our everyday lives."*)
- **Top 5 Retrieved Pages**: `[Page 34, Page 7, Page 35, Page 21, Page 7]`
- **Gold Page Outcome**: Page 8 did not appear in the top 5 (ranked #7).

#### Root Cause Analysis
1. **Chunk Boundary Splitting**: Under 100-word chunking, Page 8 was split into multiple chunks. The phrase *"three components"* and the subsequent text introducing computing power were separated from the deeper discussion of compute infrastructure.
2. **Dense Model Bias for Broad Semantic Themes**: The query contains the high-level semantic keywords *"modern AI systems"* and *"intelligent behaviour"*. The bi-encoder mapped these phrases to generic introductory discussions of AI definitions and applications:
   - *Page 34* (score 0.5638): Discusses the future impact of AI on the travel experience.
   - *Page 7* (score 0.5479): Quotes the Oxford English Dictionary definition of Artificial Intelligence as *"the capacity of computers... to exhibit intelligent behaviour"*.
   - *Page 35* (score 0.5319): Defines Narrow AI automating business tasks.
   Because Page 7 contained the exact phrase *"to exhibit intelligent behaviour"*, its vector dot product outscored Page 8's chunk. The dense model lacked the capacity to recognize that the user specifically requested the **three components** rather than the definition of intelligent behavior.

#### Concrete Technical Fixes
1. **BM25 Lexical Hybrid Search**: The phrase *"three components"* along with the co-occurrence of *"algorithms"*, *"data"*, and *"computing power"* is lexically distinctive. In BM25, rare terms carry massive Inverse Document Frequency (IDF). BM25 would score Page 8 near the top, and combining dense + sparse scores via Reciprocal Rank Fusion (RRF) would promote Page 8 into Rank 1.
2. **Context Window Expansion (300-word Chunks)**: Evaluating with 300-word chunks grouped the three components together with the surrounding introductory narrative. This boosted Page 8 into the top 5 at **Rank 3** (score 0.5494).

---

### Failure Case 2: Question 10 under 300-word chunks
- **Question**: *"What are the three capability-based categories of AI?"*
- **Gold Source**: Page 21 (*"An AI system delivers either artificial narrow intelligence (ANI), artificial general intelligence (AGI), or artificial super intelligence (ASI)."*)
- **Top 5 Retrieved Pages**: `[Page 20, Page 35, Page 7, Page 35, Page 31]`
- **Gold Page Outcome**: Page 21 did not appear in the top 5 (ranked #8).

#### Root Cause Analysis
1. **Vector Semantic Dilution**: In 300-word chunking, Page 21's single sentence defining ANI, AGI, and ASI was subsumed within a large chunk that also discussed machine learning learning types, deep learning neural networks, and cloud computing. The resulting 300-word embedding vector reflected general AI learning mechanisms rather than the taxonomy of capabilities.
2. **Topical Drift to Glossary & Compute Pages**: Because the query asked for *"categories of AI"*, general terminology pages scored higher:
   - *Page 20* (score 0.5627): Discussed Algorithms + Data + Compute.
   - *Page 35* (score 0.5345): The report's formal Annex on AI Terminology.
   - *Page 7* (score 0.5327): General AI definition.
   Page 21's diluted vector fell behind these competitors and was excluded from the top 5.

#### Concrete Technical Fixes
1. **Smaller Chunk Size**: In 100-word chunks, the ANI/AGI/ASI sentence formed its own focused chunk, and retrieval succeeded at **Rank 1** (score 0.5786).
2. **Cross-Encoder Re-Ranking**: If initial retrieval fetches the top 20 candidates, a Cross-Encoder (e.g., `ms-marco-MiniLM-L-6-v2`) performs token-to-token cross-attention between the query and the candidate passages. The cross-encoder directly correlates *"three capability-based categories"* with the enumeration of *"ANI, AGI, ASI"*, promoting Page 21 to the top regardless of surrounding text.
3. **Query Expansion**: Prepending domain synonyms to the query (e.g., *"capability-based categories: narrow, general, super intelligence, ANI, AGI, ASI"*) bridges the vocabulary gap.

---

## 6. Conceptual Mastery: BM25 and Re-Rankers (In Our Own Words)

### A. What BM25 Adds to Embedding Search

Dense embedding models (bi-encoders) are celebrated for their ability to capture semantic synonyms and conceptual intent. If a user asks *"How can an AI invent a convincing false answer?"*, an embedding model reliably matches *"AI hallucinations"* even though they share no common vocabulary.

However, dense embeddings suffer from the **"soft match" trap**:
1. **Insensitivity to Low-Frequency Keywords**: In dense vector space, specific acronyms (*TPU, GPU, ANI, AGI, ANN*), model versions (*GPT-4*), historical dates (*1950*), statistical numbers (*34%, 2.6 billion, 13%*), and proper nouns (*Alan Turing, Randstad, Will Cathcart*) are smoothed out. The bi-encoder often maps them into general topical clusters (e.g. "computing hardware" or "survey results") rather than matching the exact entity.
2. **Topical Over-Smoothing**: Two passages that discuss AI in a general sense can have nearly identical cosine similarities to a query, even if only one contains the exact operational entity requested.

#### How BM25 Solves This
**BM25 (Best Matching 25)** is a probabilistic term-matching ranking function derived from the Okapi information retrieval model:
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- **Inverse Document Frequency (IDF)**: $\text{IDF}(t) = \ln \left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$. Terms that appear in few documents (e.g. *"Randstad"*, *"TPU"*, *"veracity"*) receive exponentially higher weight than ubiquitous terms (e.g. *"AI"*, *"computer"*).
- **Term Frequency Saturation ($k_1 \approx 1.2–2.0$)**: Prevents keyword-stuffing from disproportionately inflating a document's score; additional occurrences of a query term yield diminishing returns.
- **Document Length Normalization ($b \approx 0.75$)**: Penalizes verbose documents so that a long document does not score higher simply because it contains more words.

#### The Power of Hybrid Search
By uniting BM25 and Dense Embeddings using **Reciprocal Rank Fusion (RRF)**:
$$\text{RRF Score}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{1}{60 + \text{Rank}_m(d)}$$
The retrieval engine achieves the best of both worlds:
- **BM25** guarantees precision on exact keywords, technical acronyms, and statistical figures.
- **Dense Embeddings** guarantee recall on conceptual paraphrases and synonym-heavy queries.

---

### B. What a Re-Ranker Does

In state-of-the-art information retrieval, systems employ a **two-stage funnel**:
1. **Stage 1 (Retrieval)**: Fast candidate generation over thousands of chunks ($K \approx 20–50$).
2. **Stage 2 (Re-Ranking)**: High-precision scoring over the candidate pool ($K \to 3–5$).

```
[Corpus: 10,000 Chunks]
           │
           ▼  Stage 1: Bi-Encoder (NumPy / FAISS / BM25) — Latency < 10ms
   [Top-25 Candidates]
           │
           ▼  Stage 2: Cross-Encoder (Re-Ranker) — Latency ~ 20ms
     [Top-3 Passages]
           │
           ▼  LLM Context Injection
     [Final Grounded Answer]
```

#### Why First-Stage Bi-Encoders are Fundamentally Limited
First-stage vector search uses a **Bi-Encoder** architecture:
$$\text{Query Vector } \mathbf{q} = \text{Encoder}(Q), \quad \text{Passage Vector } \mathbf{d} = \text{Encoder}(P), \quad \text{Score} = \mathbf{q} \cdot \mathbf{d}$$
Because the query and the document are encoded independently into isolated vectors:
- **No Token-to-Token Cross-Attention**: Tokens in the query cannot interact with tokens in the passage during representation encoding.
- **Loss of Syntactic Nuances**: Critical nuances—such as word order, conditionality, qualifiers, and negation (e.g., *"traditional programming is not explicitly coded"*)—are compressed and lost in the single pooled vector.

#### How a Cross-Encoder Re-Ranker Works
A **Re-Ranker (Cross-Encoder)** feeds the query and candidate passage simultaneously into the transformer model:
$$\text{Input} = \text{[CLS]} + \text{Query Tokens} + \text{[SEP]} + \text{Passage Tokens} + \text{[SEP]}$$
$$\text{Score} = \text{Sigmoid}(\mathbf{W} \cdot \text{Transformer}(\text{Input})_{[\text{CLS}]})$$

Because every query token directly attends to every passage token across all transformer layers:
1. **True Semantic Verification**: The model verifies whether the passage actually answers the question or merely discusses the same high-level topic.
2. **Negation and Constraint Resolution**: The cross-encoder accurately identifies negations and restrictive clauses.
3. **Elimination of False Positives**: Irrelevant chunks that had accidentally high vector dot products are demoted to the bottom of the list.

#### Computational Trade-off
- A Cross-Encoder has computational complexity $O(M \cdot L^2)$ where $L = |Q| + |P|$. Running a cross-encoder across 10,000 chunks would take seconds per query, making it unusable for real-time search.
- However, running a Bi-Encoder first to filter 10,000 chunks down to 25 candidates, and then applying a Cross-Encoder to re-rank those 25 candidates, achieves **sub-30ms latency with maximum precision**.

---

## 7. End-to-End LLM Answer Generation (Citation Grounding)

The system supports document-grounded answer generation through any OpenAI-compatible API (`OPENAI_API_KEY`), using a strict grounding prompt:

```text
System Prompt:
You are a strict, factual Question-Answering assistant.
Answer the user's question using ONLY the retrieved source passages below.
Always cite the exact PDF Page number(s) for every fact you state.
If the passages do not contain enough information to answer the question,
clearly state: 'The provided document passages do not contain this information.'
```

### Verified Query Demonstration

#### Example Query: *"What are the four Vs of Big Data?"*
- **Retrieval Engine**: NumPy Cosine Similarity (`chunk-words 100`, `top-k 5`)
- **Top Retrieved Passage**:
  > `[Rank 1 | Score: 0.6461 | Source Page: 13 | Chunk: p13_w400]`  
  > *"These extremely large and very complex datasets are often called ‘Big Data’ and are characterised by their volume, velocity, variety and veracity, commonly referred to as the 4 V ’s."*
- **Generated LLM Answer**:
  > According to the report, Big Data is characterised by the **4 V's**:
  > 1. **Volume**: The massive scale of data generated globally.
  > 2. **Velocity**: The rapid speed at which new data is created and transmitted.
  > 3. **Variety**: The diverse types of structured and unstructured data formats.
  > 4. **Veracity**: The accuracy, truthfulness, and reliability of the data.  
  > *(Source: PDF Page 13)*

---

## 8. Verification & Reproducibility Guide

All experiments and results documented in this report can be verified through the following reproducible CLI commands:

```powershell
# 1. Inspect source PDF extraction statistics (41 pages, 99.5k characters)
python check_pdf.py

# 2. Extract text and generate 100-word and 300-word chunks (saved to artifacts/)
python src/chunking.py

# 3. Compute normalized dense embeddings (saved to artifacts/)
python src/embeddings.py

# 4. Run the comprehensive retrieval evaluation benchmark
python rag.py --evaluate
# (or directly: python evaluate.py)

# 5. Execute vector search on specific test queries
python rag.py --ask "What are the four Vs of Big Data?" --chunk-words 100
python rag.py --ask "How does a TPU differ from a GPU?" --chunk-words 300

# 6. Optional: Generate LLM-grounded answers with citations
$env:OPENAI_API_KEY = "your-api-key"
$env:LLM_MODEL = "gpt-4o-mini"
python rag.py --ask "What is Explainable AI?"
```

All evaluation metrics and candidate logs are serialized in:
- `results/summary.json`
- `results/evaluation_results.csv`
- `results/evaluation_detailed.json`
- `results/chunk_100_evaluation.json`
- `results/chunk_300_evaluation.json`
