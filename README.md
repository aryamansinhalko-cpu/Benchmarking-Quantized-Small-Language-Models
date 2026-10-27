# Benchmarking Quantized Small Language Models for Edge-Based Structured Information Extraction: An Empirical Systems Study

An empirical benchmarking framework evaluating quantized Small Language Models (SLMs) under integer compression (INT4 vs. INT8) executing structured information extraction into nested JSON schemas on consumer edge hardware.

---

## 👥 Authors & Affiliation
* **Aryaman Sinha** (`aryaman.sinha.lko@gmail.com`)
* **Nithin Rajan** (`nithinrajan48@gmail.com`)
* **Johith Babu S** (`johithbabu2006@gmail.com`)
* **Ajitesh Mohan** (`ajiteshmn@gmail.com`)

*Department of Computer Science, Vellore Institute of Technology (VIT), Vellore, India.*

---

## 📌 Research Overview

Deploying Small Language Models (SLMs) on consumer edge devices enables low-latency, privacy-preserving data extraction without transmitting personally identifiable information (PII) to centralized cloud infrastructures. However, edge execution requires post-training quantization to satisfy resident memory limits.

While existing benchmarks predominantly evaluate perplexity on unconstrained text, the impact of low-bit integer quantization on deterministic, schema-constrained structured information extraction remains under-characterized. In this study, we benchmark the Meta Llama-3-8B architecture under 4-bit (`Q4_K_M`) and 8-bit (`Q8_0`) quantization on a consumer workstation.

### Core Empirical Findings:
1. **71.3% Latency Reduction:** Aggressive 4-bit quantization reduces mean inference latency from $3.104 \pm 0.266\,\text{s}$ to $0.892 \pm 0.149\,\text{s}$ ($p = 2.46 \times 10^{-30}$), while maintaining 100.0% syntactic JSON validity.
2. **The Syntax-Semantics Discrepancy:** While the 4-bit model achieved 100.0% JSON syntax validity, true field-level extraction $F_1$ score was only **0.510**, and exact match was only **2.0%**. In zero-shot prompting, the model frequently confounds numerical distractors (telephone extensions, examination room numbers, credit counts) with test scores. Validating syntax alone with `json.loads()` severely overestimates operational accuracy.
3. **Hardware Reliability & OOM Accounting:** Under consumer memory limits (16 GB RAM), the 8-bit model experienced a **1.0% out-of-memory (OOM) termination rate** during initialization. In contrast, 4-bit executed with a 0.0% failure rate, establishing INT4 compression as a prerequisite for process stability on 16 GB edge hardware.

---

## 💻 Tested Hardware & Environment

Hardware telemetry was captured automatically via [`environment_logger.py`](environment_logger.py) and stored in [`environment_telemetry.json`](environment_telemetry.json):

* **CPU:** 13th Gen Intel(R) Core(TM) i9-13900HX (24 Physical Cores, 32 Logical Threads)
* **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MB GDDR6 VRAM, Driver 621.01, CUDA 13.5)
* **RAM:** 16.0 GB DDR5 RAM (15.73 GB Available Host Memory)
* **Operating System:** Microsoft Windows 11 Home (64-bit, Build 26300)
* **Python Runtime:** Python 3.14.7 AMD64
* **Inference Backend:** Ollama / `llama.cpp` runtime

---

## 📂 Repository Structure

```
├── IEEE_COMPILER_PAPER (3).pdf               # Compiled IEEE conference research paper
├── ieee_paper_final.tex                      # Overleaf-ready IEEEtran LaTeX source code
│
├── dataset/
│   ├── benchmark_samples.jsonl               # 1,200 multi-sentence texts with numerical distractors
│   └── ground_truth.json                     # Decoupled canonical ground-truth JSON records
│
├── generate_dataset_v2.py                    # Seeded (seed=42) N=1,200 dataset generation script
├── environment_logger.py                     # System telemetry interrogator (CPU, GPU, RAM, OS)
├── environment_telemetry.json                # Recorded hardware specs log
├── metrics_engine.py                         # Multi-dimensional evaluation engine (Syntax, Pydantic, F1)
├── experiment_harness.py                     # Multi-model benchmarking harness & un-sanitized evaluator
├── analyze_benchmark.py                      # Statistical testing (t-test, U-test) and figure generator
├── test_metrics.py                           # Pytest unit test suite for metric validation
├── reproduce_experiments.py                  # One-click master reproduction pipeline
├── demo.py                                   # Interactive live demonstration script for presentations
├── requirements.txt                          # Python dependencies
│
├── comprehensive_evaluation_results.csv      # Un-sanitized 200-row benchmark evaluation results
├── benchmark_summary_metrics.csv             # Aggregated metric summary table
├── model_matrix_audit.csv                    # 13-model availability audit across 3 families
├── statistical_evaluation.json               # Quantitative stats, p-values, and 95% confidence intervals
│
├── figure1_validity_vs_f1.png                # Figure 1: Syntax validity vs. Semantic Extraction F1
├── figure2_latency_distributions.png         # Figure 2: Inference latency distributions (log-scale)
└── figure3_system_failures_and_stability.png # Figure 3: System reliability and OOM crash accounting
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/aryamansinhalko-cpu/Benchmarking-Quantized-Small-Language-Models.git
cd Benchmarking-Quantized-Small-Language-Models
pip install -r requirements.txt
```

### 2. Run the Full Reproduction Pipeline (One-Click)
Execute the master automated pipeline to query telemetry, synthesize the corpus, run unit tests, re-evaluate all runs, and generate camera-ready figures:
```bash
python reproduce_experiments.py
```

### 3. Run Unit Tests
To verify mathematical formulas, subject normalization, and crash detection logic:
```bash
pytest test_metrics.py -v
```

### 4. Interactive Live Demo (For Presentations / Viva)
Step interactively through hardware telemetry, distractor analysis, benchmark data, and the syntax-semantics discrepancy:
```bash
python demo.py
```

---

## 📊 Evaluation Metrics

1. **JSON Syntactic Validity ($V_{\text{json}}$):** Success rate of `json.loads()` on model outputs.
2. **Schema Conformance ($C_{\text{schema}}$):** Structural validation rate using nested Pydantic v2 schemas.
3. **Field-Level Score Extraction ($F_1$):** Harmonic mean of Precision ($P$) and Recall ($R$) on extracted `(Subject, Score)` pairs:
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$
4. **Exact Match Rate ($EM$):** Binary indicator requiring concurrent exact match across all entity keys and values.
5. **Inference Latency & Telemetry:** Wall-clock seconds per query, reporting Mean, Median, P95, and 95% Confidence Intervals via Student's $t$-distribution.
6. **Statistical Significance:** Two-tailed Welch's $t$-test and non-parametric Mann-Whitney $U$ test.

---

## 📈 Benchmark Summary Results

| Metric | Llama-3-8B (4-bit, `Q4_K_M`) | Llama-3-8B (8-bit, `Q8_0`) | Significance / Delta |
| :--- | :---: | :---: | :---: |
| **Total Test Samples** | 100 | 100 | Invariant Count |
| **Hardware Crashes / OOMs** | **0** | **1 (OOM Crash)** | Hardware limit on 16 GB host |
| **Execution Failure Rate (%)** | **0.0%** | **1.0%** | Un-sanitized telemetry |
| **JSON Syntactic Validity (%)** | **100.0%** | **99.0%** | 8-bit penalized by crash |
| **Schema Conformance (%)** | **97.0%** | **99.0%** | Pydantic structural validation |
| **Mean Latency (s)** | **$0.892 \pm 0.149$** | **$3.104 \pm 0.266$** | **71.3% speedup** ($p = 2.46 \times 10^{-30}$) |
| **Median Latency (s)** | **0.810** | **2.950** | Mann-Whitney $U = 98.0$ ($p < 10^{-32}$) |
| **P95 Latency (s)** | **0.900** | **3.490** | **74.2% tail reduction** |
| **Subject Extraction $F_1$** | **$0.510 \pm 0.020$** | **$0.510 \pm 0.020$** | No quantization degradation |
| **Strict Exact Match Rate (%)** | **2.0%** | **2.0%** | Semantic divergence |

---

## 📜 Citation & Academic Use

If you utilize this benchmark, corpus, or methodology in your research, please cite our study:

```bibtex
@inproceedings{sinha2026benchmarking,
  title={Benchmarking Quantized Small Language Models for Edge-Based Structured Information Extraction: An Empirical Systems Study},
  author={Sinha, Aryaman and Rajan, Nithin and Babu S, Johith and Mohan, Ajitesh},
  booktitle={Department of Computer Science, Vellore Institute of Technology (VIT)},
  year={2026}
}
```

---

## ⚖️ License
This project is open-source under the MIT License.
