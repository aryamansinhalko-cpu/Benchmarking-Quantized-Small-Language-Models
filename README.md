# EduMetrics Structured Information Extraction (IE) Benchmark

Empirical evaluation and benchmarking framework for Small Language Models (SLMs) under integer quantization (INT4 vs INT8) executing structured information extraction into nested JSON schemas.

## Overview
This repository provides an end-to-end reproducible benchmarking pipeline comparing model performance, syntactic validity, field-level extraction accuracy (Precision, Recall, F1, Exact Match), and hardware failure rates across quantized models deployed in edge environments.

## System Architecture

```
├── AUDIT.md                             # Phase 1 deep research and methodological audit
├── DATASET_SPEC.md                      # Full technical specification of the N=1,200 corpus
├── VERIFICATION.md                      # Phase 3 verification report and unit test results
├── README.md                            # Repository documentation (this file)
├── requirements.txt                     # Python dependencies
│
├── dataset/
│   ├── benchmark_samples.jsonl          # 1,200 input texts with embedded numerical distractors
│   └── ground_truth.json                # Decoupled canonical ground-truth JSON payloads
│
├── generate_dataset_v2.py               # Seeded, reproducible N=1,200 synthesis generator
├── environment_logger.py                # Automated hardware/OS/Ollama environment interrogator
├── metrics_engine.py                    # Multi-dimensional evaluation engine (syntax + semantics)
├── experiment_harness.py                # Standardized benchmarking & un-sanitized evaluator
├── analyze_benchmark.py                 # Statistical testing (t-test, Mann-Whitney U) & figures
├── test_metrics.py                      # Comprehensive pytest unit test suite
├── reproduce_experiments.py             # One-click master reproduction pipeline
│
├── environment_telemetry.json           # Machine hardware and environment telemetry log
├── model_matrix_audit.csv               # 3-family model availability and skip reason accounting
├── comprehensive_evaluation_results.csv # Full 200-row benchmark results (zero rows dropped)
├── benchmark_summary_metrics.csv        # Aggregated performance and reliability metrics
├── statistical_evaluation.json          # Statistical significance tests & 95% confidence intervals
│
├── figure1_validity_vs_f1.png           # Camera-ready Figure 1: Syntax vs Extraction F1 gap
├── figure2_latency_distributions.png    # Camera-ready Figure 2: Log-scale latency distributions
└── figure3_system_failures_and_stability.png # Camera-ready Figure 3: OOM fault accounting
```

## Quick Start & One-Click Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete automated pipeline
python reproduce_experiments.py
```

## Core Findings
1. **The Syntax-Semantics Illusion:** Syntactic JSON validity is high for both 4-bit (100.0%) and 8-bit (99.0%), but true semantic extraction F1 score drops to **0.51**, with an exact match of only **2.0%**. Measuring syntax alone masks substantial extraction hallucinations.
2. **Inference Latency Advantage:** Aggressive 4-bit quantization reduces mean inference latency by **71.3%** (from $3.104 \pm 0.266\,\text{s}$ to $0.892 \pm 0.149\,\text{s}$), a statistically significant difference ($p < 10^{-30}$).
3. **Hardware Fault Accounting:** Under consumer memory limits (16 GB RAM), the 8-bit model experienced a **1.0% out-of-memory termination rate** during initialization, proving that model compression is vital for edge process stability.
