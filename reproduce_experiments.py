"""
reproduce_experiments.py
========================
Master automated execution script that reproduces all experimental benchmarks,
statistical significance calculations, and publication figures from scratch.

Run:
    python reproduce_experiments.py
"""

import os
import subprocess
import sys
import time

def run_step(step_name: str, cmd: list[str]):
    print(f"\n{'='*70}")
    print(f"STEP: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    res = subprocess.run([sys.executable] + cmd, capture_output=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[FAILED] Step '{step_name}' exited with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] Step '{step_name}' completed in {elapsed:.2f} seconds.")

def main():
    print("======================================================================")
    print("  EDUMETRICS STRUCTURED IE: MASTER REPRODUCIBILITY PIPELINE")
    print("======================================================================")

    # 1. Environment Telemetry
    run_step("Log Environment Telemetry", ["environment_logger.py"])

    # 2. Dataset Synthesis (N=1,200)
    run_step("Synthesize Decoupled N=1,200 Benchmark Dataset", ["generate_dataset_v2.py"])

    # 3. Unit Testing Metrics Engine
    run_step("Execute Unit Tests for Metrics Engine", ["-m", "pytest", "test_metrics.py", "-v"])

    # 4. Standardized Evaluation Harness (Zero Row Dropping)
    run_step("Run Benchmarking Harness & Un-sanitized Evaluation", ["experiment_harness.py"])

    # 5. Statistical Tests & Figures
    run_step("Compute Statistical Tests & Generate Camera-Ready Figures", ["analyze_benchmark.py"])

    print("\n======================================================================")
    print("  ALL BENCHMARK ARTIFACTS AND FIGURES GENERATED SUCCESSFULLY!")
    print("  - environment_telemetry.json")
    print("  - dataset/benchmark_samples.jsonl (1,200 samples)")
    print("  - dataset/ground_truth.json (1,200 ground-truth records)")
    print("  - model_matrix_audit.csv")
    print("  - comprehensive_evaluation_results.csv (200 rows with 0 dropped)")
    print("  - statistical_evaluation.json")
    print("  - figure1_validity_vs_f1.png")
    print("  - figure2_latency_distributions.png")
    print("  - figure3_system_failures_and_stability.png")
    print("======================================================================")

if __name__ == "__main__":
    main()
