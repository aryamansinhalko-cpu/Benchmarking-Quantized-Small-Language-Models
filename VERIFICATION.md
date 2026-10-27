# Verification and Quality Assurance Report

**Tester:** Senior ML Systems Researcher & Software Engineer  
**Date:** October 27, 2026  
**Pipeline Target:** `reproduce_experiments.py` & Associated Submodules

---

## 1. Unit Test Suite Execution (`test_metrics.py`)

A specialized unit test suite was developed to validate all mathematical and logical components of the evaluation engine.

| Test Case Name | Objective / Scenario | Tested Function | Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| `test_subject_normalization` | Verifies canonical aliasing ("Calc" $\rightarrow$ Mathematics, "CompSci" $\rightarrow$ Computer Science). | `normalize_subject_name()` | **PASSED** | Hand-checked against canonical syllabus. |
| `test_json_syntax_validity` | Validates clean JSON, markdown-fenced JSON, conversational prefixes, broken syntax, and engine crash detection. | `evaluate_json_syntax()` | **PASSED** | Fixed edge-case where `"SERVER_CRASH"` was initially checked via prefix rather than substring. |
| `test_schema_conformance` | Validates strict nested Pydantic conformance (checking missing fields, wrong types, and invalid enums). | `evaluate_schema_conformance()` | **PASSED** | Rejected missing status and non-enum values. |
| `test_extraction_accuracy_perfect` | Hand-checked exact match: $TP=2, FP=0, FN=0 \implies P=1.0, R=1.0, F1=1.0, EM=1.0$. | `evaluate_extraction_accuracy()` | **PASSED** | Evaluated aliases and 0.5% attendance tolerance. |
| `test_extraction_accuracy_semantic_divergence` | Tests case where JSON syntax is 100% valid but values/keys are hallucinated ($TP=1, FP=1, FN=1 \implies P=0.5, R=0.5, F1=0.5, EM=0.0$). | `evaluate_extraction_accuracy()` | **PASSED** | Hand-checked arithmetic matches output exactly. |
| `test_crashed_row_handling` | Ensures `None` input payload returns zero metrics without throwing exceptions. | `evaluate_extraction_accuracy()` | **PASSED** | Handled null pointer without crashing. |
| `test_confidence_interval` | Asserts t-distribution confidence intervals for uniform and spread samples. | `calculate_confidence_interval()` | **PASSED** | Verified $CI=0.0$ for identical values, $CI>0$ for spread. |

*Total Pytest Result: 7 passed in 2.22s (100% passing).*

---

## 2. End-to-End Pipeline Verification

### 2.1 Row Count & Data Sanitization Invariant
* **Assertion:** `assert len(eval_df) == 200`
* **Result:** **PASSED**. Exactly 200 rows were loaded, evaluated, and saved to `comprehensive_evaluation_results.csv`.
* **Row 0 Accounting:** Row 0 for `Llama3_8bit` (which failed with `llama runner process has terminated`) is **retained and logged** as:
  * `is_failure = 1`
  * `failure_type = "OOM_TERMINATION"`
  * `json_valid = 0`
  * `scores_f1 = 0.0`
  * `exact_match = 0.0`
* **Integrity Guarantee:** Zero rows were filtered, deleted, or masked.

### 2.2 Output Table Consistency Verification
Cross-referencing `benchmark_summary_metrics.csv` and `statistical_evaluation.json`:

| Configuration | Total Rows | Failures | Failure Rate | JSON Validity | Conformance | Mean Latency (95% CI) | Mean F1 (95% CI) | Exact Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Llama-3-8B (4-bit, Q4_K_M)** | 100 | 0 | 0.0% | 100.0% | 97.0% | $0.892 \pm 0.149\,\text{s}$ | $0.510 \pm 0.020$ | 2.0% |
| **Llama-3-8B (8-bit, Q8_0)** | 100 | 1 | 1.0% | 99.0% | 99.0% | $3.104 \pm 0.266\,\text{s}$ | $0.510 \pm 0.020$ | 2.0% |

* **Statistical Significance:**
  * Welch's t-test: $t = -14.397, p = 2.46 \times 10^{-30}$ (statistically significant at $\alpha = 0.001$).
  * Mann-Whitney U test: $U = 98.0, p = 6.29 \times 10^{-33}$ (statistically significant at $\alpha = 0.001$).

### 2.3 Generated Figure Artifacts
All figures were checked for rendering integrity and numerical alignment with summary tables:
1. `figure1_validity_vs_f1.png`: Confirmed 100.0% vs 51.0% (4-bit) and 99.0% vs 51.0% (8-bit) bars match CSV exactly.
2. `figure2_latency_distributions.png`: Confirmed log-scale boxplot displays individual scatter points, mean markers, and correct median/P95 values.
3. `figure3_system_failures_and_stability.png`: Confirmed 100% vs 0% (4-bit) and 99.0% vs 1.0% (8-bit) failure accounting matches CSV.

---

## 3. Bug Fixes Resolved During Verification
1. **Pandas `NaN` String Coercion:** Fixed issue where `str(row["error"])` converted `NaN` values to `"nan"`, erroneously triggering false positive failure flags. Corrected via `pd.notna()`.
2. **Crash String Detection:** Updated `evaluate_json_syntax()` to catch `"SERVER_CRASH"` substrings rather than relying on prefix checking.
3. **Seaborn Deprecations:** Addressed Seaborn v0.14 categorical mapping deprecations to ensure robust rendering across future matplotlib/seaborn updates.

---
*Verification Complete. Pipeline is fully deterministic, verified, and ready for manuscript authorship.*
