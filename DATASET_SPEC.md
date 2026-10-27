# Dataset Specification: EduMetrics-Structured-IE (v2.0)

## Overview
This corpus was designed to benchmark Small Language Models (SLMs) on complex, multi-sentence Information Extraction (IE) into nested JSON schemas without ground-truth key leakage.

## Design Parameters
* **Corpus Size:** $N = 1,200$ samples.
* **Pseudorandom Seed:** `42` (`random.seed(42)`).
* **Generation Engine:** [`generate_dataset_v2.py`](generate_dataset_v2.py).
* **Storage Format:**
  * Inputs: `dataset/benchmark_samples.jsonl` (line-delimited JSON containing `sample_id` and `raw_text`).
  * Ground Truth: `dataset/ground_truth.json` (keyed by `sample_id`).

## Target Nested Schema Specification
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "student": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "student_id": { "type": "string" }
      },
      "required": ["name", "student_id"],
      "additionalProperties": false
    },
    "academic_records": {
      "type": "object",
      "properties": {
        "scores": {
          "type": "object",
          "additionalProperties": { "type": "integer" }
        },
        "semester": { "type": "integer" },
        "attendance_percentage": { "type": ["number", "integer", "null"] }
      },
      "required": ["scores", "semester"],
      "additionalProperties": false
    },
    "status": {
      "type": "string",
      "enum": ["PASSED", "PROBATION", "FAILED", "WITHDRAWN", "INCOMPLETE"]
    }
  },
  "required": ["student", "academic_records", "status"],
  "additionalProperties": false
}
```

## Challenge Vectors
1. **Numerical Distractors:** Unrelated integers and floats embedded in the text:
   * Office telephone extensions (e.g., `ext. 6643`)
   * Examination halls and room numbers (e.g., `Hall B-12`, `Room 101`)
   * Cohort rankings (e.g., `ranked #14`)
   * Credit units attempted (e.g., `18 credit units`)
   * Calendar years (e.g., `Spring 2024`)
2. **Missing & Optional Data:**
   * 15% of samples intentionally lack attendance records (`null` ground truth).
   * Irregular subject distributions (1 to 4 subjects per student).
3. **Lexical Aliasing & Abbreviations:**
   * Canonical categories mapped from informal abbreviations (e.g., "Calc" / "Advanced Math" $\rightarrow$ `Mathematics`; "CS" / "Programming" $\rightarrow$ `Computer Science`).
   * Varied score notation ("attained 85", "scored 85/100", "grade: 85%", "recieved 85").
4. **Structural Decoupling:**
   * The prompter never provides ground-truth subject keys. The model must perform open-world entity and relation extraction guided strictly by the target schema.
