"""
metrics_engine.py
=================
Comprehensive, mathematically rigorous evaluation engine for Structured
Information Extraction (IE).

Metrics:
1. Syntactic Validity: Strict JSON decode success rate.
2. Schema Conformance: Structural alignment with expected nested Pydantic model.
3. Field-Level Information Extraction:
   - Student Name (normalized string match)
   - Student ID (exact match)
   - Academic Semester (exact integer match)
   - Attendance Percentage (tolerant float match +/- 0.5% or dual null)
   - Academic Status (exact enum match)
   - Subject-Score Extraction: Precision, Recall, F1 against ground truth dictionary.
4. Exact Match (EM): All fields concurrently matching ground truth.
5. Systems Telemetry: Wall-clock latency (mean, median, p95, std), throughput (tok/s),
   and memory tracking.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
from pydantic import BaseModel, Field, ValidationError

# Canonical Subject mapping for alias resolution
SUBJECT_SYNONYMS = {
    "math": "Mathematics",
    "mathematics": "Mathematics",
    "calc": "Mathematics",
    "calculus": "Mathematics",
    "advanced math": "Mathematics",
    "physics": "Physics",
    "phys": "Physics",
    "general physics": "Physics",
    "applied physics": "Physics",
    "chemistry": "Chemistry",
    "chem": "Chemistry",
    "organic chem": "Chemistry",
    "general chemistry": "Chemistry",
    "computer science": "Computer Science",
    "compsci": "Computer Science",
    "cs": "Computer Science",
    "data structures": "Computer Science",
    "programming": "Computer Science",
    "biology": "Biology",
    "bio": "Biology",
    "molecular biology": "Biology",
    "biosciences": "Biology",
    "english literature": "English Literature",
    "english": "English Literature",
    "eng lit": "English Literature",
    "academic writing": "English Literature"
}

def normalize_subject_name(raw: str) -> str:
    cleaned = raw.strip().lower()
    return SUBJECT_SYNONYMS.get(cleaned, raw.strip())

# Target Schema Pydantic Specifications
class StudentProfile(BaseModel):
    name: str
    student_id: str

class AcademicRecords(BaseModel):
    scores: Dict[str, int]
    semester: int
    attendance_percentage: Optional[float] = None

class ExtractedPayload(BaseModel):
    student: StudentProfile
    academic_records: AcademicRecords
    status: str = Field(pattern=r"^(PASSED|PROBATION|FAILED|WITHDRAWN|INCOMPLETE)$")

def extract_json_block(text: str) -> str:
    """Robustly extracts JSON substring, stripping markdown fences or stray commentary."""
    text = text.strip()
    # Match markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Match outermost curly braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    
    return text

def evaluate_json_syntax(raw_output: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Evaluates whether the raw model output can be parsed as valid JSON."""
    if not raw_output or "CRASH" in raw_output or "TIMEOUT" in raw_output or raw_output.startswith("ENGINE_"):
        return False, None, "ENGINE_FAILURE"
    
    clean_str = extract_json_block(raw_output)
    try:
        parsed = json.loads(clean_str)
        if isinstance(parsed, dict):
            return True, parsed, None
        return False, None, "Root is not a JSON object"
    except Exception as e:
        return False, None, str(e)

def evaluate_schema_conformance(parsed_json: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Evaluates whether the parsed JSON satisfies the nested Pydantic schema."""
    if parsed_json is None:
        return False, "No JSON to validate"
    try:
        ExtractedPayload.model_validate(parsed_json)
        return True, None
    except ValidationError as ve:
        return False, str(ve)
    except Exception as e:
        return False, str(e)

def evaluate_extraction_accuracy(extracted: Optional[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes rigorous field-level Precision, Recall, F1, and Exact Match.
    """
    if extracted is None:
        return {
            "name_match": 0.0,
            "id_match": 0.0,
            "semester_match": 0.0,
            "attendance_match": 0.0,
            "status_match": 0.0,
            "scores_precision": 0.0,
            "scores_recall": 0.0,
            "scores_f1": 0.0,
            "exact_match": 0.0
        }
    
    # 1. Student Name (case-insensitive trimmed match)
    ext_name = extracted.get("student", {}).get("name", "").strip().lower()
    gt_name = ground_truth.get("student", {}).get("name", "").strip().lower()
    name_match = 1.0 if ext_name == gt_name and gt_name != "" else 0.0
    
    # 2. Student ID (exact case-insensitive match)
    ext_id = str(extracted.get("student", {}).get("student_id", "")).strip().upper()
    gt_id = str(ground_truth.get("student", {}).get("student_id", "")).strip().upper()
    id_match = 1.0 if ext_id == gt_id and gt_id != "" else 0.0
    
    # 3. Semester (integer match)
    ext_sem = extracted.get("academic_records", {}).get("semester")
    gt_sem = ground_truth.get("academic_records", {}).get("semester")
    semester_match = 1.0 if ext_sem == gt_sem and gt_sem is not None else 0.0
    
    # 4. Attendance (tolerant numeric match within 0.5% or dual null)
    ext_att = extracted.get("academic_records", {}).get("attendance_percentage")
    gt_att = ground_truth.get("academic_records", {}).get("attendance_percentage")
    if ext_att is None and gt_att is None:
        attendance_match = 1.0
    elif ext_att is not None and gt_att is not None:
        try:
            attendance_match = 1.0 if abs(float(ext_att) - float(gt_att)) <= 0.5 else 0.0
        except (ValueError, TypeError):
            attendance_match = 0.0
    else:
        attendance_match = 0.0
        
    # 5. Status (exact enum string match)
    ext_status = str(extracted.get("status", "")).strip().upper()
    gt_status = str(ground_truth.get("status", "")).strip().upper()
    status_match = 1.0 if ext_status == gt_status and gt_status != "" else 0.0
    
    # 6. Scores (Subject-Value pairs)
    ext_scores_raw = extracted.get("academic_records", {}).get("scores", {})
    gt_scores_raw = ground_truth.get("academic_records", {}).get("scores", {})
    
    # Normalize keys in extracted scores
    normalized_ext_scores = {}
    if isinstance(ext_scores_raw, dict):
        for k, v in ext_scores_raw.items():
            norm_k = normalize_subject_name(str(k))
            try:
                normalized_ext_scores[norm_k] = int(v)
            except (ValueError, TypeError):
                pass
                
    normalized_gt_scores = {normalize_subject_name(k): int(v) for k, v in gt_scores_raw.items()}
    
    # Compute set intersection for Key-Value pairs
    tp = 0
    fp = 0
    for k, v in normalized_ext_scores.items():
        if k in normalized_gt_scores and normalized_gt_scores[k] == v:
            tp += 1
        else:
            fp += 1
            
    fn = 0
    for k, v in normalized_gt_scores.items():
        if k not in normalized_ext_scores or normalized_ext_scores[k] != v:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(normalized_gt_scores) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 7. Exact Match: All attributes must be strictly correct
    exact_match = 1.0 if (
        name_match == 1.0 and
        id_match == 1.0 and
        semester_match == 1.0 and
        attendance_match == 1.0 and
        status_match == 1.0 and
        precision == 1.0 and
        recall == 1.0
    ) else 0.0
    
    return {
        "name_match": name_match,
        "id_match": id_match,
        "semester_match": semester_match,
        "attendance_match": attendance_match,
        "status_match": status_match,
        "scores_precision": round(precision, 4),
        "scores_recall": round(recall, 4),
        "scores_f1": round(f1, 4),
        "exact_match": exact_match
    }

def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates empirical mean and half-width of (1 - alpha) confidence interval."""
    arr = np.array(data, dtype=float)
    n = len(arr)
    if n < 2:
        return float(np.mean(arr)) if n == 1 else 0.0, 0.0
    mean = float(np.mean(arr))
    sem = stats.sem(arr)
    h = float(sem * stats.t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, h
