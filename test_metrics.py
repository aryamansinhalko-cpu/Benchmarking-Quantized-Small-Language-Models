"""
test_metrics.py
===============
Unit tests for metrics_engine.py with hand-checked deterministic test cases.
Run with:
    python -m pytest test_metrics.py -v
"""

import pytest
from metrics_engine import (
    evaluate_json_syntax,
    evaluate_schema_conformance,
    evaluate_extraction_accuracy,
    normalize_subject_name,
    calculate_confidence_interval
)

def test_subject_normalization():
    assert normalize_subject_name("Calc") == "Mathematics"
    assert normalize_subject_name("advanced math") == "Mathematics"
    assert normalize_subject_name("CompSci") == "Computer Science"
    assert normalize_subject_name("Programming") == "Computer Science"
    assert normalize_subject_name("Bio") == "Biology"
    assert normalize_subject_name("Phys") == "Physics"
    assert normalize_subject_name("UnknownSubject") == "UnknownSubject"

def test_json_syntax_validity():
    # 1. Clean JSON
    valid, obj, err = evaluate_json_syntax('{"name": "Aarav", "score": 90}')
    assert valid is True
    assert obj == {"name": "Aarav", "score": 90}
    assert err is None

    # 2. Markdown fenced JSON
    fenced = """```json
    {
      "name": "Priya",
      "status": "PASSED"
    }
    ```"""
    valid, obj, err = evaluate_json_syntax(fenced)
    assert valid is True
    assert obj["name"] == "Priya"

    # 3. Conversational wrapper with embedded JSON
    conversational = 'Here are your results: {"name": "Rohan", "score": 82}. Hope this helps!'
    valid, obj, err = evaluate_json_syntax(conversational)
    assert valid is True
    assert obj["name"] == "Rohan"

    # 4. Broken JSON
    broken = '{"name": "Sneha", "score": 85,'
    valid, obj, err = evaluate_json_syntax(broken)
    assert valid is False
    assert obj is None
    assert err is not None

    # 5. Server Crash / Engine failure
    valid, obj, err = evaluate_json_syntax("SERVER_CRASH")
    assert valid is False
    assert err == "ENGINE_FAILURE"

def test_schema_conformance():
    # Valid payload matching Pydantic schema
    valid_payload = {
        "student": {"name": "Aarav Sharma", "student_id": "24BCE1001"},
        "academic_records": {
            "scores": {"Mathematics": 92, "Physics": 88},
            "semester": 4,
            "attendance_percentage": 94.5
        },
        "status": "PASSED"
    }
    conformant, err = evaluate_schema_conformance(valid_payload)
    assert conformant is True
    assert err is None

    # Missing status
    missing_status = {
        "student": {"name": "Aarav Sharma", "student_id": "24BCE1001"},
        "academic_records": {
            "scores": {"Mathematics": 92},
            "semester": 4
        }
    }
    conformant, err = evaluate_schema_conformance(missing_status)
    assert conformant is False
    assert "status" in err

    # Invalid enum value
    bad_enum = {
        "student": {"name": "Aarav Sharma", "student_id": "24BCE1001"},
        "academic_records": {"scores": {}, "semester": 1},
        "status": "TOP_STUDENT" # not in enum
    }
    conformant, err = evaluate_schema_conformance(bad_enum)
    assert conformant is False

def test_extraction_accuracy_perfect():
    gt = {
        "student": {"name": "Vikram Patel", "student_id": "24BCE2020"},
        "academic_records": {
            "scores": {"Mathematics": 85, "Computer Science": 90},
            "semester": 3,
            "attendance_percentage": 88.0
        },
        "status": "PASSED"
    }
    extracted = {
        "student": {"name": "Vikram Patel", "student_id": "24bce2020"},
        "academic_records": {
            "scores": {"Calc": 85, "CompSci": 90}, # Aliased names
            "semester": 3,
            "attendance_percentage": 88.2 # within 0.5% tolerance
        },
        "status": "PASSED"
    }
    res = evaluate_extraction_accuracy(extracted, gt)
    assert res["name_match"] == 1.0
    assert res["id_match"] == 1.0
    assert res["semester_match"] == 1.0
    assert res["attendance_match"] == 1.0
    assert res["status_match"] == 1.0
    assert res["scores_precision"] == 1.0
    assert res["scores_recall"] == 1.0
    assert res["scores_f1"] == 1.0
    assert res["exact_match"] == 1.0

def test_extraction_accuracy_semantic_divergence():
    """Tests case where JSON is valid but extracted values are hallucinated."""
    gt = {
        "student": {"name": "Anjali Rao", "student_id": "24BCE3030"},
        "academic_records": {
            "scores": {"Biology": 78, "Chemistry": 65},
            "semester": 2,
            "attendance_percentage": None
        },
        "status": "PROBATION"
    }
    # AI extracted valid JSON, but hallucinated wrong student and wrong score
    extracted = {
        "student": {"name": "Wrong Person", "student_id": "24BCE9999"},
        "academic_records": {
            "scores": {"Biology": 10, "Chemistry": 65}, # 1 score wrong
            "semester": 5, # wrong semester
            "attendance_percentage": 90.0 # hallucinated attendance
        },
        "status": "PASSED" # wrong status
    }
    res = evaluate_extraction_accuracy(extracted, gt)
    assert res["name_match"] == 0.0
    assert res["id_match"] == 0.0
    assert res["semester_match"] == 0.0
    assert res["attendance_match"] == 0.0
    assert res["status_match"] == 0.0
    # TP = 1 (Chemistry: 65), FP = 1 (Biology: 10), FN = 1 (Biology: 78)
    assert res["scores_precision"] == 0.5
    assert res["scores_recall"] == 0.5
    assert res["scores_f1"] == 0.5
    assert res["exact_match"] == 0.0

def test_crashed_row_handling():
    gt = {
        "student": {"name": "Rohan Deshmukh", "student_id": "24BCE4040"},
        "academic_records": {"scores": {"Physics": 50}, "semester": 1, "attendance_percentage": None},
        "status": "PASSED"
    }
    res = evaluate_extraction_accuracy(None, gt)
    assert res["exact_match"] == 0.0
    assert res["scores_f1"] == 0.0

def test_confidence_interval():
    data = [1.0, 1.0, 1.0, 1.0, 1.0]
    mean, h = calculate_confidence_interval(data)
    assert mean == 1.0
    assert h == 0.0

    data2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean2, h2 = calculate_confidence_interval(data2)
    assert mean2 == 3.0
    assert h2 > 0.0
