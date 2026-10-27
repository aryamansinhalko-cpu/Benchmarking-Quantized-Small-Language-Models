"""
generate_dataset_v2.py
======================
Generates a realistic, rigorous benchmark dataset (N >= 1,000) for evaluating
Small Language Models (SLMs) on structured Information Extraction (IE) into
nested JSON schemas.

Design Principles:
1. Multi-sentence naturalistic text (advising notes, transcript reviews, grade memos).
2. Numerical distractors: credit hours, room numbers, term years, cohort rank, phone ext.
3. Missing/Optional fields: irregular subject counts (1-4 subjects), optional attendance.
4. Lexical variance & abbreviations: "Calc" vs "Calculus", "CompSci" vs "Computer Science", etc.
5. Nested schema output target (student info, academic records, status enum).
6. Complete decoupling of input text from ground truth (no key leakage).
7. Strict reproducibility via fixed pseudorandom seed.
"""

import json
import os
import random
from typing import Dict, Any, List

RANDOM_SEED = 42
TOTAL_SAMPLES = 1200

FIRST_NAMES = [
    "Aarav", "Priya", "Rahul", "Sneha", "Vikram", "Anjali", "Rohan", "Neha",
    "Aditya", "Pooja", "Kavya", "Siddharth", "Meera", "Arjun", "Tanvi",
    "Ishaan", "Riya", "Karan", "Diya", "Varun", "Sunita", "Deepak", "Nandini"
]

LAST_NAMES = [
    "Sharma", "Verma", "Iyer", "Patel", "Reddy", "Nair", "Gupta", "Kulkarni",
    "Mukherjee", "Deshmukh", "Choudhury", "Bose", "Mehta", "Joshi", "Rao"
]

CANONICAL_SUBJECTS = {
    "Mathematics": ["Math", "Mathematics", "Calc", "Calculus", "Advanced Math"],
    "Physics": ["Physics", "Phys", "General Physics", "Applied Physics"],
    "Chemistry": ["Chemistry", "Chem", "Organic Chem", "General Chemistry"],
    "Computer Science": ["Computer Science", "CompSci", "CS", "Data Structures", "Programming"],
    "Biology": ["Biology", "Bio", "Molecular Biology", "Biosciences"],
    "English Literature": ["English Literature", "English", "Eng Lit", "Academic Writing"]
}

DISTRACTOR_ROOMS = ["Room 101", "Hall B-12", "Lecture Hall 4", "Auditorium 3", "Lab 204", "Block C-302"]
DISTRACTOR_TERMS = ["Fall 2023", "Spring 2024", "Fall 2024", "Winter 2025", "Summer Term"]

def compute_status(scores: Dict[str, int], attendance: float | None) -> str:
    """Deterministic academic standing rule based on extracted values."""
    if not scores:
        return "INCOMPLETE"
    vals = list(scores.values())
    avg = sum(vals) / len(vals)
    min_score = min(vals)
    
    if min_score < 40 or avg < 50:
        return "FAILED"
    elif min_score < 55 or avg < 65:
        return "PROBATION"
    else:
        return "PASSED"

def generate_sample(sample_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    # Identity
    fname = random.choice(FIRST_NAMES)
    lname = random.choice(LAST_NAMES)
    full_name = f"{fname} {lname}"
    roll_num = f"24BCE{random.randint(1000, 9999)}"
    
    # Semester & distractors
    semester = random.randint(1, 8)
    credits_attempted = random.choice([16, 18, 20, 22, 24])
    room = random.choice(DISTRACTOR_ROOMS)
    term = random.choice(DISTRACTOR_TERMS)
    cohort_rank = random.randint(3, 140)
    advisor_ext = random.randint(1100, 8990)
    
    # Subjects and scores (1 to 4 subjects selected)
    num_subs = random.choices([1, 2, 3, 4], weights=[0.1, 0.45, 0.35, 0.1])[0]
    chosen_canonical = random.sample(list(CANONICAL_SUBJECTS.keys()), num_subs)
    
    scores = {}
    subject_mentions = []
    
    for subj in chosen_canonical:
        score_val = random.randint(25, 99)
        scores[subj] = score_val
        alias = random.choice(CANONICAL_SUBJECTS[subj])
        
        phrasing_types = [
            f"attained a score of {score_val} in {alias}",
            f"scored {score_val}/100 on the final {alias} examination",
            f"logged {score_val} marks for {alias}",
            f"recieved {score_val} in {alias}", # intentional natural typo
            f"earned a grade of {score_val} in {alias}",
            f"{alias}: {score_val}%"
        ]
        subject_mentions.append(random.choice(phrasing_types))
    
    # Attendance (optional, 15% missing)
    has_attendance = random.random() > 0.15
    attendance_pct = round(random.uniform(55.0, 98.5), 1) if has_attendance else None
    
    # Status calculation
    status = compute_status(scores, attendance_pct)
    
    # Compose multi-sentence unstructured text with distractors
    text_templates = [
        # Narrative 1: Official Advisor Audit
        (
            f"Academic Assessment for candidate {full_name} (ID: {roll_num}) registered in {term} for Semester {semester}. "
            f"During the audit conducted in {room}, the committee reviewed {credits_attempted} attempted credits. "
            f"Course evaluation reports indicate the student {' and '.join(subject_mentions)}. "
            f"{f'Overall lecture attendance stood at {attendance_pct}% across all modules.' if has_attendance else 'Attendance logs remain unsubmitted for this reporting cycle.'} "
            f"Student currently ranks #{cohort_rank} in their section. Queries may be routed to ext. {advisor_ext}."
        ),
        # Narrative 2: Faculty Board Minutes
        (
            f"Review Board Memo: Student {roll_num}, known as {full_name}, currently progressing through Semester {semester}. "
            f"The faculty supervisor (contact ext. {advisor_ext}) logged performance metrics following midterm sessions in {room}. "
            f"Records verify that the student {' while also '.join(subject_mentions)}. "
            f"{f'Cumulative attendance verification reached {attendance_pct} percent.' if has_attendance else 'No biometric attendance records were recorded.'} "
            f"Note that {credits_attempted} credit units were registered for {term}."
        ),
        # Narrative 3: Disorganized Departmental Scratchpad
        (
            f"File update - {term}. Candidate {full_name} [{roll_num}], Sem {semester}, seating sector {room}. "
            f"Breakdown of recent exam results: {', '.join(subject_mentions)}. "
            f"{f'Biometric presence is at {attendance_pct}%.' if has_attendance else 'Attendance field marked pending review.'} "
            f"Target credit cap: {credits_attempted}. Section standing: ranked {cohort_rank} overall."
        )
    ]
    
    raw_text = random.choice(text_templates)
    
    # Ground truth data structure
    ground_truth = {
        "sample_id": sample_id,
        "student": {
            "name": full_name,
            "student_id": roll_num
        },
        "academic_records": {
            "scores": scores,
            "semester": semester,
            "attendance_percentage": attendance_pct
        },
        "status": status
    }
    
    input_sample = {
        "sample_id": sample_id,
        "raw_text": raw_text
    }
    
    return input_sample, ground_truth

def main():
    random.seed(RANDOM_SEED)
    os.makedirs("dataset", exist_ok=True)
    
    samples = []
    ground_truths = {}
    
    for i in range(TOTAL_SAMPLES):
        inp, gt = generate_sample(i)
        samples.append(inp)
        ground_truths[str(i)] = gt
    
    # Save benchmark samples (input only, no ground-truth key leakage)
    samples_path = os.path.join("dataset", "benchmark_samples.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    # Save ground truth in dedicated decoupled file
    gt_path = os.path.join("dataset", "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truths, f, indent=2)
        
    print(f"Generated {TOTAL_SAMPLES} samples successfully.")
    print(f"Inputs saved to: {samples_path}")
    print(f"Ground truth saved to: {gt_path}")

if __name__ == "__main__":
    main()
