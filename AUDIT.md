# Comprehensive Research Audit: EduMetrics-Agent-Benchmark

**Auditor:** Senior ML Systems Researcher & Software Engineer  
**Date:** October 27, 2026  
**Target Codebase & Manuscript:** `TOC IEEE_PAPER` (`generate_data.py`, `student_data.csv`, `run_experiment.py`, `experiment_results.csv`, `analyze_results.py`, `to show sir.pdf`)

---

## 1. Executive Summary & Assessment Ratings

| Dimension | Rating | Academic Verdict |
| :--- | :---: | :--- |
| **Research Topic Relevance** | **7.0 / 10** | Timely and practically valuable. Edge deployment of quantized Small Language Models (SLMs) for deterministic structured data extraction is an active area of systems and NLP research. |
| **Methodological Rigor** | **2.0 / 10** | Critically deficient. Trivial synthetic dataset, prompt ground-truth key leakage, evaluation of syntax rather than extraction accuracy, uncontrolled decoding hyperparameters, and no statistical testing. |
| **Scientific Integrity** | **1.0 / 10** | Severe violation. An Out-Of-Memory (OOM) crash on the 8-bit baseline was silently dropped in the analysis script to report a "100.0% formatting accuracy rate" in the manuscript. |
| **Current Publishability** | **1.0 / 10** | Immediate Desk Reject at any peer-reviewed IEEE conference or Scopus-indexed venue. |
| **Stylometric AI Signature** | **~90% AI** | Extensive hallmarks of unconstrained LLM writing: conversational idioms, textbook JPEG analogies, uniform cadence, boilerplate citations, and an unpopulated placeholder GitHub URL. |

---

## 2. Line-by-Line Code Flaw Verification

### Flaw 1: Trivial Synthetic Dataset and Lack of Ecological Validity
* **Files:** [`generate_data.py`](generate_data.py#L4-L26), [`student_data.csv`](student_data.csv)
* **Code Evidence:**
  * Lines 4–11: Dataset draws from only 10 student names, 5 subjects, and exactly **4 fixed phrase templates**:
    ```python
    phrases = [
        "scored {score} in {sub}",
        "managed to get a {score} on the {sub} exam",
        "failed with a {score} in {sub}",
        "got an outstanding {score} marks in {sub}"
    ]
    ```
  * Line 20: Every row adheres to an identical sentence structure:
    ```python
    text = f"Student {name} {random.choice(phrases).format(score=score1, sub=sub1)} and also {random.choice(phrases).format(score=score2, sub=sub2)}."
    ```
  * Lines 14–23: No random seed is initialized (`random.seed()` is omitted), making generation non-reproducible.
* **Impact:** The claim in the manuscript that the benchmark tests *"unpredictable, messy human text with bad grammar or confusing sentence structures"* (Manuscript Section I & III-B) is false. The dataset contains zero typographical errors, zero distractors (numbers representing dates, ages, or roll numbers), zero missing values, and zero nested relationships.

---

### Flaw 2: Prompt Ground-Truth Key Leakage
* **File:** [`run_experiment.py`](run_experiment.py#L28-L34)
* **Code Evidence:**
  * Lines 28–34:
    ```python
    prompt = f"""
    Extract the student name and their scores from the following text.
    Output the result STRICTLY as a valid JSON object with keys: "name", "{row['sub1']}", and "{row['sub2']}".
    Do NOT output any conversational text. ONLY output the JSON block.
    
    Text: {raw_text}
    """
    ```
* **Impact:** The prompt directly leaks `{row['sub1']}` and `{row['sub2']}` from the ground-truth table into the system instruction. The model does not perform entity recognition or schema mapping; it is merely filling in values for keys explicitly supplied by the prompter.

---

### Flaw 3: Inadequate Evaluation Metric (Syntax Validity != Extraction Correctness)
* **File:** [`run_experiment.py`](run_experiment.py#L49-L55)
* **Code Evidence:**
  * Lines 49–54:
    ```python
    is_valid_json = 1
    try:
        parsed_json = json.loads(ai_output)
    except json.JSONDecodeError:
        is_valid_json = 0
    ```
* **Impact:** The benchmark measures only whether `json.loads()` does not throw an exception. If the model generates `{"name": "FakeStudent", "Math": 0, "Physics": 0}`, it receives a perfect accuracy score of `1.0`. Extraction correctness—Precision, Recall, F1 score, field-level exact match, and numerical value verification—is completely absent.

---

### Flaw 4: Data Sanitization and Scientific Integrity Violation
* **Files:** [`run_experiment.py`](run_experiment.py#L65-L75), [`experiment_results.csv`](experiment_results.csv#L101), [`analyze_results.py`](analyze_results.py#L13-L20), [`to show sir.pdf`](to%20show%20sir.pdf)
* **Code Evidence:**
  * `run_experiment.py` lines 65–75 caught a crash on `Llama3_8bit` row 0:
    ```
    row_id: 0, model: Llama3_8bit, latency_seconds: 0.0, is_valid_json: 0, raw_output: SERVER_CRASH, error: "llama runner process has terminated: ..."
    ```
  * `analyze_results.py` lines 13–15 explicitly drop this row before calculating metrics:
    ```python
    # Clean data: Remove the row where the server crashed (Row 0 for 8-bit)
    clean_df = df[df['latency_seconds'] > 0]
    ```
  * Lines 17–20 compute `Accuracy = clean_df.groupby('model')['is_valid_json'].mean() * 100`, yielding 99/99 = 100.0%.
  * Manuscript Section V-A falsely claims:
    > *"Surprisingly, both the 8-bit baseline and the 4-bit compressed model achieved a perfect 100.0% success rate across the dataset. We didn’t see a single JSONDecodeError."*
* **Impact:** Discarding failed test points to report an unblemished 100% accuracy violates fundamental scientific research ethics. The failure on Row 0 was an Out-Of-Memory termination, which is a critical stability data point for edge deployment benchmarking.

---

### Flaw 5: Inaccurate and Misleading Terminology ("Autonomous Agent")
* **Files:** Throughout [`to show sir.pdf`](to%20show%20sir.pdf) (Title, Abstract, Section I, II-B, V-C, VII)
* **Critique:** The manuscript brands a single-pass zero-shot prompt extractor as an "Autonomous Agent." In AI systems literature, an autonomous agent requires:
  1. Multi-step reasoning loops (e.g., ReAct, Plan-and-Solve);
  2. External tool invocation / API interaction;
  3. Dynamic environment state feedback;
  4. Autonomous goal correction.
* Executing `ollama.chat()` once on a text string is static Information Extraction (IE), not an agent. Reviewers will view this as hype-driven terminology inflation.

---

### Flaw 6: Uncontrolled Decoding Parameters & Missing Cold-Start Isolation
* **File:** [`run_experiment.py`](run_experiment.py#L38-L46)
* **Code Evidence:**
  * Line 42: `ollama.chat(model=model_tag, messages=[{'role': 'user', 'content': prompt}])` passes no `options` dictionary (`temperature`, `top_p`, `top_k`, `seed`, `num_predict`). Ollama defaults to stochastic sampling (temperature ~0.7-0.8), rendering runs non-deterministic.
  * Line 38: `time.time()` on row 0 includes initial model paging/cold start into memory. In `experiment_results.csv`, `Llama3_4bit` row 0 had a latency of **8.33 seconds**, whereas subsequent rows averaged **0.81 seconds**. Mixing cold-start paging latency into mean inference latency without a warmup run skews reported averages.

---

### Flaw 7: Absence of Resource and Hardware Profiling
* **Files:** [`run_experiment.py`](run_experiment.py), [`analyze_results.py`](analyze_results.py)
* **Code Evidence:** The scripts contain no memory instrumentation.
* **Impact:** The paper claims significant RAM and thermal reductions (*"the computer’s fans were running loudly... the 4-bit model ran much cooler"*), but these are anecdotal observations. Neither peak resident set size (RSS), GPU VRAM allocation, nor throughput (tokens/second) was measured programmatically.

---

### Flaw 8: Incomplete and Unverified Manuscript Artifacts
* **File:** [`to show sir.pdf`](to%20show%20sir.pdf)
* **Citations & Placeholders:**
  * Section III: `(Available at: https://github.com/YourUsername/EduMetrics-Agent-Benchmark)` - unedited placeholder URL.
  * Table I: Processor listed as `Consumer Multi-core CPU (e.g., M-Series/i7)` - actual machine is an **Intel Core i9-13900HX (24 cores / 32 threads)** with an **NVIDIA GeForce RTX 4060 Laptop GPU (8GB GDDR6)** and 16 GB DDR5 RAM. Writing "e.g., M-Series/i7" indicates guesswork.
  * Section V-B: Latency numbers (3.10s vs 0.89s) lack standard deviation, interquartile ranges, 95% confidence intervals, and statistical significance testing (e.g., Mann-Whitney U test).

---

## 3. Actionable Remediation Roadmap

To elevate this codebase to IEEE publishable standard, the following steps are mandatory in Phase 2:
1. **Realistic Benchmark Corpus (N >= 1,000):** Synthesize multi-sentence paragraphs with conversational noise, distractor figures (dates, room numbers, course codes), spelling errors, and missing subjects. Maintain an isolated ground-truth key-value JSON file.
2. **Strict Schema-Guided Prompting (No Key Leakage):** Provide only the target JSON Schema. Compare three decoding regimes:
   - Regime A: Standard zero-shot unconstrained decoding.
   - Regime B: Constrained decoding via grammar / schema constraints (`format="json"` or GBNF).
   - Regime C: Pydantic-validated parsing / schema enforcement.
3. **Multi-Model & Multi-Quantization Grid:** Benchmark at least 3 distinct model architectures across multiple quantization tiers (e.g., Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0).
4. **Multi-Faceted Evaluation Metrics:**
   - JSON Syntactic Validity Rate (`%`)
   - Schema Conformance Rate (`%`)
   - Field-Level Precision, Recall, and F1 Score (`0.0 - 1.0`)
   - Strict Exact-Match Rate (`%`)
   - Inference Latency: Mean, Median, P95, and Standard Deviation
   - Generation Throughput: Tokens per second (`tok/s`)
   - Peak Host RAM & GPU VRAM utilization
5. **Rigorous Failure Accounting:** Retain all crashes, OOMs, and timeouts in the dataset. Compute and report failure rates explicitly.
6. **Automated Environment Logging & One-Click Reproducibility:** Automatically query and record CPU, GPU, OS, driver, and library versions into metadata logs. Provide `requirements.txt`, `README.md`, unit tests, and a master execution pipeline.

---
*End of Audit. Proceeding to Phase 2 upon verification.*
