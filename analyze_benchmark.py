"""
analyze_benchmark.py
====================
Computes statistical significance, confidence intervals, and generates
camera-ready, high-resolution IEEE-styled figures from the un-sanitized benchmark data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# IEEE publication aesthetic
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "figure.titlesize": 13,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

def compute_detailed_statistics(df: pd.DataFrame) -> dict:
    stats_out = {}
    models = df["model"].unique()

    for m in models:
        m_df = df[df["model"] == m]
        # Latency (filtering out 0 latency from crash for speed analysis, but reporting crash)
        valid_lats = m_df[m_df["latency_seconds"] > 0]["latency_seconds"].values
        mean_lat = float(np.mean(valid_lats))
        std_lat = float(np.std(valid_lats, ddof=1))
        sem_lat = float(stats.sem(valid_lats))
        ci95_lat = float(sem_lat * stats.t.ppf(0.975, len(valid_lats) - 1))

        # JSON Validity
        valid_json = m_df["json_valid"].values
        val_mean = float(np.mean(valid_json) * 100)

        # F1
        f1_vals = m_df["scores_f1"].values
        f1_mean = float(np.mean(f1_vals))
        f1_std = float(np.std(f1_vals, ddof=1))
        f1_sem = float(stats.sem(f1_vals))
        f1_ci95 = float(f1_sem * stats.t.ppf(0.975, len(f1_vals) - 1))

        # Exact Match
        em_vals = m_df["exact_match"].values
        em_mean = float(np.mean(em_vals) * 100)

        stats_out[m] = {
            "total_samples": len(m_df),
            "failures": int(m_df["is_failure"].sum()),
            "failure_rate_pct": float(m_df["is_failure"].mean() * 100),
            "json_validity_pct": val_mean,
            "schema_conformance_pct": float(m_df["schema_conformance"].mean() * 100),
            "latency_mean_s": round(mean_lat, 3),
            "latency_std_s": round(std_lat, 3),
            "latency_ci95_s": round(ci95_lat, 3),
            "latency_median_s": round(float(np.median(valid_lats)), 3),
            "latency_p95_s": round(float(np.percentile(valid_lats, 95)), 3),
            "f1_mean": round(f1_mean, 3),
            "f1_std": round(f1_std, 3),
            "f1_ci95": round(f1_ci95, 3),
            "exact_match_pct": round(em_mean, 2)
        }

    # Hypothesis testing between 4-bit and 8-bit latency
    lat_4bit = df[(df["model"] == "Llama3_4bit") & (df["latency_seconds"] > 0)]["latency_seconds"].values
    lat_8bit = df[(df["model"] == "Llama3_8bit") & (df["latency_seconds"] > 0)]["latency_seconds"].values

    mwu_res = stats.mannwhitneyu(lat_4bit, lat_8bit, alternative="two-sided")
    ttest_res = stats.ttest_ind(lat_4bit, lat_8bit, equal_var=False)

    stats_out["statistical_tests"] = {
        "latency_mann_whitney_u": {
            "statistic": float(mwu_res.statistic),
            "p_value": float(mwu_res.pvalue),
            "significant_at_001": bool(mwu_res.pvalue < 0.001)
        },
        "latency_welch_t_test": {
            "statistic": float(ttest_res.statistic),
            "p_value": float(ttest_res.pvalue),
            "significant_at_001": bool(ttest_res.pvalue < 0.001)
        }
    }
    return stats_out

def generate_figure1_validity_vs_f1(df: pd.DataFrame):
    """
    Figure 1: Compares JSON Syntactic Validity against Semantic Extraction F1 Score.
    Exposes the critical research finding: high syntax compliance does NOT imply correct data extraction.
    """
    summary = df.groupby("model").agg(
        JSON_Validity=("json_valid", lambda x: x.mean() * 100),
        Extraction_F1=("scores_f1", lambda x: x.mean() * 100)
    ).reset_index()

    melted = pd.melt(summary, id_vars=["model"], value_vars=["JSON_Validity", "Extraction_F1"],
                     var_name="Metric", value_name="Score")
    melted["Metric"] = melted["Metric"].replace({
        "JSON_Validity": "Syntax Validity (%)",
        "Extraction_F1": "Extraction F1 (%)"
    })

    plt.figure(figsize=(6.5, 4.5))
    palette = {"Syntax Validity (%)": "#2b5c8f", "Extraction F1 (%)": "#d95f02"}
    ax = sns.barplot(x="model", y="Score", hue="Metric", data=melted, palette=palette)
    plt.title("Syntax Validity vs. Semantic Extraction F1 (IEEE Metric Discrepancy)", fontweight="bold")
    plt.xlabel("Model & Quantization Configuration")
    plt.ylabel("Score (%)")
    plt.ylim(0, 115)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom", fontsize=10, xytext=(0, 4),
                        textcoords="offset points")

    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("figure1_validity_vs_f1.png", dpi=300)
    plt.close()
    print("Generated 'figure1_validity_vs_f1.png'.")

def generate_figure2_latency(df: pd.DataFrame):
    """
    Figure 2: Inference Latency Distributions across quantization tiers.
    """
    clean = df[df["latency_seconds"] > 0]
    plt.figure(figsize=(6.5, 4.5))
    palette = {"Llama3_4bit": "#7570b3", "Llama3_8bit": "#e7298a"}
    ax = sns.boxplot(x="model", y="latency_seconds", data=clean, palette=palette, width=0.45, showmeans=True,
                     meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"})
    sns.stripplot(x="model", y="latency_seconds", data=clean, color="black", alpha=0.3, jitter=0.2, size=4)

    plt.title("Wall-Clock Inference Latency by Quantization Precision", fontweight="bold")
    plt.xlabel("Model Configuration")
    plt.ylabel("Inference Latency (Seconds)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("figure2_latency_distributions.png", dpi=300)
    plt.close()
    print("Generated 'figure2_latency_distributions.png'.")

def generate_figure3_stability(df: pd.DataFrame):
    """
    Figure 3: System Stability, Failure Rates, and Memory Fault Accounting.
    Documenting that 8-bit suffered a 1.0% Out-Of-Memory failure on consumer RAM.
    """
    summary = df.groupby("model").agg(
        Success_Rate=("is_failure", lambda x: (1.0 - x.mean()) * 100),
        Failure_Rate=("is_failure", lambda x: x.mean() * 100)
    ).reset_index()

    melted = pd.melt(summary, id_vars=["model"], value_vars=["Success_Rate", "Failure_Rate"],
                     var_name="Status", value_name="Percentage")
    melted["Status"] = melted["Status"].replace({
        "Success_Rate": "Execution Success Rate (%)",
        "Failure_Rate": "OOM Crash / Fault Rate (%)"
    })

    plt.figure(figsize=(6.5, 4.2))
    palette = {"Execution Success Rate (%)": "#1b9e77", "OOM Crash / Fault Rate (%)": "#e41a1c"}
    ax = sns.barplot(x="model", y="Percentage", hue="Status", data=melted, palette=palette)
    plt.title("Execution Reliability & Edge OOM Fault Accounting", fontweight="bold")
    plt.xlabel("Model Configuration")
    plt.ylabel("Execution Outcome (%)")
    plt.ylim(0, 115)

    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 4),
                    textcoords="offset points")

    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("figure3_system_failures_and_stability.png", dpi=300)
    plt.close()
    print("Generated 'figure3_system_failures_and_stability.png'.")

def main():
    print("=== COMPUTING ADVANCED STATISTICAL BENCHMARKS ===")
    df = pd.read_csv("comprehensive_evaluation_results.csv")
    
    stats_data = compute_detailed_statistics(df)
    with open("statistical_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)
    print("Logged statistics and significance tests to 'statistical_evaluation.json':")
    print(json.dumps(stats_data, indent=2))

    generate_figure1_validity_vs_f1(df)
    generate_figure2_latency(df)
    generate_figure3_stability(df)
    print("Benchmark analysis complete.")

if __name__ == "__main__":
    main()
