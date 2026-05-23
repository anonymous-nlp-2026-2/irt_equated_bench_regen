"""
Rank shift distribution analysis from plan_017 rank_movers data.
Extracts full distribution statistics, family-level breakdowns, and top-20 movers.
"""
import csv
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "plan_017" / "rank_movers.csv"
OUT_DIR = Path(__file__).parent

def load_data(cohort_filter="all"):
    rows = []
    with open(DATA_PATH) as f:
        for row in csv.DictReader(f):
            if row["cohort"] != cohort_filter:
                continue
            rows.append({
                "model": row["model_name"],
                "family": row["family"] if row["family"] and row["family"] != "nan" else "unknown",
                "method": row["training_method"],
                "acc_full": float(row["accuracy_full"]),
                "acc_stable": float(row["accuracy_stable"]),
                "rank_full": int(row["rank_full"]),
                "rank_stable": int(row["rank_stable"]),
                "rank_change": int(row["rank_change"]),
                "abs_rank_change": int(row["abs_rank_change"]),
            })
    return rows

def compute_distribution(values):
    a = np.array(values)
    return {
        "n": len(a),
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)),
        "median": float(np.median(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }

def family_stats(rows):
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r["rank_change"])
    results = []
    for fam, vals in sorted(by_fam.items(), key=lambda x: -len(x[1])):
        a = np.array(vals)
        results.append({
            "family": fam,
            "n": len(a),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "min": int(np.min(a)),
            "max": int(np.max(a)),
            "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
        })
    return results

def top_movers(rows, n=20):
    sorted_pos = sorted(rows, key=lambda r: -r["rank_change"])[:n]
    sorted_neg = sorted(rows, key=lambda r: r["rank_change"])[:n]
    return sorted_pos, sorted_neg

def format_report(dist, fam_results, top_drops, top_gains, rows):
    lines = []
    lines.append("# Rank Shift Distribution Report (plan_017 supplement)")
    lines.append("")
    lines.append("## Data Source")
    lines.append(f"- File: `artifacts/plan_017/rank_movers.csv`, cohort = `all`")
    lines.append(f"- N models: {dist['n']}")
    lines.append(f"- Rank shift = rank_stable − rank_full (positive = model drops when DIF items removed)")
    lines.append("")

    lines.append("## (a) Full-Model Rank Shift Distribution")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| N models | {dist['n']} |")
    lines.append(f"| Mean ± SD | {dist['mean']:.1f} ± {dist['sd']:.1f} |")
    lines.append(f"| Median | {dist['median']:.0f} |")
    lines.append(f"| IQR (Q1, Q3) | ({dist['q1']:.0f}, {dist['q3']:.0f}) |")
    lines.append(f"| P5, P95 | ({dist['p5']:.0f}, {dist['p95']:.0f}) |")
    lines.append(f"| Max positive shift (biggest rank drop) | +{dist['max']:.0f} |")
    lines.append(f"| Max negative shift (biggest rank gain) | {dist['min']:.0f} |")
    lines.append("")

    abs_vals = np.array([r["abs_rank_change"] for r in rows])
    lines.append("### Absolute rank shift")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Mean |Δrank| | {np.mean(abs_vals):.1f} |")
    lines.append(f"| Median |Δrank| | {np.median(abs_vals):.0f} |")
    lines.append(f"| P90 |Δrank| | {np.percentile(abs_vals, 90):.0f} |")
    lines.append(f"| P95 |Δrank| | {np.percentile(abs_vals, 95):.0f} |")
    lines.append(f"| P99 |Δrank| | {np.percentile(abs_vals, 99):.0f} |")
    n_lt10 = int(np.sum(abs_vals < 10))
    n_lt50 = int(np.sum(abs_vals < 50))
    n_lt100 = int(np.sum(abs_vals < 100))
    n_ge100 = int(np.sum(abs_vals >= 100))
    n_ge300 = int(np.sum(abs_vals >= 300))
    n_ge500 = int(np.sum(abs_vals >= 500))
    lines.append("")
    lines.append("### Cumulative distribution")
    lines.append("")
    lines.append("| Threshold | N models | % |")
    lines.append("|-----------|----------|---|")
    lines.append(f"| |Δrank| < 10 | {n_lt10} | {100*n_lt10/len(abs_vals):.1f}% |")
    lines.append(f"| |Δrank| < 50 | {n_lt50} | {100*n_lt50/len(abs_vals):.1f}% |")
    lines.append(f"| |Δrank| < 100 | {n_lt100} | {100*n_lt100/len(abs_vals):.1f}% |")
    lines.append(f"| |Δrank| ≥ 100 | {n_ge100} | {100*n_ge100/len(abs_vals):.1f}% |")
    lines.append(f"| |Δrank| ≥ 300 | {n_ge300} | {100*n_ge300/len(abs_vals):.1f}% |")
    lines.append(f"| |Δrank| ≥ 500 | {n_ge500} | {100*n_ge500/len(abs_vals):.1f}% |")
    lines.append("")

    lines.append("## (b) Family-Level Rank Shift")
    lines.append("")
    lines.append("### Families with N ≥ 10")
    lines.append("")
    lines.append("| Family | N | Mean ΔRank | Median ΔRank | SD | Min | Max | P5 | P95 |")
    lines.append("|--------|---|-----------|-------------|-----|-----|-----|-----|-----|")
    for f in fam_results:
        if f["n"] >= 10:
            lines.append(f"| {f['family']} | {f['n']} | {f['mean']:+.1f} | {f['median']:+.0f} | {f['sd']:.1f} | {f['min']:+d} | {f['max']:+d} | {f['p5']:+.0f} | {f['p95']:+.0f} |")
    lines.append("")

    lines.append("### Key families: individual extremes vs family average")
    lines.append("")
    lines.append("> **Warning**: The paper mentions '+700–850 rank shifts' for Phi-3/Llama-3.")
    lines.append("> These are **individual model extremes**, not family averages.")
    lines.append("> The table below separates the two to avoid misrepresentation.")
    lines.append("")
    key_families = ["phi-3", "llama-3", "mistral", "yi", "gemma", "qwen-1.5", "llama-2", "mixtral"]
    lines.append("| Family | N | Mean ΔRank | Median | SD | Individual Min | Individual Max |")
    lines.append("|--------|---|-----------|--------|-----|----------------|----------------|")
    for f in fam_results:
        if f["family"] in key_families:
            lines.append(f"| {f['family']} | {f['n']} | {f['mean']:+.1f} | {f['median']:+.0f} | {f['sd']:.1f} | {f['min']:+d} | {f['max']:+d} |")
    lines.append("")

    lines.append("## (c) Top-20 Most Affected Models")
    lines.append("")
    lines.append("### Biggest rank drops (benefited from DIF items — rank inflated by DIF)")
    lines.append("")
    lines.append("| # | Model | Family | Rank Full | Rank Stable | ΔRank |")
    lines.append("|---|-------|--------|-----------|-------------|-------|")
    for i, r in enumerate(top_drops, 1):
        lines.append(f"| {i} | {r['model']} | {r['family']} | {r['rank_full']} | {r['rank_stable']} | +{r['rank_change']} |")
    lines.append("")

    lines.append("### Biggest rank gains (hurt by DIF items — rank deflated by DIF)")
    lines.append("")
    lines.append("| # | Model | Family | Rank Full | Rank Stable | ΔRank |")
    lines.append("|---|-------|--------|-----------|-------------|-------|")
    for i, r in enumerate(top_gains, 1):
        lines.append(f"| {i} | {r['model']} | {r['family']} | {r['rank_full']} | {r['rank_stable']} | {r['rank_change']} |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    rows = load_data("all")
    rank_changes = [r["rank_change"] for r in rows]
    dist = compute_distribution(rank_changes)
    fam_results = family_stats(rows)
    top_drops, top_gains = top_movers(rows, n=20)
    report = format_report(dist, fam_results, top_drops, top_gains, rows)

    report_path = OUT_DIR / "rank_shift_distribution_report.md"
    report_path.write_text(report)
    print(f"Report written to {report_path}")

    json_path = OUT_DIR / "rank_shift_stats.json"
    json_path.write_text(json.dumps({
        "distribution": dist,
        "family_stats": fam_results,
        "top_drops": [{"model": r["model"], "family": r["family"], "rank_full": r["rank_full"], "rank_stable": r["rank_stable"], "rank_change": r["rank_change"]} for r in top_drops],
        "top_gains": [{"model": r["model"], "family": r["family"], "rank_full": r["rank_full"], "rank_stable": r["rank_stable"], "rank_change": r["rank_change"]} for r in top_gains],
    }, indent=2))
    print(f"JSON stats written to {json_path}")
