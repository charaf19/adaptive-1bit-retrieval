"""Generate paper-ready tables (Markdown) and summary figures from results/logs.

This script is intentionally *deterministic* and reads only from:
  - artifacts/ (index sizes)
  - results/logs/*.csv (experiment outputs)

Outputs:
  - results/tables/*.md
  - results/figures/*_summary.png
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import config


def _ensure_dirs():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _write_table(path_md: Path, title: str, df: pd.DataFrame):
    """Write .md + .csv + .tex with the same stem."""
    path_md.parent.mkdir(parents=True, exist_ok=True)
    stem = path_md.with_suffix("")

    # Markdown
    md = f"# {title}\n\n" + df.to_markdown(index=False) + "\n"
    path_md.write_text(md, encoding="utf-8")

    # CSV (paper-friendly copy)
    df.to_csv(stem.with_suffix(".csv"), index=False)

    # LaTeX
    tex = df.to_latex(index=False)
    stem.with_suffix(".tex").write_text(tex, encoding="utf-8")


def indexing_metrics_table():
    metrics_path = config.RESULTS_DIR / "indexing_metrics.json"
    if not metrics_path.exists():
        print("⚠️ indexing_metrics.json not found (run: python -m src.indexer)")
        return

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for domain, m in metrics.items():
        rows.append({
            "Domain": domain,
            "Float (MB)": round(m.get("float_mb", 0.0), 3),
            "Binary (MB)": round(m.get("binary_mb", 0.0), 3),
            "PQ (MB)": round(m.get("pq_mb", 0.0), 3),
            "HNSW (MB)": round(m.get("hnsw_mb", 0.0), 3),
            "IVF-Flat (MB)": round(m.get("ivf_flat_mb", 0.0), 3),
            "IVF-PQ (MB)": round(m.get("ivf_pq_mb", 0.0), 3),
            "Float→Binary (×)": round(m.get("compression_x", 0.0), 2),
            "Quantization": m.get("quantization_mode", ""),
        })

    df = pd.DataFrame(rows)
    _write_table(config.TABLES_DIR / "table_indexing_metrics.md", "Indexing & Compression Metrics", df)
    print("✅ Wrote table_indexing_metrics.md")


def index_build_metrics_table():
    """Per-index build time + size table (helps reviewers judge indexing cost)."""
    p = config.LOGS_DIR / "index_build_metrics.csv"
    if not p.exists():
        print("⚠️ index_build_metrics.csv not found (run: python -m src.indexer)")
        return

    df = pd.read_csv(p)
    # Friendly rounding
    for c in ["train_s", "add_s", "build_s", "size_mb"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(3)

    # Order columns if present
    cols = [c for c in ["domain", "index", "params", "size_mb", "train_s", "add_s", "build_s"] if c in df.columns]
    df = df[cols].sort_values(["domain", "index", "size_mb"], ascending=[True, True, True])
    _write_table(config.TABLES_DIR / "table_index_build_metrics.md", "Index Build Time & Size (Per Index)", df)
    print("✅ Wrote table_index_build_metrics.md")


def baselines_tables_and_figures(domain: str):
    p_search = config.LOGS_DIR / f"baseline_search_{domain}.csv"
    p_e2e = config.LOGS_DIR / f"baseline_end2end_{domain}.csv"
    if not p_search.exists() or not p_e2e.exists():
        print(f"⚠️ Missing baseline logs for {domain}. Run: python -m experiments.benchmark_baselines")
        return

    df_s = pd.read_csv(p_search)
    df_e = pd.read_csv(p_e2e)

    # Nice ordering: ours last so it is easy to spot
    df_s = df_s.sort_values(by=["Recall@10", "Search_ms"], ascending=[False, True])
    df_e = df_e.sort_values(by=["Recall@10", "Total_ms"], ascending=[False, True])

    _write_table(config.TABLES_DIR / f"table_baselines_{domain}_search.md", f"Baselines (Search-only) — {domain.upper()}", df_s)
    _write_table(config.TABLES_DIR / f"table_baselines_{domain}.md", f"Baselines (End-to-End) — {domain.upper()}", df_e)

    # Summary figure: Recall vs latency
    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(df_e["Total_ms"], df_e["Recall@10"], s=60)
    for _, r in df_e.iterrows():
        plt.annotate(r["Method"], (r["Total_ms"], r["Recall@10"]), fontsize=8, xytext=(5, 3), textcoords="offset points")
    plt.xlabel("End-to-end latency per query (ms)")
    plt.ylabel("Recall@10 (%)")
    plt.title(f"Baselines trade-off — {domain.upper()} (End-to-end)")
    plt.grid(True, alpha=0.3)
    out = config.FIGURES_DIR / f"baselines_e2e_{domain}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(df_s["Search_ms"], df_s["Recall@10"], s=60)
    for _, r in df_s.iterrows():
        plt.annotate(r["Method"], (r["Search_ms"], r["Recall@10"]), fontsize=8, xytext=(5, 3), textcoords="offset points")
    plt.xlabel("Search-only latency per query (ms)")
    plt.ylabel("Recall@10 (%)")
    plt.title(f"Baselines trade-off — {domain.upper()} (Search-only)")
    plt.grid(True, alpha=0.3)
    out2 = config.FIGURES_DIR / f"baselines_search_{domain}.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Wrote baseline tables + figures for {domain}")


def baseline_sweep_tables_and_pareto(domain: str):
    """Pareto-ready table + plot from baseline hyperparameter sweeps."""
    p = config.LOGS_DIR / f"baseline_sweep_search_{domain}.csv"
    if not p.exists():
        print(f"⚠️ Missing baseline sweep for {domain}. Run: python -m experiments.baseline_sweep")
        return

    df = pd.read_csv(p)
    df = df.sort_values(["Recall@10", "Search_ms"], ascending=[False, True])
    _write_table(
        config.TABLES_DIR / f"table_baseline_sweep_{domain}.md",
        f"Baseline Hyperparameter Sweep (Search-only) — {domain.upper()}",
        df,
    )

    # Pareto frontier (maximize recall, minimize latency)
    df2 = df.sort_values("Search_ms", ascending=True).reset_index(drop=True)
    pareto_idx = []
    best_r = -1.0
    for i, r in df2.iterrows():
        if r["Recall@10"] > best_r + 1e-9:
            pareto_idx.append(i)
            best_r = r["Recall@10"]
    pareto = df2.loc[pareto_idx]

    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(df2["Search_ms"], df2["Recall@10"], s=35)
    plt.plot(pareto["Search_ms"], pareto["Recall@10"], linewidth=2)
    # Annotate only Pareto points (keeps plot readable)
    for _, r in pareto.iterrows():
        plt.annotate(r["Method"], (r["Search_ms"], r["Recall@10"]), fontsize=8, xytext=(5, 3), textcoords="offset points")
    plt.xlabel("Search-only latency per query (ms)")
    plt.ylabel("Recall@10 (%)")
    plt.title(f"Pareto frontier — {domain.upper()} (Search-only)")
    plt.grid(True, alpha=0.3)
    out = config.FIGURES_DIR / f"pareto_search_{domain}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    # Save Pareto table too
    _write_table(
        config.TABLES_DIR / f"table_pareto_search_{domain}.md",
        f"Pareto Frontier (Search-only) — {domain.upper()}",
        pareto.sort_values(["Recall@10", "Search_ms"], ascending=[False, True]),
    )
    print(f"✅ Wrote baseline sweep + Pareto for {domain}")


def efficiency_tables(domain: str):
    p = config.LOGS_DIR / f"efficiency_{domain}.csv"
    if not p.exists():
        print(f"⚠️ Missing efficiency log for {domain}. Run: python -m experiments.benchmark_efficiency")
        return
    df = pd.read_csv(p)
    _write_table(config.TABLES_DIR / f"table_efficiency_{domain}.md", f"Index Efficiency — {domain.upper()}", df)


def sensitivity_tables(domain: str):
    p = config.LOGS_DIR / f"sensitivity_{domain}.csv"
    if not p.exists():
        print(f"⚠️ Missing sensitivity log for {domain}. Run: python -m experiments.sensitivity_analysis")
        return
    df = pd.read_csv(p)
    _write_table(config.TABLES_DIR / f"table_sensitivity_{domain}.md", f"Sensitivity (k_recall sweep) — {domain.upper()}", df)


def robustness_tables(domain: str):
    p = config.LOGS_DIR / f"robustness_{domain}.csv"
    if not p.exists():
        print(f"⚠️ Missing robustness log for {domain}. Run: python -m experiments.model_robustness")
        return
    df = pd.read_csv(p)
    _write_table(config.TABLES_DIR / f"table_robustness_{domain}.md", f"Model Robustness — {domain.upper()}", df)


def scalability_table():
    p = config.LOGS_DIR / "scalability.csv"
    if not p.exists():
        print("⚠️ Missing scalability.csv. Run: python -m experiments.scalability_test")
        return
    df = pd.read_csv(p)
    _write_table(config.TABLES_DIR / "table_scalability.md", "Scalability (Search-only)", df)


def main():
    _ensure_dirs()

    indexing_metrics_table()
    index_build_metrics_table()

    for d in ("books", "arxiv"):
        baselines_tables_and_figures(d)
        baseline_sweep_tables_and_pareto(d)
        efficiency_tables(d)
        sensitivity_tables(d)
        robustness_tables(d)

    scalability_table()
    print("✅ Analytics complete. See results/tables and results/figures")


if __name__ == "__main__":
    main()
