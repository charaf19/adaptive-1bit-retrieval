# experiments/failure_analysis.py
import pandas as pd
import config


def analyze_failures(domain="books"):
    print("--- 🕵️ Failure Mode Analysis ---")
    log_path = config.LOGS_DIR / f"recommender_eval_{domain}.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {log_path.name}. Run: python -m experiments.evaluate_recommender")

    df = pd.read_csv(log_path)
    total = len(df)

    success = df[(df["retrieval_status"] == "✅") & (df["gen_status"] == "✅")]
    halluc = df[(df["retrieval_status"] == "❌") & (df["gen_status"] == "✅")]
    reasoning = df[(df["retrieval_status"] == "✅") & (df["gen_status"] == "❌")]
    system = df[(df["retrieval_status"] == "❌") & (df["gen_status"] == "❌")]

    print(f"Total samples: {total}")
    print(f"✅ Complete success:   {len(success)} ({len(success)/total*100:.1f}%)")
    print(f"🧠 Reasoning failure:  {len(reasoning)} ({len(reasoning)/total*100:.1f}%)")
    print(f"👻 Hallucination risk: {len(halluc)} ({len(halluc)/total*100:.1f}%)")
    print(f"❌ System failure:     {len(system)} ({len(system)/total*100:.1f}%)")

    report = config.LOGS_DIR / f"failure_report_{domain}.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("Reasoning failures (retrieval ok, generation wrong):\n")
        f.write("\n".join(reasoning["target_title"].astype(str).tolist()))
        f.write("\n\nSystem failures (retrieval + generation wrong):\n")
        f.write("\n".join(system["target_title"].astype(str).tolist()))

    print(f"📄 Saved: {report.name}")


if __name__ == "__main__":
    analyze_failures("books")
