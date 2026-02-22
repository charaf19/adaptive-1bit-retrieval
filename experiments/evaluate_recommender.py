# experiments/evaluate_recommender.py
import json
import random
import pandas as pd
from tqdm import tqdm

import config
from src.engine import RetrievalEngine
from src.generator import Generator


TEST_SIZE = 50
K_RECALL = 200
K_RERANK = 10


def generate_ground_truth(domain: str):
    file_path = config.PROCESSED_DIR / f"{domain}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"Data not found: {file_path} (run: python -m src.data_loader)")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    random.seed(config.RANDOM_SEED)
    targets = random.sample(data, min(TEST_SIZE, len(data)))

    test_set = []
    for item in targets:
        title = item.get("title", "")
        if domain == "books":
            query = f"I want a book titled '{title}'"
        else:
            query = f"Find the paper '{title}'"
        test_set.append({
            "query": query,
            "target_title": title,
            "target_id": item["id"]
        })
    return test_set


def run_eval(domain: str):
    print(f"\n--- 🎯 Experiment: Recommender Quality ({domain.upper()}) ---")

    engine = RetrievalEngine(domain)
    generator = Generator()

    test_set = generate_ground_truth(domain)
    rows = []

    for case in tqdm(test_set, desc="Testing"):
        retrieved = engine.search(case["query"], k_recall=K_RECALL, k_rerank=K_RERANK)
        retrieved_ids = [r["id"] for r in retrieved]

        is_retrieved = case["target_id"] in retrieved_ids
        if is_retrieved:
            rank = retrieved_ids.index(case["target_id"]) + 1
            mrr = 1.0 / rank
        else:
            mrr = 0.0

        recommendation = generator.explain(case["query"], retrieved)

        gen_hit = case["target_title"].lower() in recommendation.lower()

        rows.append({
            "query": case["query"],
            "target_title": case["target_title"],
            "retrieved_in_top10": int(is_retrieved),
            "mrr": mrr,
            "gen_hit": int(gen_hit),
            "retrieval_status": "✅" if is_retrieved else "❌",
            "gen_status": "✅" if gen_hit else "❌",
        })

    df = pd.DataFrame(rows)
    recall_at_10 = df["retrieved_in_top10"].mean() * 100
    mrr = df["mrr"].mean()
    gen_hit = df["gen_hit"].mean() * 100

    summary = pd.DataFrame([{
        "Domain": domain,
        "Recall@10(%)": round(recall_at_10, 2),
        "MRR": round(mrr, 4),
        "GenHitRate(%)": round(gen_hit, 2),
        "K_recall": K_RECALL,
        "K_rerank": K_RERANK,
    }])

    out_csv = config.LOGS_DIR / f"recommender_eval_{domain}.csv"
    out_sum = config.LOGS_DIR / f"recommender_summary_{domain}.csv"
    df.to_csv(out_csv, index=False)
    summary.to_csv(out_sum, index=False)

    print("\n" + "=" * 60)
    print(f"   📊 RESULTS ({domain.upper()})")
    print("=" * 60)
    print(summary.to_markdown(index=False))
    print(f"✅ Saved: {out_csv.name} and {out_sum.name}")


if __name__ == "__main__":
    for d in ("books", "arxiv"):
        run_eval(d)
