# experiments/rag_qa_eval.py
import pandas as pd
from tqdm import tqdm

import config
from src.engine import RetrievalEngine
from src.generator import Generator


QA_SETS = {
    "books": [
        {"query": "What is the plot of The Great Gatsby?", "keywords": ["Gatsby", "Daisy", "Jazz"]},
        {"query": "Who wrote 1984?", "keywords": ["Orwell", "Big Brother"]},
        {"query": "Summary of The Hobbit", "keywords": ["Bilbo", "Smaug", "dragon"]},
    ],
    "arxiv": [
        {"query": "What does Attention Is All You Need propose?", "keywords": ["Transformer", "attention"]},
        {"query": "What is BERT?", "keywords": ["bidirectional", "pre-training", "transformer"]},
        {"query": "Explain YOLO object detection", "keywords": ["object", "detection", "real-time"]},
    ]
}


def simple_judge(text: str, keywords: list[str]) -> int:
    if not text:
        return 0
    low = text.lower()
    return int(any(k.lower() in low for k in keywords))


def run_qa(domain: str):
    print(f"\n--- 🧬 RAG QA Quality Check ({domain.upper()}) ---")
    engine = RetrievalEngine(domain)
    generator = Generator()

    test_set = QA_SETS.get(domain, [])
    rows = []
    for item in tqdm(test_set):
        ctx = engine.search(item["query"], k_recall=200, k_rerank=10)
        ans = generator.explain(item["query"], ctx)
        score = simple_judge(ans, item["keywords"])
        rows.append({"Query": item["query"], "Score": score, "Answer": ans})

    df = pd.DataFrame(rows)
    acc = df["Score"].mean() * 100 if len(df) else 0.0
    out = config.LOGS_DIR / f"qa_quality_{domain}.csv"
    df.to_csv(out, index=False)

    print(f"✅ {domain.upper()} QA Accuracy: {acc:.1f}%")
    print(f"✅ Saved: {out.name}")


if __name__ == "__main__":
    for d in ("books", "arxiv"):
        run_qa(d)
