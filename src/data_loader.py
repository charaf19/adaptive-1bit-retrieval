# src/data_loader.py
import json
from datasets import load_dataset
from tqdm import tqdm
import config


def ingest_domain(domain: str) -> None:
    if domain not in {"books", "arxiv"}:
        raise ValueError("domain must be one of: books, arxiv")

    output_file = config.PROCESSED_DIR / f"{domain}.jsonl"
    data = []
    count = 0
    skipped = 0

    if domain == "books":
        print("   Source: amazon_polarity (Hugging Face)")
        dataset = load_dataset("amazon_polarity", split="train", streaming=True)
        for item in tqdm(dataset, total=config.INGEST_LIMIT, desc="   Processing Books"):
            if config.INGEST_LIMIT and count >= config.INGEST_LIMIT:
                break
            # Filter: Positive reviews (label=1) with meaningful text length
            if item.get("label", 0) == 1 and item.get("content") and len(item["content"]) > 100:
                text = f"Title: {item['title']}\nReview: {item['content']}"
                data.append({
                    "id": f"book_{count}",
                    "text": text,
                    "title": item["title"],
                    "domain": "books"
                })
                count += 1
            else:
                skipped += 1

    if domain == "arxiv":
        print("   Source: CShorten/ML-ArXiv-Papers (Hugging Face)")
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train", streaming=True)
        for item in tqdm(dataset, total=config.INGEST_LIMIT, desc="   Processing ArXiv"):
            if config.INGEST_LIMIT and count >= config.INGEST_LIMIT:
                break
            if item.get("title") and item.get("abstract") and len(item["abstract"]) > 100:
                text = f"Title: {item['title']}\nAbstract: {item['abstract']}"
                data.append({
                    "id": f"arxiv_{count}",
                    "text": text,
                    "title": item["title"],
                    "domain": "arxiv"
                })
                count += 1
            else:
                skipped += 1

    print(f"   💾 Saving {len(data)} items to {output_file}...")
    print(f"   (Skipped {skipped} low-quality items)")

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ingest_data(domains=("books", "arxiv")) -> None:
    for domain in domains:
        print(f"\n--- 📥 Ingesting Domain: {domain.upper()} ---")
        ingest_domain(domain)


if __name__ == "__main__":
    ingest_data()
