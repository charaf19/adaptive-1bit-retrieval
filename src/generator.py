# src/generator.py
from __future__ import annotations
import os
import config


class Generator:
    """LLM-based explainer for RAG recommendations.

    Backend: llama-cpp-python (local GGUF). If the model is missing, the class raises an error.
    """

    def __init__(self):
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise ImportError(
                "llama-cpp-python is required for generation. "
                "Install it (recommended wheels) via the README, or run: pip install -r requirements-gen.txt"
            ) from e

        if not os.path.exists(config.GENERATION_MODEL_PATH):
            raise FileNotFoundError(
                f"Generation model not found at: {config.GENERATION_MODEL_PATH}\n"
                "Download a GGUF model (see README) or set BINARY_RAG_GENERATION_MODEL_PATH."
            )

        self.llm = Llama(
            model_path=config.GENERATION_MODEL_PATH,
            n_ctx=config.GENERATION_N_CTX,
            n_threads=config.GENERATION_N_THREADS,
            verbose=False,
        )

    def explain(self, query: str, context_items: list[dict]) -> str:
        if not context_items:
            return "I could not find any relevant items to recommend."

        catalog_lines = []
        for i, item in enumerate(context_items[:10]):
            content = (item.get("text", "") or "").replace("\n", " ").strip()[:400]
            catalog_lines.append(f"[Item {i+1}]: {content}")

        catalog_str = "\n".join(catalog_lines)

        prompt = f"""<|user|>
You are an expert librarian and recommendation engine.
User preference: "{query}"

Available catalog items:
{catalog_str}

Task:
1) Identify the best matching item.
2) Recommend it explicitly and justify using ONLY the catalog text.
Format: "I recommend [Item X] because ...".
<|end|>
<|assistant|>"""

        out = self.llm(
            prompt,
            max_tokens=180,
            stop=["<|end|>"],
            echo=False,
            temperature=0.1,
        )
        return out["choices"][0]["text"].strip()
