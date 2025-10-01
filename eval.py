"""
Evaluation suite for the Multimodal Depression RAG.

Two modes:
1) **RAGAS** LLM-based metrics (faithfulness, answer relevancy, context precision/recall)
   - Requires an LLM provider (OpenAI or Bedrock via LangChain). Set env in .env
2) **Retrieval metrics** (Recall@K, MRR, latency) if you provide ground-truth doc ids in the dataset.

Dataset format (JSONL) for --questions:
{
  "question": "What is PHQ-9?",
  "ground_truth": "PHQ-9 is a 9-item questionnaire...",    # optional but recommended
  "ground_truth_sources": ["paper1.pdf"],                  # to compute retrieval Recall@K/MRR
  "expected_phrases": ["9-item","severity"],             # optional keyword checks
}

We will call the live pipeline to retrieve + answer, then evaluate.
"""
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from typing import List, Dict

from pipeline import RAGPipeline
from config import settings

# RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_aws import ChatBedrockConverse
except Exception:
    ChatBedrockConverse = None

load_dotenv()

def get_llm_for_ragas():
    if settings.provider == "openai" and settings.openai_api_key:
        return ChatOpenAI(model=settings.openai_model, temperature=0.0, openai_api_key=settings.openai_api_key)
    if settings.provider == "bedrock" and ChatBedrockConverse is not None:
        return ChatBedrockConverse(model=settings.bedrock_model_id, region_name=settings.aws_region, temperature=0.0)
    # Fallback (may error if no key):
    return ChatOpenAI(model=settings.openai_model, temperature=0.0)

def get_embed_for_ragas():
    # Prefer local HF embeddings to avoid paid tokens
    return HuggingFaceEmbeddings(model_name=settings.text_model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="Path to JSONL with questions")
    parser.add_argument("--k", type=int, default=settings.top_k)
    args = parser.parse_args()

    # Load questions
    rows = []
    with open(args.questions, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    pipe = RAGPipeline()

    # Run the pipeline for each question and collect results
    eval_records = []
    for row in rows:
        q = row["question"]
        t0 = time.time()
        result = pipe.query(q, args.k)
        latency = time.time() - t0
        answer = result["answer"]
        contexts = [c.get("document") or json.dumps(c.get("metadata", {})) for c in result["contexts"]]
        sources = [ (c.get("metadata") or {}).get("source", "") for c in result["contexts"] ]
        eval_records.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "ground_truth": row.get("ground_truth", ""),
            "ground_truth_sources": row.get("ground_truth_sources", []),
            "latency_sec": latency,
        })

    df = pd.DataFrame(eval_records)

    # Retrieval metrics (if ground_truth_sources provided)
    def recall_at_k(sources: List[str], gt: List[str]) -> float:
        if not gt:
            return np.nan
        return 1.0 if any(g in sources for g in gt) else 0.0

    def mrr_at_k(sources: List[str], gt: List[str]) -> float:
        for i, s in enumerate(sources, start=1):
            if s in gt:
                return 1.0 / i
        return 0.0 if gt else np.nan

    df["Recall@K"] = df.apply(lambda r: recall_at_k(r["sources"], r["ground_truth_sources"]), axis=1)
    df["MRR@K"] = df.apply(lambda r: mrr_at_k(r["sources"], r["ground_truth_sources"]), axis=1)

    # RAGAS (LLM-based)
    llm = get_llm_for_ragas()
    embed = get_embed_for_ragas()
    ragas_ds = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])

    ragas_results = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embed,
    )

    ragas_df = ragas_results.to_pandas()
    out = df.join(ragas_df)

    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/eval_results.csv"
    out.to_csv(out_csv, index=False)

    print("===== Aggregate Metrics =====")
    agg = {
        "n": len(out),
        "Recall@K_mean": float(np.nanmean(out["Recall@K"])) if len(out) else 0.0,
        "MRR@K_mean": float(np.nanmean(out["MRR@K"])) if len(out) else 0.0,
        "faithfulness_mean": float(out["faithfulness"].mean()) if "faithfulness" in out else None,
        "answer_relevancy_mean": float(out["answer_relevancy"].mean()) if "answer_relevancy" in out else None,
        "context_precision_mean": float(out["context_precision"].mean()) if "context_precision" in out else None,
        "context_recall_mean": float(out["context_recall"].mean()) if "context_recall" in out else None,
        "latency_sec_mean": float(out["latency_sec"].mean()) if len(out) else 0.0,
    }
    for k, v in agg.items():
        print(f"{k}: {v}")
    print(f"\nPer-sample results saved to: {out_csv}")

if __name__ == "__main__":
    main()
