"""
Run reference-free RAG evaluation on a list of queries.

python -m src.evaluate_ragas
"""

import asyncio
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
)
from typing import List, Dict, Any
from langchain_community.chat_models import ChatOpenAI
from datasets import Dataset, Features, Sequence, Value

from rag import RAGSystem
from agent import AivancityAgent

load_dotenv()

# ↓ Put 10-30 dev questions here (no answers needed)
DEV_QUERIES: List[str] = [
    "What programs does Aivancity offer?",
    "Who is the president of Aivancity?",
    "Does Aivancity have partnerships with companies?",
    "Where is the Aivancity campus located?",
    "What are the admission requirements for the Grande École programme?",
]

def save_evaluation_results(
    scores: Dict[str, float],
    dataset: Dataset,
    output_dir: str = "evaluation_results"
) -> None:
    """Save evaluation results to files.
    
    Args:
        scores: Dictionary of metric scores
        dataset: HuggingFace dataset with questions, answers, and contexts
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw scores
    scores_file = output_path / f"scores_{timestamp}.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)
    
    # Save detailed results including questions, answers, and contexts
    results_df = pd.DataFrame({
        "question": dataset["question"],
        "answer": dataset["answer"],
        "contexts": dataset["contexts"],
        "ground_truths": dataset["ground_truths"]
    })
    results_file = output_path / f"detailed_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    
    # Generate and save summary report
    summary = [
        "=== RAGAS Evaluation Summary ===",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Number of queries evaluated: {len(dataset)}",
        "\nMetric Scores:",
    ]
    for metric, score in scores.items():
        summary.append(f"{metric}: {score:.4f}")
    
    summary_file = output_path / f"summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary))
    
    print(f"\nResults saved to:")
    print(f"- Scores: {scores_file}")
    print(f"- Detailed results: {results_file}")
    print(f"- Summary: {summary_file}")

async def main():
    # Initialize RAG system with vector store
    print("Initializing RAG system...")
    rag_system = RAGSystem()
    rag_system.load_index("data/faiss_index")  # Load existing index like in agent.py
    
    # Initialize agent with the RAG system
    print("Initializing agent...")
    agent = AivancityAgent(rag_system=rag_system, model_name="gpt-3.5-turbo")

    async def run_query(q: str):
        # run retrieval
        docs = rag_system.retrieve(q)
        context_chunks = [d.page_content for d in docs]

        # generate answer (sync invoke for simplicity)
        history = []
        prompt_messages = agent.prompt.format_messages(
            chat_history=history, input=q
        )
        # add context to system msg
        sys_msg = prompt_messages[0]
        prompt_messages[0] = sys_msg.__class__(
            content=f"{sys_msg.content}\n\n{'\n'.join(context_chunks)}"
        )
        answer = agent.llm.invoke(prompt_messages).content
        return q, answer, context_chunks

    print("Running queries...")
    rows = []
    for q in DEV_QUERIES:
        print(f"Processing query: {q}")
        rows.append(await run_query(q))

    # Convert to pandas DataFrame first
    df = pd.DataFrame(rows, columns=["question", "answer", "contexts"])
    print("\nCollected samples:", len(df))

    # Add ground truths column to DataFrame (empty list for each row)
    df["ground_truths"] = df.apply(lambda _: [""], axis=1)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    print("Running Ragas evaluation...")
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            context_precision,
            context_recall,
            answer_relevancy,
        ]
    )
    print("\n===  RAGAS scores  ===")
    print(scores)         # DataFrame with metric columns
    
    # Calculate and print mean scores
    print("\nMean scores:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    
    # Save results to files
    save_evaluation_results(scores, dataset)

if __name__ == "__main__":
    asyncio.run(main())