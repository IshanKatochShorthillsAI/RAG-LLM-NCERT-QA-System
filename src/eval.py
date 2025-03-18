#!/usr/bin/env python

import pandas as pd
import nltk
import argparse
from nltk.tokenize import word_tokenize
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

# Download necessary NLTK resources.
nltk.download("punkt")
nltk.download("wordnet")


class EvaluationMetrics:
    """
    Encapsulates the metric computations:
      - Context token overlap precision & recall.
      - Faithfulness (cosine similarity between embeddings).
      - Answer relevancy (BERTScore F1).
    """

    def __init__(self, embedder: SentenceTransformer = None):
        # Initialize embedder; default to all-MiniLM-L6-v2 if none provided.
        self.embedder = (
            embedder if embedder else SentenceTransformer("all-MiniLM-L6-v2")
        )

    def compute_context_overlap(
        self, contexts: str, ground_truth: str
    ) -> Tuple[float, float]:
        """
        Compute context precision and recall based on token overlap.
          - context_precision = (# common tokens) / (# tokens in contexts)
          - context_recall = (# common tokens) / (# tokens in ground truth)
        """
        tokens_context = set(word_tokenize(contexts.lower()))
        tokens_gt = set(word_tokenize(ground_truth.lower()))
        intersection = tokens_context.intersection(tokens_gt)

        precision = len(intersection) / len(tokens_context) if tokens_context else 0.0
        recall = len(intersection) / len(tokens_gt) if tokens_gt else 0.0

        return precision, recall

    def compute_faithfulness(self, contexts: str, model_response: str) -> float:
        """
        Compute faithfulness as the cosine similarity between the embeddings of contexts and the model's response.
        """
        emb_context = self.embedder.encode(contexts)
        emb_response = self.embedder.encode(model_response)
        faithfulness = cosine_similarity([emb_context], [emb_response])[0][0]
        return faithfulness

    def compute_answer_relevancy(self, model_response: str, ground_truth: str) -> float:
        """
        Compute answer relevancy using BERTScore (F1).
        """
        # bertscore returns (P, R, F1) as tensors; we use F1.
        _, _, F1 = bertscore([model_response], [ground_truth], lang="en", verbose=False)
        return F1[0].item()


class Evaluator:
    """
    Evaluator handles loading the CSV, computing all metrics for each row,
    and saving the results (with an additional summary row) to a new CSV file.
    """

    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.metrics = EvaluationMetrics()

    def evaluate(self) -> None:
        # Load the CSV file with columns: Question, Contexts, Model's Response, Ground Truths
        df = pd.read_csv(self.input_file)

        # Initialize lists for the metrics.
        context_precisions = []
        context_recalls = []
        faithfulness_scores = []
        answer_relevancies = []

        # Iterate over each row in the DataFrame.
        for _, row in df.iterrows():
            # Ensure non-null strings.
            contexts = str(row["Contexts"]) if pd.notnull(row["Contexts"]) else ""
            model_response = (
                str(row["Model's Response"])
                if pd.notnull(row["Model's Response"])
                else ""
            )
            ground_truth = (
                str(row["Ground Truths"]) if pd.notnull(row["Ground Truths"]) else ""
            )

            # Compute context precision and recall.
            prec, rec = self.metrics.compute_context_overlap(contexts, ground_truth)
            context_precisions.append(prec)
            context_recalls.append(rec)

            # Compute faithfulness.
            faith_score = self.metrics.compute_faithfulness(contexts, model_response)
            faithfulness_scores.append(faith_score)

            # Compute answer relevancy.
            ans_rel = self.metrics.compute_answer_relevancy(
                model_response, ground_truth
            )
            answer_relevancies.append(ans_rel)

        # Append computed metrics as new columns.
        df["context_precision"] = context_precisions
        df["context_recall"] = context_recalls
        df["faithfulness"] = faithfulness_scores
        df["answer_relevancy"] = answer_relevancies

        # Compute overall (average) metrics for the entire CSV.
        metric_cols = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]
        overall_metrics = df[metric_cols].mean()

        # Create a summary row with overall averages.
        overall_row = {
            "Question": "Overall Averages",
            "Contexts": "",
            "Model's Response": "",
            "Ground Truths": "",
        }
        overall_row.update(overall_metrics.to_dict())
        overall_df = pd.DataFrame([overall_row])

        # Append the summary row using pd.concat.
        final_df = pd.concat([df, overall_df], ignore_index=True)

        # Save the final DataFrame to a CSV file.
        final_df.to_csv(self.output_file, index=False)
        print(f"Evaluation CSV saved to {self.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG outputs with context_precision, context_recall, faithfulness, and answer_relevancy."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_metrics.csv",
        help="Path to the output CSV file",
    )
    args = parser.parse_args()

    evaluator = Evaluator(args.input, args.output)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
