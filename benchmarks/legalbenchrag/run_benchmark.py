import argparse
import asyncio
import logging
import math
import os
import random
from typing import List, Tuple, Dict, Any
import pandas as pd
from pydantic import BaseModel, computed_field, model_validator
from functools import cached_property
from tqdm import tqdm
from typing_extensions import Self
from datetime import datetime

from sac_rag.data_models import Benchmark, Document, QAGroundTruth, RetrievalMethod, RetrievedSnippet
from sac_rag.utils.credentials import credentials
from sac_rag.utils.config_loader import load_strategy_from_file
from sac_rag.utils.retriever_factory import create_retriever
from sac_rag.methods.baseline import BaselineRetrievalStrategy
from sac_rag.methods.hybrid import HybridStrategy
from sac_rag.utils.utils import sanitize_filename


# --- Pydantic Models for this Benchmark's Evaluation Logic ---
class QAResult(BaseModel):
    qa_gt: QAGroundTruth
    retrieved_snippets: list[RetrievedSnippet]

    @cached_property
    def _relevant_retrieved_length(self) -> int:
        """Calculates the total character overlap. Is cached after the first call."""
        overlap_len = 0
        for snippet in self.retrieved_snippets:
            for gt_snippet in self.qa_gt.snippets:
                if snippet.file_path == gt_snippet.file_path:
                    # Calculate the length of the overlapping segment
                    overlap_start = max(snippet.span[0], gt_snippet.span[0])
                    overlap_end = min(snippet.span[1], gt_snippet.span[1])
                    if overlap_end > overlap_start:
                        overlap_len += overlap_end - overlap_start
        return overlap_len

    @cached_property
    def _total_retrieved_length(self) -> int:
        """Calculates the total length of all retrieved snippets."""
        return sum(s.span[1] - s.span[0] for s in self.retrieved_snippets)

    @cached_property
    def _total_relevant_length(self) -> int:
        """Calculates the total length of all ground truth snippets."""
        return sum(gt.span[1] - gt.span[0] for gt in self.qa_gt.snippets)

    # --- Public API Properties (now simple and clean) ---

    @computed_field
    @property
    def precision(self) -> float:
        if self._total_retrieved_length == 0:
            # If nothing was retrieved, precision is conventionally 0 or 1.
            # 0 is safer as it won't inflate scores for retrievers that return nothing.
            return 0.0
        return self._relevant_retrieved_length / self._total_retrieved_length

    @computed_field
    @property
    def recall(self) -> float:
        if self._total_relevant_length == 0:
            # This case should be rare, but if there's no ground truth text, recall is undefined or 1.
            # Returning 0.0 is a safe default.
            return 0.0
        return self._relevant_retrieved_length / self._total_relevant_length


def avg(arr: list[float]) -> float:
    return sum(arr) / len(arr) if arr else float("nan")


class BenchmarkResult(BaseModel):
    qa_result_list: list[QAResult]
    weights: list[float]

    def get_avg_recall_and_precision(self, tag_filter: str | None = None) -> tuple[float, float]:
        indices = [
            i for i, qa_result in enumerate(self.qa_result_list)
            if tag_filter is None or tag_filter in qa_result.qa_gt.tags
        ]
        if not indices:
            return float("nan"), float("nan")

        filtered_results = [self.qa_result_list[i] for i in indices]
        filtered_weights = [self.weights[i] for i in indices]

        total_weight = sum(filtered_weights)
        if total_weight == 0:  # Unweighted average
            avg_recall = avg([r.recall for r in filtered_results])
            avg_precision = avg([r.precision for r in filtered_results])
            return avg_recall, avg_precision

        # Weighted average
        recall_weighted_avg = sum(r.recall * w for r, w in zip(filtered_results, filtered_weights)) / total_weight
        precision_weighted_avg = sum(r.precision * w for r, w in zip(filtered_results, filtered_weights)) / total_weight
        return recall_weighted_avg, precision_weighted_avg

    @computed_field
    @property
    def avg_precision(self) -> float:
        return self.get_avg_recall_and_precision()[1]

    @computed_field
    @property
    def avg_recall(self) -> float:
        return self.get_avg_recall_and_precision()[0]

    @computed_field
    @property
    def avg_f1_score(self) -> float:
        precision, recall = self.avg_precision, self.avg_recall
        if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall > 0):
            return 2 * (precision * recall) / (precision + recall)
        return float('nan')

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        if len(self.qa_result_list) != len(self.weights):
            raise ValueError("Length of qa_result_list and weights do not match!")
        return self


# --- Core Benchmark Execution Logic ---

async def run_strategy(
        qa_gt_list: list[QAGroundTruth],
        corpus: list[Document],
        retriever: RetrievalMethod,
        weights: list[float] | None = None,
) -> BenchmarkResult:
    """Executes a benchmark run for a given retriever and test set."""
    for document in tqdm(corpus, desc="Ingesting documents"):
        await retriever.ingest_document(document)
    await retriever.sync_all_documents()

    async def run_query(qa_gt: QAGroundTruth) -> QAResult:
        query_response = await retriever.query(qa_gt.query)
        return QAResult(qa_gt=qa_gt, retrieved_snippets=query_response.retrieved_snippets)

    tasks = [run_query(qa_gt) for qa_gt in qa_gt_list]
    results = await asyncio.gather(
        *[t for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Running queries")])

    await retriever.cleanup()

    return BenchmarkResult(
        qa_result_list=results,
        weights=weights if weights is not None else [1.0] * len(results),
    )


# --- Data Setup Logic ---

def setup_and_load_data(max_tests: int, sort_by_doc: bool) -> Tuple[List[Document], List[QAGroundTruth], List[float]]:
    """Loads, samples, and prepares all data needed for the benchmark."""
    all_tests, weights, used_doc_paths = [], [], set()

    for dataset_name, weight in benchmark_name_to_weight.items():
        benchmark_file = f"./data/benchmarks/{dataset_name}.json"
        if not os.path.exists(benchmark_file):
            print(f"Warning: Benchmark file not found: {benchmark_file}. Skipping.")
            continue

        with open(benchmark_file, encoding='utf-8') as f:
            tests = Benchmark.model_validate_json(f.read()).tests

        # Sampling logic
        sampled_tests = tests
        if 0 < max_tests < len(tests):
            print(f"Sampling {max_tests} tests from {dataset_name} ({len(tests)} total)")
            if sort_by_doc:
                tests = sorted(tests, key=lambda t: t.snippets[0].file_path if t.snippets else "")
            else:
                random.seed(dataset_name + str(max_tests))
                random.shuffle(tests)
            sampled_tests = tests[:max_tests]

        for t in sampled_tests:
            for s in t.snippets:
                sanitized_path = sanitize_filename(s.file_path, f"{dataset_name}/")
                used_doc_paths.add(sanitized_path)

        for t in sampled_tests:
            t.tags = [dataset_name]

        all_tests.extend(sampled_tests)
        if sampled_tests:
            per_test_weight = weight / len(sampled_tests)
            weights.extend([per_test_weight] * len(sampled_tests))

    print(f"Total tests selected across all benchmarks: {len(all_tests)}")

    # Corpus loading
    corpus, loaded_paths = [], set()
    print(f"Attempting to load {len(used_doc_paths)} required corpus documents...")
    for doc_path in sorted(list(used_doc_paths)):
        full_path = f"./data/corpus/{doc_path}"
        if not os.path.exists(full_path):
            print(f"Warning: Corpus file not found at '{full_path}'. Skipping.")
            continue

        with open(full_path, encoding='utf-8') as f:
            content = f.read()
            if content.strip():
                corpus.append(Document(file_path=doc_path, content=content))
                loaded_paths.add(doc_path)

    print(f"Successfully loaded {len(loaded_paths)} corpus documents.")

    # Filter tests to only those with loaded documents
    final_tests, final_weights = [], []
    for i, test in enumerate(all_tests):
        all_loaded = True
        for s in test.snippets:
            sanitized_path = sanitize_filename(s.file_path, f"{test.tags[0]}/")
            if sanitized_path not in loaded_paths:
                all_loaded = False
                break
        if all_loaded:
            final_tests.append(test)
            final_weights.append(weights[i])

    if len(final_tests) != len(all_tests):  # should never happen
        print(f"Filtered out {len(all_tests) - len(final_tests)} tests due to missing corpus files.")

    if not final_tests:
        raise RuntimeError("No valid tests remaining after document filtering. Exiting.")

    return corpus, final_tests, final_weights


benchmark_name_to_weight: dict[str, float] = {
    "privacy_qa": 0.25, "contractnli": 0.25, "maud": 0.25, "cuad": 0.25,
}


def create_summary_row(idx: int, config_path: str, strategy: Any, result: BenchmarkResult) -> Dict[str, Any]:
    """Creates a detailed dictionary row for the summary CSV."""

    # Start with basic info
    row = {
        "i": idx,
        "config_file": config_path,
        "recall": result.avg_recall,
        "precision": result.avg_precision,
        "f1_score": result.avg_f1_score,
        "chunk_strategy_name": strategy.chunking_strategy.strategy_name,
        "chunk_size": strategy.chunking_strategy.chunk_size,
        "embedding_model_company": strategy.embedding_model.company,
        "embedding_model_name": strategy.embedding_model.model,
        "embedding_top_k": strategy.embedding_top_k,
        "rerank_model_company": strategy.rerank_model.company if strategy.rerank_model else None,
        "rerank_model_name": strategy.rerank_model.model if strategy.rerank_model else None,
        "rerank_top_k": strategy.rerank_top_k,
    }

    # Deconstruct the strategy object to get detailed columns
    if isinstance(strategy, BaselineRetrievalStrategy):
        row.update({
            "method": "baseline",
        })
    elif isinstance(strategy, HybridStrategy):
        row.update({
            "method": "hypa",
            "bm25_top_k": strategy.bm25_top_k,
            "fusion_top_k": strategy.fusion_top_k,
            "fusion_weight": strategy.fusion_weight,
        })
    else:
        print("WARNING: Unsupported strategy type. Skipping.")

    # Add the per-benchmark metrics
    for benchmark_name in benchmark_name_to_weight:
        avg_recall, avg_precision = result.get_avg_recall_and_precision(benchmark_name)
        row[f"{benchmark_name}|recall"] = avg_recall
        row[f"{benchmark_name}|precision"] = avg_precision

        f1 = float('nan')
        if not (math.isnan(avg_precision) or math.isnan(avg_recall)) and (avg_precision + avg_recall > 0):
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        row[f"{benchmark_name}|f1_score"] = f1

    return row


# --- Main Orchestrator ---

async def main(args):
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    start_time = datetime.now()
    print(f"Starting Legalbench-RAG benchmark run at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.environ["OPENAI_API_KEY"] = credentials.ai.openai_api_key.get_secret_value()
    os.environ["COHERE_API_KEY"] = credentials.ai.cohere_api_key.get_secret_value()
    os.environ["VOYAGEAI_API_KEY"] = credentials.ai.voyageai_api_key.get_secret_value()

    # 1. Setup and load all data once
    corpus, tests, weights = setup_and_load_data(args.max_tests_per_benchmark, args.sort_by_document)

    # 2. Prepare for results
    run_name = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Benchmark results will be saved to: {results_dir}")

    summary_rows = []

    # 3. Loop through each provided configuration file
    for i, config_path in enumerate(args.retrieval_configs):
        print(f"\n--- Running Config {i + 1}/{len(args.retrieval_configs)}: {config_path} ---")

        try:
            strategy = load_strategy_from_file(config_path)
            retriever = create_retriever(strategy)

            # Execute the benchmark
            result = await run_strategy(tests, corpus, retriever, weights=weights)

            # Save detailed JSON result for this run
            config_basename, _ = os.path.splitext(os.path.basename(config_path))
            result_filename = os.path.join(results_dir, f"{i}_{config_basename}.json")
            with open(result_filename, "w", encoding='utf-8') as f:
                f.write(result.model_dump_json(indent=2))

            # Prepare the DETAILED summary row
            row = create_summary_row(i, config_path, strategy, result)
            summary_rows.append(row)

            print(f"  Overall Avg Recall:    {100 * result.avg_recall: .2f}%")
            print(f"  Overall Avg Precision: {100 * result.avg_precision: .2f}%")
            print(f"  Overall Avg F1-Score:  {100 * result.avg_f1_score: .2f}%")

        except Exception as e:
            import traceback
            print(f"!!!!!!!!!!!! ERROR running benchmark for config {config_path} !!!!!!!!!!!!")
            print(f"Error: {e}")
            traceback.print_exc()
            summary_rows.append(
                {"config_file": config_path, "recall": "ERROR", "precision": "ERROR", "f1_score": "ERROR"})

    # 4. Save final summary CSV
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(results_dir, "results_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f'\nOverall Benchmark summary saved to: "{summary_path}"')

    print(f"\nBenchmark run '{run_name}' finished.")

    end_time = datetime.now()
    print(f"Run finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the legalbench-rag benchmark.")
    parser.add_argument(  # TODO: Implement that results are stored for all top-k values in one run (no additional overhead)
        "--retrieval-configs", "-rc",
        nargs='+', required=True,
        help="One or more paths to retrieval strategy JSON config files."
    )
    parser.add_argument(
        "--max-tests-per-benchmark", "-m", type=int, default=194,
        help="Maximum number of tests to sample from each sub-benchmark (e.g., cuad, maud). Set a low number for debug."
    )
    parser.add_argument(
        "--sort-by-document", action="store_true",
        help="Enable sorting by document to potentially speed up ingestion during testing."
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results/legalbenchrag",
        help="Base directory to save the output run folder."
    )

    args = parser.parse_args()
    asyncio.run(main(args))
