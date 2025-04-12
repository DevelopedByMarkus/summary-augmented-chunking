import asyncio
import datetime as dt
import os
import random
import pandas as pd

from legalbenchrag.benchmark_types import Benchmark, Document, QAGroundTruth, RetrievalMethod
# Import Baseline components
from legalbenchrag.methods.baseline import BaselineRetrievalMethod, RetrievalStrategy as BaselineStrategy  # BaselineStrategy are the 4 original legalbench-rag strategies
# Import HyPA components
from legalbenchrag.methods.hypa import HypaRetrievalMethod, HypaStrategy
from legalbenchrag.methods.retrieval_strategies import ALL_RETRIEVAL_STRATEGIES
from legalbenchrag.run_benchmark import run_benchmark
from legalbenchrag.utils.credentials import credentials

benchmark_name_to_weight: dict[str, float] = {
    "privacy_qa": 0.25,
    "contractnli": 0.25,
    "maud": 0.25,
    "cuad": 0.25,
}

# This takes a random sample of the benchmark, to speed up query processing.
# p-values can be calculated, to statistically predict the theoretical performance on the "full test"
MAX_TESTS_PER_BENCHMARK = 1  # 194
# This sorts the tests by document,
# so that the MAX_TESTS_PER_BENCHMARK tests are over the fewest number of documents,
# This speeds up ingestion processing, but
# p-values cannot be calculated, because this settings drastically reduces the search space size.
SORT_BY_DOCUMENT = True


async def main() -> None:
    os.environ["OPENAI_API_KEY"] = credentials.ai.openai_api_key.get_secret_value()

    all_tests: list[QAGroundTruth] = []
    weights: list[float] = []
    document_file_paths_set: set[str] = set()
    used_document_file_paths_set: set[str] = set()
    for benchmark_name, weight in benchmark_name_to_weight.items():
        with open(f"./data/benchmarks/{benchmark_name}.json", encoding='utf-8') as f:
            benchmark = Benchmark.model_validate_json(f.read())
            tests = benchmark.tests
            document_file_paths_set |= {
                snippet.file_path for test in tests for snippet in test.snippets
            }
            # Cap queries for a given benchmark
            if len(tests) > MAX_TESTS_PER_BENCHMARK:
                if SORT_BY_DOCUMENT:
                    # Use random based on file path seed, rather than the file path itself, to prevent bias.
                    tests = sorted(
                        tests,
                        key=lambda test: (
                            random.seed(test.snippets[0].file_path),
                            random.random(),
                        )[1],
                    )
                else:
                    # Keep seed consistent, for better caching / testing.
                    random.seed(benchmark_name)
                    random.shuffle(tests)
                tests = tests[:MAX_TESTS_PER_BENCHMARK]
            used_document_file_paths_set |= {
                snippet.file_path for test in tests for snippet in test.snippets
            }
            for test in tests:
                test.tags = [benchmark_name]
            all_tests.extend(tests)
            weights.extend([weight / len(tests)] * len(tests))
    benchmark = Benchmark(
        tests=all_tests,
    )

    # Create corpus (sorted for consistent processing)
    corpus: list[Document] = []
    for document_file_path in sorted(
        document_file_paths_set
        if not SORT_BY_DOCUMENT
        else used_document_file_paths_set
    ):
        with open(f"./data/corpus/{document_file_path}", encoding='utf-8') as f:
            corpus.append(
                Document(
                    file_path=document_file_path,
                    content=f.read(),
                )
            )

    # Create a save location for this run
    run_name = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    benchmark_path = f"./benchmark_results/{run_name}"
    os.makedirs(benchmark_path, exist_ok=True)

    rows: list[dict[str, str | None | int | float]] = []
    for i, retrieval_strategy in enumerate(ALL_RETRIEVAL_STRATEGIES):
        retrieval_method: RetrievalMethod  # Define type hint

        # --- Instantiate correct RetrievalMethod based on Strategy type ---
        if isinstance(retrieval_strategy, BaselineStrategy):
            print(f"\n--- Running Baseline Strategy {i} ---")
            # Cast needed if strategy types aren't perfectly aligned, Pydantic usually handles this
            retrieval_method = BaselineRetrievalMethod(
                retrieval_strategy=retrieval_strategy
            )
            # Prepare row data for Baseline
            row: dict[str, str | None | int | float] = {
                "i": i,
                "method": "baseline",  # Add method identifier
                "chunk_strategy_name": retrieval_strategy.chunking_strategy.strategy_name,
                "chunk_size": retrieval_strategy.chunking_strategy.chunk_size,
                "embedding_model": retrieval_strategy.embedding_model.model, # Get model name
                "top_k": retrieval_strategy.embedding_topk,
                "rerank_model": retrieval_strategy.rerank_model.company
                if retrieval_strategy.rerank_model is not None
                else None,
                "top_k_rerank": retrieval_strategy.rerank_topk,
                "token_limit": retrieval_strategy.token_limit,
            }
        elif isinstance(retrieval_strategy, HypaStrategy):
            print(f"\n--- Running HyPA Strategy {i} ---")
            retrieval_method = HypaRetrievalMethod(
                strategy=retrieval_strategy
            )
            # Prepare row data for HyPA
            row: dict[str, str | None | int | float] = {
                "i": i,
                "method": "hypa", # Add method identifier
                "chunk_size": retrieval_strategy.chunk_size,
                "embedding_model": retrieval_strategy.embedding_model.model,  # Get model name
                "embedding_top_k": retrieval_strategy.embedding_top_k,
                "bm25_top_k": retrieval_strategy.bm25_top_k,
                "fusion_top_k": retrieval_strategy.fusion_top_k,
                # Add other HyPA specific params here later if needed
            }
        else:
            print(f"WARNING: Unknown strategy type at index {i}. Skipping.")
            continue

        # --- Run Benchmark Logic (keep as is) ---
        print(f"Strategy Config: {retrieval_strategy.model_dump()}") # Log strategy
        print(f"Num Documents: {len(corpus)}")
        print(
            f"Num Corpus Characters: {sum(len(document.content) for document in corpus)}"
        )
        print(f"Num Queries: {len(benchmark.tests)}")

        benchmark_result = await run_benchmark(
            benchmark.tests,
            corpus,
            retrieval_method,
            weights=weights,
        )

        # Save the results
        with open(f"{benchmark_path}/{i}_results.json", "w") as f:
            f.write(benchmark_result.model_dump_json(indent=4))

        row["recall"] = benchmark_result.avg_recall
        row["precision"] = benchmark_result.avg_precision

        # Per-benchmark metrics
        for benchmark_name in benchmark_name_to_weight:
            avg_recall, avg_precision = benchmark_result.get_avg_recall_and_precision(
                benchmark_name
            )
            row[f"{benchmark_name}|recall"] = avg_recall
            row[f"{benchmark_name}|precision"] = avg_precision
            # Optional: print per-benchmark details
            print(f"{benchmark_name} Avg Recall: {100*avg_recall:.2f}%")
            print(f"{benchmark_name} Avg Precision: {100*avg_precision:.2f}%")

        print(f"Overall Avg Recall: {100*benchmark_result.avg_recall:.2f}%")
        print(f"Overall Avg Precision: {100*benchmark_result.avg_precision:.2f}%")
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure all expected columns exist, fill missing with None or NaN
    # This handles cases where strategies have different parameters
    # df = df.reindex(columns=[... list of all possible columns ...])
    df.to_csv(f"{benchmark_path}/results.csv", index=False)

    print(f'\nAll Benchmark runs saved to: "{benchmark_path}"')


if __name__ == "__main__":
    # Ensure asyncio event loop compatibility if needed (e.g., on Windows)
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) # Only if needed
    asyncio.run(main())