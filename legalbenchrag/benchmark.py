import asyncio
import datetime as dt
import os
import random
import pandas as pd

from legalbenchrag.benchmark_types import Benchmark, Document, QAGroundTruth, RetrievalMethod
# Import Baseline components
from legalbenchrag.methods.baseline import BaselineRetrievalMethod, RetrievalStrategy as BaselineStrategy
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

# --- Sampling Settings ---
MAX_TESTS_PER_BENCHMARK = 25
SORT_BY_DOCUMENT = True  # Keep True for faster ingestion during testing


async def main() -> None:
    # Set API keys from credentials
    os.environ["OPENAI_API_KEY"] = credentials.ai.openai_api_key.get_secret_value()
    os.environ["COHERE_API_KEY"] = credentials.ai.cohere_api_key.get_secret_value()
    os.environ["VOYAGEAI_API_KEY"] = credentials.ai.voyageai_api_key.get_secret_value()
    # Anthropic key not directly used by baseline/hypa embeddings/rerankers here, but set if needed elsewhere
    # os.environ["ANTHROPIC_API_KEY"] = credentials.ai.anthropic_api_key.get_secret_value()

    # --- Load Data ---
    all_tests: list[QAGroundTruth] = []
    weights: list[float] = []
    document_file_paths_set: set[str] = set()
    used_document_file_paths_set: set[str] = set()
    for benchmark_name, weight in benchmark_name_to_weight.items():
        benchmark_file = f"./data/benchmarks/{benchmark_name}.json"
        if not os.path.exists(benchmark_file):
            print(f"Warning: Benchmark file not found: {benchmark_file}. Skipping.")
            continue
        try:
            with open(benchmark_file, encoding='utf-8') as f:
                benchmark = Benchmark.model_validate_json(f.read())
        except Exception as e:
            print(f"Error loading benchmark {benchmark_name}: {e}")
            continue

        tests = benchmark.tests
        doc_paths_in_benchmark = {snippet.file_path for test in tests for snippet in test.snippets}
        document_file_paths_set.update(doc_paths_in_benchmark)

        # Sampling logic
        sampled_tests = tests
        if 0 < MAX_TESTS_PER_BENCHMARK < len(tests):
            print(f"Sampling {MAX_TESTS_PER_BENCHMARK} tests from {benchmark_name} ({len(tests)} total)")
            tests_with_indices = list(enumerate(tests))
            if SORT_BY_DOCUMENT:
                tests_with_indices = sorted(
                    tests_with_indices,
                    key=lambda item: (  # item[1] is the test object
                        random.seed(item[1].snippets[0].file_path),
                        random.random(),
                    )[1],
                )
            else:
                random.seed(benchmark_name + str(MAX_TESTS_PER_BENCHMARK))  # Seed for reproducibility
                random.shuffle(tests_with_indices)

            # Take the sampled subset
            sampled_tests_with_indices = tests_with_indices[:MAX_TESTS_PER_BENCHMARK]
            # Unzip back into tests list
            if sampled_tests_with_indices:
                sampled_tests = [item[1] for item in sampled_tests_with_indices]  # Get just the test objects
            else:
                sampled_tests = []

        # Update used document paths based on the sampled tests
        used_document_file_paths_set.update(
             {snippet.file_path for test in sampled_tests for snippet in test.snippets}
        )
        for test in sampled_tests:
            test.tags = [benchmark_name]

        all_tests.extend(sampled_tests)
        # Assign correct per-test weight based on the number actually sampled and added
        if sampled_tests:
            num_sampled = len(sampled_tests)
            per_test_weight = weight / num_sampled if num_sampled > 0 else 0
            weights.extend([per_test_weight] * num_sampled)  # Weights list is now parallel to all_tests
        else:
            print(f"Warning: No tests selected for benchmark {benchmark_name} after sampling/filtering.")

    benchmark = Benchmark(tests=all_tests)
    print(f"Total tests selected across all benchmarks: {len(benchmark.tests)}")

    # --- Create Corpus ---
    corpus_docs_to_load = used_document_file_paths_set if SORT_BY_DOCUMENT else document_file_paths_set
    corpus: list[Document] = []
    loaded_corpus_paths = set()
    for document_file_path in sorted(corpus_docs_to_load):
        full_path = f"./data/corpus/{document_file_path}"
        if not os.path.exists(full_path):
            print(f"Warning: Corpus file not found: {full_path}. Skipping.")
            continue
        try:
            with open(full_path, encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    print(f"Warning: Corpus file is empty: {full_path}. Skipping.")
                    continue
                corpus.append(Document(file_path=document_file_path, content=content))
                loaded_corpus_paths.add(document_file_path)
        except Exception as e:
            print(f"Error reading corpus file {full_path}: {e}")

    # Filter tests (and weights) to only include those whose documents were successfully loaded
    original_test_count = len(benchmark.tests)
    filtered_tests: list[QAGroundTruth] = []
    filtered_weights: list[float] = []
    for i, test in enumerate(benchmark.tests):
        # Check if ALL required documents for this test are present in the loaded corpus
        if all(snippet.file_path in loaded_corpus_paths for snippet in test.snippets):
            filtered_tests.append(test)
            filtered_weights.append(weights[i])

    benchmark.tests = filtered_tests
    weights = filtered_weights

    if len(benchmark.tests) != original_test_count:
        print(f"Filtered out {original_test_count - len(benchmark.tests)} tests (and their weights) due to missing/empty corpus files.")

    if not benchmark.tests:
        print("Error: No valid tests remaining after document loading and filtering. Exiting.")
        return  # Exit if no tests can be run

    # --- Prepare Results Storage ---
    run_name = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    benchmark_path = f"./benchmark_results/{run_name}"
    os.makedirs(benchmark_path, exist_ok=True)
    print(f"Benchmark results will be saved to: {benchmark_path}")

    # --- Run Benchmarks ---
    rows: list[dict[str, str | None | int | float]] = []
    # Use Union type hint for the loop variable
    for i, retrieval_strategy in enumerate(ALL_RETRIEVAL_STRATEGIES):
        print(f"\n--- Running Strategy {i+1}/{len(ALL_RETRIEVAL_STRATEGIES)} ---")
        retrieval_method: RetrievalMethod
        row: dict[str, str | None | int | float] = {"i": i}

        # --- Instantiate correct RetrievalMethod and prepare row data ---
        if isinstance(retrieval_strategy, BaselineStrategy):
            print(f"Method Type: Baseline")
            retrieval_method = BaselineRetrievalMethod(retrieval_strategy=retrieval_strategy)
            row.update({
                "method": "baseline",
                "chunk_strategy_name": retrieval_strategy.chunking_strategy.strategy_name,
                "chunk_size": retrieval_strategy.chunking_strategy.chunk_size,
                "embedding_model_company": retrieval_strategy.embedding_model.company,
                "embedding_model_name": retrieval_strategy.embedding_model.model,
                "embedding_topk": retrieval_strategy.embedding_topk,
                "rerank_model_company": retrieval_strategy.rerank_model.company if retrieval_strategy.rerank_model else None,
                "rerank_model_name": retrieval_strategy.rerank_model.model if retrieval_strategy.rerank_model else None,
                "rerank_topk": retrieval_strategy.rerank_topk if retrieval_strategy.rerank_model else None,
                "token_limit": retrieval_strategy.token_limit,
            })
        elif isinstance(retrieval_strategy, HypaStrategy):
            print(f"Method Type: HyPA")
            retrieval_method = HypaRetrievalMethod(strategy=retrieval_strategy)
            row.update({
                "method": "hypa",
                "chunk_strategy_name": retrieval_strategy.chunk_strategy_name, # Add chunk info
                "chunk_size": retrieval_strategy.chunk_size,
                "embedding_model_company": retrieval_strategy.embedding_model.company, # Split company/model
                "embedding_model_name": retrieval_strategy.embedding_model.model,
                "embedding_top_k": retrieval_strategy.embedding_top_k,
                "bm25_top_k": retrieval_strategy.bm25_top_k,
                "fusion_top_k": retrieval_strategy.fusion_top_k,
            })
        else:
            print(f"WARNING: Unknown strategy type at index {i}. Skipping.")
            continue

        # --- Run Benchmark Logic ---
        print(f"Strategy Config: {retrieval_strategy.model_dump()}")
        print(f"Num Documents: {len(corpus)}")
        print(f"Num Corpus Characters: {sum(len(document.content) for document in corpus):,}")
        print(f"Num Queries: {len(benchmark.tests)}")

        try:
            benchmark_result = await run_benchmark(
                benchmark.tests,
                corpus,
                retrieval_method,
                weights=weights,
            )

            # Save individual results
            result_filename = f"{benchmark_path}/{i}_{row.get('method', 'unknown')}_{row.get('embedding_model_name', 'unknown').replace('/','_')}.json"
            with open(result_filename, "w", encoding='utf-8') as f:
                f.write(benchmark_result.model_dump_json(indent=2))  # Use indent=2 for smaller files

            # Add metrics to the summary row
            row["recall"] = benchmark_result.avg_recall
            row["precision"] = benchmark_result.avg_precision

            # Per-benchmark metrics
            for benchmark_name in benchmark_name_to_weight:
                # Only calculate if tests for this benchmark exist in the final set
                if any(benchmark_name in qa_result.qa_gt.tags for qa_result in benchmark_result.qa_result_list):
                    avg_recall, avg_precision = benchmark_result.get_avg_recall_and_precision(benchmark_name)
                    row[f"{benchmark_name}|recall"] = avg_recall
                    row[f"{benchmark_name}|precision"] = avg_precision
                    # Optional: print per-benchmark details
                    print(f"  {benchmark_name} Avg Recall: {100*avg_recall:.2f}%")
                    print(f"  {benchmark_name} Avg Precision: {100*avg_precision:.2f}%")
                else:
                    # Assign NaN or None if no tests for this benchmark were run/valid
                    row[f"{benchmark_name}|recall"] = float('nan')
                    row[f"{benchmark_name}|precision"] = float('nan')

            print(f"Overall Avg Recall: {100*benchmark_result.avg_recall:.2f}%")
            print(f"Overall Avg Precision: {100*benchmark_result.avg_precision:.2f}%")
            rows.append(row)

        except Exception as e:
            print(f"!!!!!!!!!!!! ERROR running benchmark for strategy {i} !!!!!!!!!!!!")
            print(f"Strategy Config: {retrieval_strategy.model_dump()}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # Add a row indicating failure
            row["recall"] = "ERROR"
            row["precision"] = "ERROR"
            rows.append(row)
            # Optional: decide whether to continue with other benchmarks or stop
            # continue # Continue to next strategy
            # break # Stop benchmark run

    # --- Save Overall Results ---
    if rows:
        # Define expected columns based on union of possible keys
        all_keys = set(key for row in rows for key in row.keys())
        # Define a sensible order
        ordered_columns = sorted(list(all_keys), key=lambda x: (
             0 if x=='i' else
             1 if x=='method' else
             2 if x=='chunk' in x else
             3 if x=='embedding' in x else
             4 if x=='rerank' in x or x=='bm25' in x or x=='fusion' in x else
             5 if x=='recall' or x=='precision' else # Overall first
             6 # Per-benchmark metrics last
        ))
        # Ensure main metrics are present
        if 'recall' not in ordered_columns: ordered_columns.append('recall')
        if 'precision' not in ordered_columns: ordered_columns.append('precision')

        df = pd.DataFrame(rows)
        # Reindex to ensure all columns are present and in order, fill missing with NaN
        df = df.reindex(columns=ordered_columns)
        # Sort by index 'i'
        df = df.sort_values(by='i').reset_index(drop=True)

        results_csv_path = f"{benchmark_path}/results_summary.csv"
        df.to_csv(results_csv_path, index=False)
        print(f'\nOverall Benchmark summary saved to: "{results_csv_path}"')
    else:
        print("\nNo benchmark strategies were successfully run or processed.")

    print(f"Benchmark run '{run_name}' finished.")


if __name__ == "__main__":
    # Ensure asyncio event loop compatibility if needed
    # if os.name == 'nt': # Example for Windows
    #      asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
