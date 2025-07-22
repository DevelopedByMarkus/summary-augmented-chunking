from tqdm.auto import tqdm
import datasets
import os
import argparse
import torch
import asyncio
from datetime import datetime

from .src.tasks import TASKS
from .src.utils import write_summary_results, write_verbose_output
from .src.evaluation import evaluate
from .src.retrieval import load_corpus_for_dataset, generate_prompts_with_rag_context

from sac_rag.data_models import Document as SacRagDocument
from sac_rag.utils.config_loader import load_strategy_from_file
from sac_rag.utils.retriever_factory import create_retriever
from sac_rag.utils.generate import create_generator

_license_cache = {}
BASE_PROMPT_FILENAME = "claude_prompt.txt"  # refined prompt, or could be "base_prompt.txt"
base_task_files_path = "legalbench/tasks"


def get_task_license(task_name, repo_id="nguha/legalbench"):
    if task_name in _license_cache:
        return _license_cache[task_name]
    try:
        builder = datasets.load_dataset_builder(repo_id, task_name, trust_remote_code=True)
        license_info = builder.info.license
        _license_cache[task_name] = license_info
        return license_info
    except Exception as e:
        print(f"Warning: Could not retrieve license for task '{task_name}': {e}")
        _license_cache[task_name] = None
        return None


def get_selected_tasks(
        license_filter=None,
        include_tasks=None,
        ignore_tasks=None,
):
    candidate_tasks = []
    if include_tasks is not None:
        for task_name in include_tasks:
            if task_name not in TASKS:
                print(f"Warning: Task '{task_name}' from 'include_tasks' is not in TASKS. Skipping.")
            else:
                candidate_tasks.append(task_name)
    else:
        candidate_tasks = list(TASKS)

    if license_filter:
        licensed_tasks = []
        for task_name in tqdm(candidate_tasks, desc="Checking licenses"):
            task_license = get_task_license(task_name)
            if task_license == license_filter:
                licensed_tasks.append(task_name)
            elif task_name in (include_tasks or []):
                print(
                    f"Warning: Task '{task_name}' was in 'include_tasks' but does not match license '{license_filter}' (license: {task_license}). Skipping.")
        candidate_tasks = licensed_tasks

    final_selected_tasks = []
    if ignore_tasks:
        for task_name in candidate_tasks:
            if task_name in ignore_tasks:
                if include_tasks and task_name in include_tasks:
                    print(
                        f"Warning: Task '{task_name}' is in both 'include_tasks' and 'ignore_tasks'. Prioritizing 'include_tasks' - task will be USED.")
                    final_selected_tasks.append(task_name)
                else:
                    continue
            else:
                final_selected_tasks.append(task_name)
    else:
        final_selected_tasks = list(candidate_tasks)

    print(f"--- Selected {len(final_selected_tasks)} tasks (see 10 first below): ---\n{final_selected_tasks[:10]}")
    return final_selected_tasks


async def main(args):
    start_timestamp = datetime.now()
    start_ts_str = start_timestamp.strftime('%Y%m%d_%H%M%S')
    print(f"Start run at (YYYYmmdd_HHMMSS): {start_ts_str}")

    datasets.utils.logging.set_verbosity_error()
    print("All available tasks (count):", len(TASKS))

    tasks_to_run = get_selected_tasks(
        license_filter=args.license_filter,
        include_tasks=args.include_tasks,
        ignore_tasks=args.ignore_tasks,
    )

    if not tasks_to_run:
        print("\nNo tasks selected to run. Exiting.")
        return

    # --- Initialize Generator and Retriever ---
    try:
        llm_generator = create_generator(args)
        retriever = None

        if args.retrieval_config:
            print(f"RAG mode enabled. Loading config from: {args.retrieval_config}")
            retrieval_strategy = load_strategy_from_file(args.retrieval_config)
            retriever = create_retriever(retrieval_strategy)
        else:
            print("No retrieval config provided. Running in 'No RAG' mode.")

    except (ValueError, TypeError) as e:
        print(f"Error creating generator or retriever: {e}")
        return
    except Exception as e:
        import traceback
        print(f"Unexpected error during setup: {e}")
        traceback.print_exc()
        return

    if args.retrieval_config:
        config_basename = os.path.basename(args.retrieval_config)
        retrieval_strategy_name_for_logging, _ = os.path.splitext(config_basename)
    else:
        retrieval_strategy_name_for_logging = "No-RAG"

    all_tasks_results_summary = {}
    retrievers_cache = {}  # Cache for initialized retrievers per top-level dataset_id
    corpus_cache = {}  # Cache for loaded corpus docs per top-level dataset_id

    print(
        f"\n--- Running and Evaluating Model: {args.model_name} with Retrieval: {retrieval_strategy_name_for_logging} ---")

    # Determine loop for final_top_k
    k_values_to_run = args.final_top_k
    if not retriever or k_values_to_run is None or k_values_to_run == [] or k_values_to_run == [0]:
        k_values_to_run = [0]
    else:
        if not isinstance(k_values_to_run, list):  # If single int was passed
            k_values_to_run = [k_values_to_run]

    for current_final_top_k in k_values_to_run:
        if current_final_top_k > 0:
            print(f"\n===== Running with retrieval and final_top_k = {current_final_top_k} =====")
        else:
            print("\n===== Running without retrieval (final_top_k = None) =====")

        for idx, task_name in enumerate(tasks_to_run, start=1):
            # Unique key for results if running multiple k values, to avoid overwriting in all_tasks_results_summary
            task_result_key = f"{task_name}_k-{current_final_top_k}"
            print(f"\n--- Processing task ({idx}/{len(tasks_to_run)}): '{task_name}' for k={current_final_top_k}")

            try:
                dataset_splits = {}
                try:
                    dataset_splits["test"] = datasets.load_dataset("nguha/legalbench", task_name, split="test",
                                                                   trust_remote_code=True)
                except Exception as e:
                    print(f"Could not load 'test' split for {task_name}: {e}. Skipping task.")
                    all_tasks_results_summary[task_result_key] = {"error": f"Dataset load failed - {e}"}
                    continue

                prompt_file_path = os.path.join(base_task_files_path, task_name, BASE_PROMPT_FILENAME)
                if not os.path.exists(prompt_file_path):
                    print(f"Prompt file not found for {task_name} at {prompt_file_path}. Skipping task.")
                    all_tasks_results_summary[task_result_key] = {"error": "Prompt file not found"}
                    continue
                with open(prompt_file_path, encoding='utf-8') as in_file:
                    base_prompt_text_template = in_file.read()

                test_df = dataset_splits["test"].to_pandas()

                # Prepare lists for verbose output
                original_indices_for_verbose = []
                original_queries_for_verbose = []
                final_prompts_for_llm_verbose = []
                query_responses_for_verbose = []  # To store QueryResponse objects

                current_task_prompts_to_llm = []  # Prompts after RAG, for this task

                # Determine top-level dataset_id (e.g., "cuad" from "cuad_...")
                if task_name.startswith("cuad"):
                    dataset_id = "cuad"
                elif task_name.startswith("maud"):
                    dataset_id = "maud"
                elif task_name.startswith("contract_nli"):
                    dataset_id = "contractnli"
                elif task_name.startswith("privacy_policy_qa"):
                    dataset_id = "privacy_qa"
                else:
                    dataset_id = ""  # handled in load_corpus_for_dataset()
                # dataset_id = "cuad_test"  # TODO: Just for debugging

                # --- Retrieval Step ---
                if retriever:  # Only if a retrieval strategy is chosen
                    if dataset_id not in retrievers_cache:
                        print(f"Initializing retriever and corpus for dataset: {dataset_id}")
                        if dataset_id not in corpus_cache:
                            corpus_docs: list[SacRagDocument] = load_corpus_for_dataset(dataset_id)
                            corpus_cache[dataset_id] = corpus_docs
                        else:
                            corpus_docs = corpus_cache[dataset_id]

                        if not corpus_docs:
                            print(f"No corpus documents loaded for {dataset_id}. Retrieval will yield no results.")
                            # Store a dummy retriever or handle this state appropriately
                            # For now, we'll proceed, and retriever.query will likely return empty.
                            # To be robust, one might skip retrieval or use a non-retrieval path.
                            # Storing the retriever instance even if corpus is empty to avoid re-init logic.
                            retrievers_cache[dataset_id] = retriever  # Store the common retriever instance

                        # Ingest documents only if corpus_docs exist for this dataset_id
                        if corpus_docs:
                            active_retriever_for_dataset = retrievers_cache.get(dataset_id)
                            if active_retriever_for_dataset is None or not hasattr(active_retriever_for_dataset,
                                                                                   '_ingested_marker_' + dataset_id):
                                # If retriever not cached for this dataset OR not marked as ingested for this dataset
                                print(
                                    f"Ingesting {len(corpus_docs)} documents for {dataset_id} into {args.retrieval_strategy}...")

                                async def ingest_op():
                                    # Important: If retriever instance is shared, ensure it handles new corpus correctly
                                    # For BaselineRetrievalMethod, it recreates its DB, so it's okay.
                                    # If cleanup is needed between different datasets using the *same* retriever instance:
                                    if hasattr(retriever, 'cleanup_for_new_corpus'):  # Hypothetical method
                                        await retriever.cleanup_for_new_corpus()
                                    elif hasattr(retriever, 'cleanup'):  # General cleanup might reset internal state
                                        await retriever.cleanup()  # Call cleanup if new dataset, to clear old state

                                    for doc in corpus_docs:
                                        await retriever.ingest_document(doc)
                                    await retriever.sync_all_documents()
                                    setattr(retriever, '_ingested_marker_' + dataset_id, True)  # Mark as ingested

                                await ingest_op()
                                retrievers_cache[dataset_id] = retriever  # Cache the ingested retriever

                    current_task_retriever = retrievers_cache.get(dataset_id)

                    if not current_task_retriever or not corpus_cache.get(dataset_id):
                        print(
                            f"Retriever or corpus not available for {dataset_id}. Proceeding without retrieval for this task's items.")
                        for i, row in test_df.iterrows():
                            original_query_text = row['text']
                            final_prompt = generate_prompts_with_rag_context(base_prompt_text_template,
                                                                             original_query_text, [])
                            current_task_prompts_to_llm.append(final_prompt)
                            if args.verbose:
                                original_indices_for_verbose.append(i)
                                original_queries_for_verbose.append(original_query_text)
                                final_prompts_for_llm_verbose.append(final_prompt)
                                query_responses_for_verbose.append(None)  # No QueryResponse object
                    else:
                        print(f"Performing retrieval for {len(test_df)} items in task '{task_name}'...")
                        for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Retrieving contexts"):
                            original_query_text = row['text']
                            retrieved_texts_for_llm_contexts = []
                            query_response_obj = None  # For verbose output

                            try:
                                query_response_obj = await current_task_retriever.query(original_query_text)
                                if query_response_obj and query_response_obj.retrieved_snippets:
                                    for snip_idx, snippet in enumerate(query_response_obj.retrieved_snippets):
                                        if snip_idx < current_final_top_k:
                                            try:
                                                # .answer property reads file; ensure paths are correct
                                                # Or, if retriever provides text directly (e.g., from BaselineRetrievalMethod.get_embedding_info_text)
                                                # We need snippet.answer for "original text indicated by span"
                                                retrieved_texts_for_llm_contexts.append(snippet.answer)
                                            except FileNotFoundError:
                                                print(f"Verbose: File for snippet {snippet.file_path} not found.")
                                                retrieved_texts_for_llm_contexts.append(
                                                    f"[Content for {snippet.file_path} (span {snippet.span}) not found]")
                                            except Exception as e_ans:
                                                print(
                                                    f"Verbose: Error extracting answer for snippet {snippet.file_path}: {e_ans}")
                                                retrieved_texts_for_llm_contexts.append(
                                                    f"[Error extracting content for {snippet.file_path}]")
                                        else:
                                            break  # Stop if we have enough for current_final_top_k
                            except Exception as e_query:
                                print(f"Error during retrieval for query '{original_query_text[:50]}...': {e_query}")
                                # query_response_obj remains None or its last state

                            final_prompt = generate_prompts_with_rag_context(base_prompt_text_template,
                                                                             original_query_text,
                                                                             retrieved_texts_for_llm_contexts)
                            current_task_prompts_to_llm.append(final_prompt)
                            if args.verbose:
                                original_indices_for_verbose.append(i)  # Or row.name if it's the original index
                                original_queries_for_verbose.append(original_query_text)
                                final_prompts_for_llm_verbose.append(final_prompt)
                                query_responses_for_verbose.append(query_response_obj)
                else:  # No retriever chosen (retrieval_strategy == "X")
                    print("No retrieval strategy selected. Generating prompts without retrieved context.")
                    for i, row in test_df.iterrows():
                        original_query_text = row['text']
                        final_prompt = generate_prompts_with_rag_context(base_prompt_text_template, original_query_text,
                                                                         [])  # Empty context list
                        current_task_prompts_to_llm.append(final_prompt)
                        if args.verbose:
                            original_indices_for_verbose.append(i)
                            original_queries_for_verbose.append(original_query_text)
                            final_prompts_for_llm_verbose.append(final_prompt)
                            query_responses_for_verbose.append(None)

                prompts_to_send_to_llm = current_task_prompts_to_llm
                # --- End of Retrieval Step ---

                if not prompts_to_send_to_llm or all(p is None for p in prompts_to_send_to_llm):
                    print(f"No valid prompts after RAG for {task_name}. Skipping generation.")
                    all_tasks_results_summary[task_result_key] = {"error": "No valid prompts after RAG"}
                    continue

                print(
                    f"Generating {len(prompts_to_send_to_llm)} responses for '{task_name}' using '{args.model_name}'...")
                generation_kwargs_llm = {
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                }
                gold_answers = test_df["answer"].tolist()
                generations = await llm_generator.generate(prompts_to_send_to_llm, gold_answers, **generation_kwargs_llm)

                if len(generations) != len(prompts_to_send_to_llm):
                    print(
                        f"Warning: Number of generations ({len(generations)}) does not match number of prompts ({len(prompts_to_send_to_llm)}). Skipping.")
                    all_tasks_results_summary[task_result_key] = {
                        "error": "Mismatch in generations and prompts length after RAG"}
                    continue

                if args.verbose:
                    write_verbose_output(
                        args.prompt_dir, task_name, args.model_name, args.retrieval_strategy, current_final_top_k,
                        original_indices_for_verbose,
                        original_queries_for_verbose,
                        final_prompts_for_llm_verbose,
                        generations, gold_answers,
                        query_responses_for_verbose,
                        start_ts_str,
                    )

                print(f"Evaluating predictions for {task_name}... ({len(gold_answers)} tests)")
                task_eval_results = evaluate(task_name, generations, gold_answers)
                print(f"Results for {task_name} (k={current_final_top_k}): {task_eval_results}")
                all_tasks_results_summary[task_result_key] = task_eval_results

            except Exception as e:
                print(f"An error occurred while processing task {task_name} (k={current_final_top_k}): {e}")
                import traceback
                traceback.print_exc()
                print(f"Skipping task {task_name}")
                all_tasks_results_summary[task_result_key] = {"error": str(e)}
                continue
        # End of k-loop

    # After all k-values and all tasks, write the single summary CSV
    # The `args` object passed to write_summary_results will reflect the last k if modified,
    # or the original args if not. The column 'final_top_k' will capture the specific k for each row.
    # We need to adapt write_summary_results or how we pass data to it if we want one row per (task, k) combo.
    # The current all_tasks_results_summary has keys like "task_k_val".
    # Let's flatten this for the summary writer.

    flattened_summary_for_csv = {}
    for task_k_combo, result in all_tasks_results_summary.items():
        # task_name_part = "_k".join(task_k_combo.split("_k")[:-1]) # E.g. "cuad_something_k10" -> "cuad_something"
        # k_val_part = int(task_k_combo.split("_k")[-1])
        # For now, let's just use the combined key, and the writer can parse it or we add columns
        flattened_summary_for_csv[task_k_combo] = result

    write_summary_results(args.result_dir, args.model_name, retrieval_strategy_name_for_logging,
                          flattened_summary_for_csv, args, timestamp=start_ts_str)

    print("\n--- Overall Benchmark Summary ---")
    for task_k_key, results in all_tasks_results_summary.items():  # Iterate through the collected results
        print(f"Task_K_Config: {task_k_key}, Results: {results}")

    if retriever and hasattr(retriever, 'cleanup'):
        print("Cleaning up retriever...")
        await retriever.cleanup()

    if llm_generator and hasattr(llm_generator, 'close_http_client'):
        await llm_generator.close_http_client()

    from sac_rag.utils.ai import close_all_ai_connections
    await close_all_ai_connections()

    end_timestamp = datetime.now()
    print(f"\nScript finished at: {end_timestamp.strftime('%Y%m%d_%H%M%S')}")
    print(f"Total duration: {end_timestamp - start_timestamp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM model for selected LegalBench tasks with RAG.")
    parser.add_argument("--license_filter", "-l", type=str, default=None, help="Filter tasks by license.")
    parser.add_argument("--include_tasks", "-i", type=str, nargs='+', default=None,
                        help="List of task names to include.")
    parser.add_argument("--ignore_tasks", "-x", type=str, nargs='+', default=None,
                        help="List of task names to exclude.")

    parser.add_argument("--model_name", "-m", type=str, required=True,
                        help="Name/path of the LLM (e.g., 'gpt-4o', 'meta-llama/Llama-2-7b-chat-hf').")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for local models.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for LLM generation.")  # Changed default to 0.7
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens for LLM generation.")  # Increased default
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for local LLM inference.")

    parser.add_argument("--retrieval-config", "-rc", type=str, help="Path to the retrieval strategy JSON config file.")

    parser.add_argument("--final_top_k", type=int, nargs='+', default=[4],
                        help="List of K values for context snippets.")

    parser.add_argument("--result_dir", type=str, default="./results/legalbench")
    parser.add_argument("--prompt_dir", type=str, default="./results/legalbench/output")

    parser.add_argument("--verbose", "-v", action="store_true", help="Write detailed output CSV for each task.")

    parsed_args = parser.parse_args()

    if not TASKS:
        print("Error: The global TASKS list is empty.")
    else:
        asyncio.run(main(parsed_args))
