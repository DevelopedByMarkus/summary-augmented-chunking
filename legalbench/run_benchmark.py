from tqdm.auto import tqdm
import datasets
import os
import argparse
import torch
import asyncio
import csv
import datetime  # For timestamp in results filename (optional, but good practice)
import re

from legalbench.tasks import TASKS
from legalbench.utils import generate_prompts
from legalbench.evaluation import evaluate
from legalbench.generate import create_generator

_license_cache = {}
# BASE_PROMPT = "base_prompt.txt"
BASE_PROMPT = "claude_prompt.txt"  # refined prompt


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
                    # print(f"Ignoring task: {task_name}") # Can be verbose
                    continue
            else:
                final_selected_tasks.append(task_name)
    else:
        final_selected_tasks = list(candidate_tasks)

    print(f"--- Selected {len(final_selected_tasks)} tasks (see 10 first below): ---\n{final_selected_tasks[:10]}")
    return final_selected_tasks


def write_verbose_output(output_dir, task_name, prompts, generations, gold_answers, model_name):
    """Writes detailed output to a CSV file for a given task."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize task_name for filename if it contains characters not suitable for filenames
    safe_task_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in task_name)
    # Sanitize model_name for filename
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")  # Replace slashes
    safe_model_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_model_name)

    filename = os.path.join(output_dir, f"{safe_task_name}_{safe_model_name}_verbose.csv")

    print(f"Writing verbose output to: {filename}")

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['index', 'prompt', 'generated_output', 'gold_answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(prompts)):
            writer.writerow({
                'index': i,
                'prompt': prompts[i],
                'generated_output': generations[i] if i < len(generations) else "N/A (generation missing)",
                'gold_answer': gold_answers[i] if i < len(gold_answers) else "N/A (gold_answer missing)"
            })


def write_summary_results(results_dir, model_name, retrieval_strategy, all_task_results, args_params):
    """Writes summary results and parameters for all tasks to a single CSV file."""
    os.makedirs(results_dir, exist_ok=True)

    # Sanitize model_name for filename
    safe_model_name = re.sub(r'[\\/*?:"<>|]', "_", model_name)  # More robust sanitization
    safe_model_name = safe_model_name.replace("/", "_")  # Replace slashes specifically if re didn't catch all

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"{safe_model_name}_{retrieval_strategy}_{timestamp}.csv")

    print(f"Writing summary results to: {filename}")

    # Define fieldnames - include common parameters and then flexible result fields
    # The `evaluate` function returns a dictionary, so we'll dynamically get keys from the first valid result.
    base_fieldnames = ['index', 'task_name', 'model_name', 'retrieval_strategy',
                       'temperature', 'max_new_tokens', 'batch_size', 'device']

    # Determine result keys from the first successful task evaluation
    result_keys = []
    for task_result in all_task_results.values():
        if isinstance(task_result, dict) and "error" not in task_result:
            result_keys = list(task_result.keys())
            break

    fieldnames = base_fieldnames + result_keys
    if not result_keys:  # If all tasks errored or no results, add a generic 'result' field
        if 'result' not in fieldnames:  # ensure it's not already there
            fieldnames.append('result_or_error')

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')  # extrasaction='ignore' is safer
        writer.writeheader()

        for i, (task_name, task_result) in enumerate(all_task_results.items()):
            row_data = {
                'index': i,
                'task_name': task_name,
                'model_name': args_params.model_name,
                'retrieval_strategy': retrieval_strategy,
                'temperature': args_params.temperature,
                'max_new_tokens': args_params.max_new_tokens,
                'batch_size': args_params.batch_size,
                'device': args_params.device,
            }
            if isinstance(task_result, dict):
                if "error" in task_result:
                    # If only 'result_or_error' is a field, put error there. Otherwise, it will be ignored by DictWriter.
                    if 'result_or_error' in fieldnames:
                        row_data['result_or_error'] = f"ERROR: {task_result['error']}"
                    else:  # If specific result keys exist, the error won't have a column, print it
                        print(
                            f"Note: Task '{task_name}' had an error: {task_result['error']}. Error not written to dedicated CSV column unless 'result_or_error' is present.")
                        # To ensure something is written even if result_keys are from successful tasks
                        # one could add a generic 'status' column. For now, this will be an empty row for result keys.
                else:
                    row_data.update(task_result)  # Add the specific metric scores
            else:  # Should not happen if evaluate returns dict or we store error dict
                if 'result_or_error' in fieldnames:
                    row_data['result_or_error'] = str(task_result)
                else:
                    print(f"Note: Task '{task_name}' had an unexpected result format: {task_result}")

            writer.writerow(row_data)


def main(args):
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

    # Instantiate the generator once
    try:
        generator = create_generator(args)
    except ValueError as e:
        print(f"Error creating generator: {e}")
        return
    except Exception as e:
        print(f"Unexpected error creating generator for model '{args.model_name}': {e}")
        import traceback
        traceback.print_exc()
        return

    base_task_files_path = "legalbench/tasks"
    verbose_output_dir = "legalbench/output"
    summary_results_dir = "legalbench/results"  # Define summary results directory

    # Determine retrieval strategy (placeholder for now)
    retrieval_strategy = args.retrieval_strategy if hasattr(args, 'retrieval_strategy') else "X"

    all_tasks_results_summary = {}  # To store evaluation results for summary CSV

    print(f"\n--- Running and Evaluating Model: {args.model_name} ---")
    for idx, task_name in enumerate(tasks_to_run, start=1):
        print(f"\nProcessing task ({idx}/{len(tasks_to_run)}): '{task_name}'")
        try:
            # 1. Download the dataset for the task
            dataset_splits = {}
            try:
                dataset_splits["test"] = datasets.load_dataset("nguha/legalbench", task_name, split="test",
                                                               trust_remote_code=True)
            except Exception as e:
                print(f"Could not load 'test' split for {task_name}: {e}. Skipping task.")
                all_tasks_results_summary[task_name] = {"error": f"Dataset load failed - {e}"}
                continue
            # try:
            #     dataset_splits["train"] = datasets.load_dataset("nguha/legalbench", task_name, split="train", trust_remote_code=True)
            # except Exception as e:
            #     print(f"Could not load 'train' split for {task_name}: {e}. Proceeding without train split.")

            # 2. Load base prompt
            prompt_file_path = os.path.join(base_task_files_path, task_name, BASE_PROMPT)
            if not os.path.exists(prompt_file_path):
                print(f"Prompt file not found for {task_name} at {prompt_file_path}. Skipping task.")
                all_tasks_results_summary[task_name] = {"error": "Prompt file not found"}
                continue
            with open(prompt_file_path, encoding='utf-8') as in_file:
                prompt_template = in_file.read()

            # 3. Create full prompts -> here the retrieved context comes in from the RAG
            # For now, assuming generate_prompts takes care of incorporating retrieved contexts.
            # You will need to integrate your RAG retrieval logic here or within generate_prompts.
            test_df = dataset_splits["test"].to_pandas()
            # Your RAG integration for prompts would go here
            # prompts = generate_prompts_with_rag(prompt_template, test_df, retrieved_contexts)
            prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)

            if not prompts or all(p is None for p in prompts):
                print(f"No valid prompts generated for {task_name}. Skipping generation.")
                all_tasks_results_summary[task_name] = {"error": "No valid prompts generated"}
                continue

            # 4. Get LLM Generations
            print(f"Generating {len(prompts)} responses for '{task_name}' using '{args.model_name}'...")
            generation_kwargs = {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
            }
            generations = asyncio.run(generator.generate(prompts, **generation_kwargs))

            if len(generations) != len(prompts):
                print(f"Warning: Number of generations ({len(generations)}) does not match number of prompts ({len(prompts)}). Skipping evaluation for this task.")
                all_tasks_results_summary[task_name] = {"error": "Mismatch in generations and prompts length"}
                continue

            # 5. Evaluation
            gold_answers = test_df["answer"].tolist()
            if args.verbose:
                write_verbose_output(verbose_output_dir, task_name, prompts, generations, gold_answers, args.model_name)
            print(f"Evaluating predictions for {task_name}... ({len(gold_answers)} tests)")
            task_eval_results = evaluate(task_name, generations, gold_answers)
            print(f"Results for {task_name}: {task_eval_results}")
            all_tasks_results_summary[task_name] = task_eval_results

        except Exception as e:
            print(f"An error occurred while processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping task {task_name}")
            all_tasks_results_summary[task_name] = {"error": str(e)}
            continue

    write_summary_results(summary_results_dir, args.model_name, retrieval_strategy, all_tasks_results_summary, args)

    print("\n--- Overall Benchmark Summary ---")
    for task_name, results in all_tasks_results_summary.items():
        print(f"Task: {task_name}, Results: {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM model for selected LegalBench tasks.")
    # Task selection arguments
    parser.add_argument("--license_filter", "-l", type=str, default=None, help="Filter tasks by license.")
    parser.add_argument("--include_tasks", "-i", type=str, nargs='+', default=None,
                        help="List of task names to include.")
    parser.add_argument("--ignore_tasks", "-x", type=str, nargs='+', default=None,
                        help="List of task names to exclude.")

    # Model and Generation arguments
    parser.add_argument("--model_name", "-m", type=str, required=True,
                        help="Name/path of the model (e.g., 'gpt-4o', 'meta-llama/Llama-2-7b-chat-hf').")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for local models (e.g., 'cuda', 'cpu', 'mps').")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for generation.")  # Default 1.0 is often deterministic for greedy. 0.7 for more diverse.
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for local model inference.")

    # Retrieval strategy arguments
    parser.add_argument("--retrieval_strategy", type=str, default="X",
                        help="Name of the retrieval strategy used (default: X for none/baseline).")

    # Other argument
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Write detailed output (index, prompt, generation, gold_answer) to a CSV file for each task.")

    parsed_args = parser.parse_args()

    if not TASKS:
        print("Error: The global TASKS list is empty. Please populate it from legalbench.tasks.")
    else:
        main(parsed_args)
