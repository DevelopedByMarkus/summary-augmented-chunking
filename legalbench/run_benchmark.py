from tqdm.auto import tqdm
import datasets
import os
import argparse
import torch
import asyncio

from legalbench.tasks import TASKS
from legalbench.utils import generate_prompts
from legalbench.evaluation import evaluate
from legalbench.generate import create_generator

_license_cache = {}


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

    base_task_files_path = "legalbench/tasks"  # Ensure this path is correct relative to script execution

    all_tasks_results = {}  # To store evaluation results for all tasks

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
                continue
            # try:
            #     dataset_splits["train"] = datasets.load_dataset("nguha/legalbench", task_name, split="train", trust_remote_code=True)
            # except Exception as e:
            #     print(f"Could not load 'train' split for {task_name}: {e}. Proceeding without train split.")

            # 2. Load base prompt
            prompt_file_path = os.path.join(base_task_files_path, task_name, "base_prompt.txt")
            if not os.path.exists(prompt_file_path):
                print(f"Prompt file not found for {task_name} at {prompt_file_path}. Skipping task.")
                continue
            with open(prompt_file_path, encoding='utf-8') as in_file:
                prompt_template = in_file.read()

            # 3. Create full prompts -> here the retrieved context comes in from the RAG
            # For now, assuming generate_prompts takes care of incorporating retrieved contexts.
            # You will need to integrate your RAG retrieval logic here or within generate_prompts.
            test_df = dataset_splits["test"].to_pandas()
            # Example: For RAG, you would first retrieve contexts for each item in test_df
            # retrieved_contexts_for_task = [retrieve_context(row['query']) for _, row in test_df.iterrows()]
            # prompts = generate_prompts_with_rag(prompt_template, test_df, retrieved_contexts_for_task)
            prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)

            if not prompts or all(p is None for p in prompts):
                print(f"No valid prompts generated for {task_name}. Skipping generation.")
                continue

            # 4. Get LLM Generations
            print(f"Generating {len(prompts)} responses for '{task_name}' using '{args.model_name}'...")

            generation_kwargs = {
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "device": args.device,
            }
            generations = asyncio.run(generator.generate(prompts, **generation_kwargs))

            if len(generations) != len(prompts):
                print(f"Warning: Number of generations ({len(generations)}) does not match number of prompts ({len(prompts)}). Skipping evaluation for this task.")
                continue

            # 5. Evaluation
            gold_answers = test_df["answer"].tolist()
            print(f"Evaluating predictions for {task_name}... ({len(gold_answers)} tests)")
            task_eval_results = evaluate(task_name, generations, gold_answers)
            print(f"Results for {task_name}: {task_eval_results}")
            all_tasks_results[task_name] = task_eval_results
            # TODO: Store retrieved snippets and answer in the resulting tsv file.

        except Exception as e:
            print(f"An error occurred while processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping task {task_name}")
            all_tasks_results[task_name] = {"error": str(e)}
            continue

    print("\n--- Overall Benchmark Summary ---")
    for task_name, results in all_tasks_results.items():
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
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for local model inference.")

    parsed_args = parser.parse_args()

    if not TASKS:
        print("Error: The global TASKS list is empty. Please populate it from legalbench.tasks.")
    else:
        main(parsed_args)
