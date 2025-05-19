from tqdm.auto import tqdm
import datasets
import numpy as np
import os
import argparse

from legalbench.tasks import TASKS
from legalbench.utils import generate_prompts
from legalbench.evaluation import evaluate

# Cache to avoid repeated API calls for the same task's license
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
        _license_cache[task_name] = None  # Cache failure to avoid retrying
        return None


def get_selected_tasks(
    license_filter=None,
    include_tasks=None,
    ignore_tasks=None,
):
    """
    Filters TASKS based on license, inclusion list, and exclusion list.

    Args:
        license_filter (str, optional): Only include tasks with this license. Defaults to None.
        include_tasks (list, optional): A list of task names to specifically include.
                                       If None, all tasks (after license filter) are considered.
                                       Defaults to None.
        ignore_tasks (list, optional): A list of task names to specifically exclude.
                                      Defaults to None.

    Returns:
        list: A list of selected task names.
    """
    candidate_tasks = []

    # 1. Start with include_tasks if provided, otherwise all TASKS
    if include_tasks is not None:
        # Validate tasks in include_tasks
        for task_name in include_tasks:
            if task_name not in TASKS:
                print(f"Warning: Task '{task_name}' from 'include_tasks' is not in TASKS. Skipping.")
            else:
                candidate_tasks.append(task_name)
    else:
        candidate_tasks = list(TASKS)

    # 2. Apply license filter (highest priority after forming initial candidates)
    if license_filter:
        licensed_tasks = []
        for task_name in tqdm(candidate_tasks, desc="Checking licenses"):
            task_license = get_task_license(task_name)
            if task_license == license_filter:
                licensed_tasks.append(task_name)
            elif task_name in (include_tasks or []):  # If explicitly included but license mismatch
                print(f"Warning: Task '{task_name}' was in 'include_tasks' but does not match license '{license_filter}' (license: {task_license}). Skipping.")
        candidate_tasks = licensed_tasks

    # 3. Apply ignore_tasks
    final_selected_tasks = []
    if ignore_tasks:
        for task_name in candidate_tasks:
            if task_name in ignore_tasks:
                if include_tasks and task_name in include_tasks:
                    print(f"Warning: Task '{task_name}' is in both 'include_tasks' and 'ignore_tasks'. Prioritizing 'include_tasks' - task will be USED.")
                    final_selected_tasks.append(task_name)
                else:
                    print(f"Ignoring task: {task_name}")
                    continue
            else:
                final_selected_tasks.append(task_name)
    else:
        final_selected_tasks = list(candidate_tasks)

    print(f"--- Selected {len(final_selected_tasks)} tasks (see 10 first below): ---\n{final_selected_tasks[:10]}")
    return final_selected_tasks


def main(args):
    # Supress progress bars for dataset loading if desired
    datasets.utils.logging.set_verbosity_error()

    print("All available tasks (count):", len(TASKS))

    # Select the desired tasks
    tasks_to_run = get_selected_tasks(
        license_filter=args.license_filter,
        include_tasks=args.include_tasks,
        ignore_tasks=args.ignore_tasks,
    )

    if not tasks_to_run:
        print("\nNo tasks selected to run. Exiting.")
        return

    base_task_files_path = "legalbench/tasks"

    print(f"\n--- Running and Evaluating the Model ---")
    for idx, task_name in enumerate(tasks_to_run):
        print(f"\nProcessing task ({idx+1}/{len(tasks_to_run)}): '{task_name}'")
        try:
            # 1. Download the dataset for the task
            dataset_splits = {}
            try:
                dataset_splits["test"] = datasets.load_dataset("nguha/legalbench", task_name, split="test", trust_remote_code=True)
            except Exception as e:
                print(f"Could not load 'test' split for {task_name}: {e}. Skipping task.")
                continue
            try:
                # Train split might not always be used for prompting in zero-shot, but good to load if available/needed
                dataset_splits["train"] = datasets.load_dataset("nguha/legalbench", task_name, split="train", trust_remote_code=True)
            except Exception as e:
                print(f"Could not load 'train' split for {task_name}: {e}. Proceeding without train split.")

            # 2. Load base prompt
            prompt_file_path = os.path.join(base_task_files_path, task_name, "base_prompt.txt")
            if not os.path.exists(prompt_file_path):
                print(f"Prompt file not found for {task_name} at {prompt_file_path}. Skipping task.")
                continue
            with open(prompt_file_path, encoding='utf-8') as in_file:
                prompt_template = in_file.read()

            # 3. Create full prompts (typically for the test set)
            test_df = dataset_splits["test"].to_pandas()
            prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)

            # 4. Get LLM Generations
            # For now, using your random predictions example
            # You need to know the possible output classes for each task if doing classification.
            # The 'answer' column in test_df often contains the gold labels.
            gold_answers = test_df["answer"].tolist()
            unique_answers = list(set(gold_answers))  # Num classes if classification task

            # This random generation is a placeholder!
            # Replace with: generations = your_rag_model.predict(prompts, retrieved_contexts)
            if not unique_answers:  # Handle case with no answers in test_df or empty test_df
                print(f"No unique answers found for {task_name} to make predictions. Skipping evaluation.")
                continue
            generations = np.random.choice(unique_answers, len(test_df))

            # 5. Evaluation
            print(f"Evaluating predictions for {task_name}... ({len(gold_answers)} tests)")
            results = evaluate(task_name, generations, gold_answers)
            print("Results", results)

        except Exception as e:
            print(f"An error occurred while processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            print("Skipping task")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model for selected LegalBench tasks.")
    parser.add_argument(
        "--license_filter",
        "-l",
        type=str,
        default=None,
        help="Filter tasks by a specific license string (e.g., 'CC BY 4.0')."
    )
    parser.add_argument(
        "--include_tasks",
        "-i",
        type=str,
        nargs='+',
        default=None,
        help="A list of task names to specifically include."
    )
    parser.add_argument(
        "--ignore_tasks",
        "-x",
        type=str,
        nargs='+',
        default=None,
        help="A list of task names to specifically exclude."
    )
    parsed_args = parser.parse_args()

    if not TASKS:
        print("Error: The global TASKS list is empty. Please populate it with LegalBench task names.")
    else:
        main(parsed_args)
