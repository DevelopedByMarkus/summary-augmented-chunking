import pandas as pd
from typing import List
import csv
import datetime
import re
import os


def generate_prompts(prompt_template: str, data_df: pd.DataFrame) -> List[str]:
    """
    Generates prompts for the rows in data_df using the template prompt_template.

    Args:
        prompt_template: a prompt template
        data_df: pandas dataframe of samples to generate prompts for
    
    Returns:
        prompts: a list of prompts corresponding to the rows of data_df
    """
    assert (
        "{{" in prompt_template
    ), f"Prompt template has no fields to fill, {prompt_template}"

    prompts = []
    dicts = data_df.to_dict(orient="records")
    for dd in dicts:
        prompt = str(prompt_template)
        for k, v in dd.items():
            prompt = prompt.replace("{{" + k + "}}", str(v))
        assert not "{{" in prompt, print(prompt)
        prompts.append(prompt)
    assert len(set(prompts)) == len(prompts), "Duplicated prompts detected"
    return prompts


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
