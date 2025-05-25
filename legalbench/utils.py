import csv
import os
import re
from datetime import datetime


def write_verbose_output(output_dir, task_name, model_name, item_indices, original_queries, final_prompts_to_llm,
                         generations, gold_answers, query_responses=None, timestamp=None):
    """Writes detailed output to a CSV file for a given task.
    item_indices: list of original indices from the test_df.
    original_queries: list of query texts from test_df['text'].
    final_prompts_to_llm: list of prompts actually sent to the LLM (with context).
    query_responses: Optional list of QueryResponse objects from retrieval.
    timestamp: Optional timestamp for the output filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_task_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in task_name)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    safe_model_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_model_name)

    filename = os.path.join(output_dir, f"{safe_task_name}_{safe_model_name}_{timestamp}.csv")

    print(f"Writing verbose output to: {filename}")

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Add new fields for RAG details
        fieldnames = ['original_index', 'original_query', 'final_llm_prompt', 'generated_output', 'gold_answer',
                      'num_retrieved_snippets', 'retrieved_snippets_full_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for i in range(len(original_queries)):  # Iterate based on original number of test items
            row_data = {
                'original_index': item_indices[i] if i < len(item_indices) else "N/A",
                'original_query': original_queries[i],
                'final_llm_prompt': final_prompts_to_llm[i] if i < len(final_prompts_to_llm) else "N/A",
                'generated_output': generations[i] if i < len(generations) else "N/A (generation missing)",
                'gold_answer': gold_answers[i] if i < len(gold_answers) else "N/A (gold_answer missing)",
                'num_retrieved_snippets': "N/A",
                'retrieved_snippets_full_text': "N/A"
            }

            if query_responses and i < len(query_responses) and query_responses[i] is not None:
                # from legalbenchrag.benchmark_types import QueryResponse # Ensure QueryResponse type is known
                # current_qr = query_responses[i] # Assuming query_responses[i] is a QueryResponse object or None
                current_qr = query_responses[i]
                if hasattr(current_qr, 'retrieved_snippets'):
                    row_data['num_retrieved_snippets'] = len(current_qr.retrieved_snippets)
                    # Store full_chunk_text of all retrieved snippets, separated by a delimiter
                    # This can be very long. Consider limiting or summarizing if needed.
                    snippets_texts = [
                        f"File: {s.file_path}, Span: {s.span}, Score: {s.score:.4f}, Text: {s.full_chunk_text[:200]}..."
                        # Show first 200 chars
                        for s in current_qr.retrieved_snippets
                    ]
                    row_data['retrieved_snippets_full_text'] = " |SNIP| ".join(
                        snippets_texts) if snippets_texts else "None"
                else:  # If not a QueryResponse object or doesn't have retrieved_snippets
                    row_data['retrieved_snippets_full_text'] = str(current_qr)  # Log the raw object if not as expected

            writer.writerow(row_data)


def write_summary_results(results_dir, model_name, retrieval_strategy, all_task_results, args_params, timestamp=None):
    """Writes summary results and parameters for all tasks to a single CSV file."""
    os.makedirs(results_dir, exist_ok=True)

    safe_model_name = re.sub(r'[\\/*?:"<>|]', "_", model_name)
    safe_model_name = safe_model_name.replace("/", "_")

    filename = os.path.join(results_dir, f"{safe_model_name}_{retrieval_strategy}_{timestamp}.csv")
    print(f"Writing summary results to: {filename}")

    base_fieldnames = ['index', 'task_name', 'model_name', 'retrieval_strategy', 'final_top_k',
                       'temperature', 'max_new_tokens', 'batch_size', 'device']

    result_keys = []
    for task_result in all_task_results.values():
        if isinstance(task_result, dict) and "error" not in task_result:
            result_keys = list(task_result.keys())
            break

    fieldnames = base_fieldnames + result_keys
    if not result_keys:
        if 'result_or_error' not in fieldnames:
            fieldnames.append('result_or_error')

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for i, (task_name, task_result) in enumerate(all_task_results.items()):
            row_data = {
                'index': i,
                'task_name': task_name,
                'model_name': args_params.model_name,
                'retrieval_strategy': retrieval_strategy,
                'final_top_k': args_params.final_top_k,  # Add final_top_k from args
                'temperature': args_params.temperature,
                'max_new_tokens': args_params.max_new_tokens,
                'batch_size': args_params.batch_size,
                'device': args_params.device,
            }
            if isinstance(task_result, dict):
                if "error" in task_result:
                    if 'result_or_error' in fieldnames:
                        row_data['result_or_error'] = f"ERROR: {task_result['error']}"
                else:
                    row_data.update(task_result)
            else:
                if 'result_or_error' in fieldnames:
                    row_data['result_or_error'] = str(task_result)
            writer.writerow(row_data)
