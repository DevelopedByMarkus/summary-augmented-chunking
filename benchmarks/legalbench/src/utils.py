import csv
import os
import re


def write_verbose_output(output_dir, task_name, model_name, retrieval_config: str, item_indices, original_queries, final_prompts_to_llm,
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

    filename = os.path.join(output_dir, f"{safe_task_name}_{safe_model_name}_{retrieval_config}_{timestamp}.csv")

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
                'retrieved_snippets_full_text': "N/A",
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
            task_name, final_top_k = task_name.rsplit('_k-', 1)
            row_data = {
                'index': i,
                'task_name': task_name,
                'model_name': args_params.model_name,
                'retrieval_strategy': retrieval_strategy,
                'final_top_k': final_top_k,
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


def load_corpus_for_dataset(dataset_id: str, corpus_base_path: str = "./data/corpus") -> list:
    """
    Loads all documents from the specified dataset's subdirectory within the corpus.
    Returns a list of legalbenchrag.benchmark_types.Document objects.
    """
    from sac_rag.data_models import Document  # Local import to avoid circular dependency if this file grows

    dataset_corpus_path = os.path.join(corpus_base_path, dataset_id)
    corpus_docs = []
    if not os.path.isdir(dataset_corpus_path):
        print(f"Warning: Corpus directory not found for dataset '{dataset_id}' at '{dataset_corpus_path}'.")
        return []

    print(f"Loading corpus documents from: {dataset_corpus_path}")
    for filename in os.listdir(dataset_corpus_path):
        # Assuming documents are text files, adjust if other extensions are used (e.g., .json, .md)
        if filename.endswith((".txt", ".md", ".json")):  # Add more extensions if needed
            file_path_in_corpus = os.path.join(dataset_id, filename)  # Relative path for Document object
            full_file_path = os.path.join(dataset_corpus_path, filename)
            try:
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():  # Ensure content is not just whitespace
                    corpus_docs.append(Document(file_path=file_path_in_corpus, content=content))
                else:
                    print(f"Warning: Document '{full_file_path}' is empty. Skipping.")
            except Exception as e:
                print(f"Error reading document '{full_file_path}': {e}. Skipping.")
    print(f"Loaded {len(corpus_docs)} documents for dataset '{dataset_id}'.")
    return corpus_docs


PROMPT_TEMPLATE_WITH_CONTEXT = """You are a legal expert. Please answer the following query considering the provided context information.

[Relevant Context Snippets Start]
{formatted_contexts}
[Relevant Context Snippets End]

[Original Query Start]
{original_query_from_base_template}
[Original Query End]

Answer:
"""


def generate_prompts_with_rag_context(
        base_prompt_template_text: str,  # This is the content of base_prompt.txt or claude_prompt.txt
        query_text_from_dataset: str,  # This is data_df['text'] or similar
        retrieved_context_strings: list[str]  # List of strings, each a retrieved snippet's content
) -> str:
    """
    Generates a final prompt string by incorporating retrieved contexts into a template
    that wraps the original query derived from base_prompt_template_text and query_text_from_dataset.
    """

    # Step 1: Construct the "original query" part.
    # The base_prompt_template_text usually has a placeholder for the specific query/text from the dataset.
    # Let's assume it's {text} or similar. We need to fill that first.
    # This logic should mirror what your existing generate_prompts(prompt_template, data_df) does
    # for a single item before RAG.
    # For simplicity, assuming base_prompt_template_text might contain "{text}"
    # or is structured such that query_text_from_dataset fits into it.
    # If base_prompt_template_text IS the query structure itself:
    original_query_filled = base_prompt_template_text.replace("{text}", query_text_from_dataset)
    # If base_prompt_template_text is more of a system message and query_text_from_dataset is the actual user query:
    # original_query_filled = f"{base_prompt_template_text}\n\nQuery: {query_text_from_dataset}" (Adjust as needed)

    # Step 2: Format the retrieved contexts.
    if not retrieved_context_strings:
        formatted_contexts = "No relevant context snippets were retrieved."
    else:
        # Enumerate contexts for clarity in the prompt
        formatted_contexts = "\n\n".join(
            [f"Snippet {i + 1}: \n{context}" for i, context in enumerate(retrieved_context_strings)]
        )

    # Step 3: Fill the main RAG prompt template
    final_llm_prompt = PROMPT_TEMPLATE_WITH_CONTEXT.format(
        formatted_contexts=formatted_contexts,
        original_query_from_base_template=original_query_filled
    )

    return final_llm_prompt
