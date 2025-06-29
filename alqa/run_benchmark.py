import argparse
import asyncio
import os
import pandas as pd
from datetime import datetime
from tqdm.asyncio import tqdm
import torch
import logging

from alqa.src.data import load_alqa_corpus, load_alqa_test_set
from alqa.src.prompts import RAG_PROMPT_TEMPLATE, NO_RAG_PROMPT_TEMPLATE
from alqa.src.evaluation import evaluate_single_item
from alqa.src.result_models import BenchmarkResultRow

from legalbench.src.retrieval import create_retriever
from legalbench.src.generate import create_generator
from legalbenchrag.benchmark_types import Document
from legalbenchrag.utils.ai import AIEmbeddingModel
from legalbenchrag.utils.credentials import credentials


async def main(args):
    # --- Logging Configuration ---
    # Set the logging level based on the verbose flag.
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Suppress overly verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("pysqlite_vec").setLevel(logging.WARNING)
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    # --- Setup Logger ---
    logger = logging.getLogger(__name__)

    start_time = datetime.now()
    print(f"Starting ALQA benchmark run at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Setup: Load data and models ---
    if args.verbose:
        print("Verbose mode enabled.")

    # Set API keys from credentials to environment
    os.environ["OPENAI_API_KEY"] = credentials.ai.openai_api_key.get_secret_value()
    os.environ["COHERE_API_KEY"] = credentials.ai.cohere_api_key.get_secret_value()

    if args.corpus == "debug":
        corpus_docs = load_alqa_corpus("./data/corpus/alqa/open_australian_legal_corpus_first3.jsonl")
        logger.info("Loaded AL corpus with first 3 documents.")
    elif args.corpus == "full":
        corpus_docs = load_alqa_corpus("./data/corpus/alqa/open_australian_legal_corpus_full.jsonl")
    else:
        corpus_docs = load_alqa_corpus("./data/corpus/alqa/alqa_corpus_gt_only.jsonl")

    corpus_map = {doc.file_path: doc.content for doc in corpus_docs}

    if args.corpus == "debug":
        test_items, updated_corpus_map = load_alqa_test_set("./data/benchmarks/open_australian_legal_qa_first3.jsonl", corpus_map)
        logger.info("Loaded ALQA test set with 3 questions for debugging.")
    else:
        test_items, updated_corpus_map = load_alqa_test_set("./data/benchmarks/open_australian_legal_qa_full.jsonl", corpus_map)

    # If the corpus_map was updated, regenerate the list of Document objects for ingestion
    if len(updated_corpus_map) > len(corpus_docs):
        print("Corpus map was updated. Regenerating Document objects for retriever.")
        corpus_docs = [Document(file_path=doc_id, content=text) for doc_id, text in updated_corpus_map.items()]

    if args.max_questions:
        if args.max_questions < 1:
            print(f"Error: --max-questions must be 1 or greater.")
            return
        test_items = test_items[:args.max_questions]
        print(f"Running benchmark on a subset of {len(test_items)} questions.")

    # --- Initialize Retriever (if in RAG mode) ---
    retriever = None
    retrieval_strategy_name_for_logging = args.retrieval_strategy
    if retrieval_strategy_name_for_logging == "X":
        retrieval_strategy_name_for_logging = "None"

    if args.retrieval_strategy and args.retrieval_strategy.upper() != 'X':
        print(f"RAG mode enabled with strategy: {retrieval_strategy_name_for_logging}")

        retriever = create_retriever(args.retrieval_strategy)

        print("Ingesting corpus into retriever...")
        for doc in tqdm(corpus_docs, desc="Ingesting Documents"):
            await retriever.ingest_document(doc)
        await retriever.sync_all_documents()
        print("Corpus ingestion complete.")
    else:
        print("Running in 'No RAG' (baseline) mode.")

    # --- Initialize Generator and Evaluator ---
    llm_generator = create_generator(args)
    eval_embedding_model = AIEmbeddingModel.model_validate(
        {'company': 'openai', 'model': args.eval_embedding_model}
    ) if args.eval_embedding_model else None
    if not eval_embedding_model:
        print("Error: --eval-embedding-model must be specified.")
        return

    # --- Main Benchmark Loop ---
    results_list = []

    progress_bar = tqdm(test_items, desc="Running Benchmark")
    for item in progress_bar:
        retrieved_snippets = []
        retrieved_context_for_prompt = "No context retrieved."

        # RAG Retrieval Step
        if retriever:
            query_response = await retriever.query(item.question)
            retrieved_snippets = query_response.retrieved_snippets[:args.top_k]

            # Format context for the prompt
            context_strings = [f"Snippet {i + 1} from {s.file_path}:\n{s.full_chunk_text}\n" for i, s in
                               enumerate(retrieved_snippets)]
            retrieved_context_for_prompt = "\n".join(context_strings)

        # Prepare prompt and generate answer
        prompt_template = RAG_PROMPT_TEMPLATE if retriever else NO_RAG_PROMPT_TEMPLATE
        final_prompt = prompt_template.format(context=retrieved_context_for_prompt, question=item.question)

        if args.verbose:
            print(f"\n--- Processing Index: {item.index} ---")
            print(f"Final Prompt:\n{final_prompt[:1000]}...")

        # NOTE: The legalbench `generate` function expects a list of prompts.
        # We process one-by-one, so we wrap in a list.
        # The `y_labels` is used by `clean_response`, for ALQA it's free-form so we pass an empty list.
        generation_result = await llm_generator.generate([final_prompt], y_labels=[])
        raw_generated_answer = generation_result[0] if generation_result else ""
        generated_answer = raw_generated_answer  # For ALQA, no special cleaning needed like in multi-choice tasks.

        # Evaluation
        eval_metrics = await evaluate_single_item(
            generated_answer=generated_answer,
            ground_truth_answer=item.answer,
            retrieved_snippets=retrieved_snippets,
            ground_truth_doc_id=item.ground_truth_info.doc_id,
            eval_embedding_model=eval_embedding_model
        )

        # Store results
        result_row = BenchmarkResultRow(
            index=item.index,
            ground_truth_doc_id=item.ground_truth_info.doc_id,
            question=item.question,
            ground_truth_answer=item.answer,
            generated_answer=generated_answer,
            full_model_answer=raw_generated_answer,
            retrieved_context=[s.full_chunk_text for s in retrieved_snippets],
            final_prompt_to_llm=final_prompt,
            answer_similarity_score=eval_metrics.get('answer_similarity_score'),
            retrieval_precision=eval_metrics.get('retrieval_precision'),
            retrieval_recall=eval_metrics.get('retrieval_recall'),
            retrieval_f1_score=eval_metrics.get('retrieval_f1_score'),
            ground_truth_snippet_span=str(item.ground_truth_info.span),
            generator_model=args.model_name,
            retrieval_strategy=args.retrieval_strategy,
            embedding_model_for_retrieval=retriever.retrieval_strategy.embedding_model.model if retriever else None,
            top_k_retrieval=args.top_k,
            eval_embedding_model=args.eval_embedding_model
        )
        results_list.append(result_row)

    # --- Finalize and Save Results ---
    if retriever:
        await retriever.cleanup()
    if hasattr(llm_generator, 'close_http_client'):
        await llm_generator.close_http_client()

    results_df = pd.DataFrame([row.model_dump() for row in results_list])

    # Print overall summary
    print("\n--- Benchmark Complete ---")
    if 'answer_similarity_score' in results_df.columns:
        avg_sim = results_df['answer_similarity_score'].mean()
        print(f"Average Answer Cosine Similarity: {avg_sim:.4f}")
    if 'retrieval_precision' in results_df.columns:
        avg_prec = results_df['retrieval_precision'].mean()
        avg_rec = results_df['retrieval_recall'].mean()
        avg_f1 = results_df['retrieval_f1_score'].mean()
        print(f"Average Retrieval Precision (Doc-Level): {avg_prec:.4f}")
        print(f"Average Retrieval Recall (Doc-Level):    {avg_rec:.4f}")
        print(f"Average Retrieval F1-Score (Doc-Level):  {avg_f1:.4f}")

    # Save to CSV
    ts_str = start_time.strftime('%Y%m%d_%H%M%S')
    mode_str = args.retrieval_strategy if args.retrieval_strategy else "no-rag"
    safe_mode_str = mode_str.replace("/", "_")
    filename = f"alqa_{args.model_name.replace('/', '_')}_{safe_mode_str}_k{args.top_k}_{ts_str}.csv"

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, filename)

    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    end_time = datetime.now()
    print(f"Run finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the ALQA benchmark for RAG evaluation.")

    # Model and Retrieval Config
    parser.add_argument("--model-name", "-m", type=str, required=True, help="Name of the generator LLM.")
    parser.add_argument("--retrieval-strategy", "-r", type=str, default="s-rcts_oai3S_X",
                        choices=["s-rcts_oai3S_X", "s-rcts_LbertB_X", "rcts_oai3S_X", "rcts_LbertB_X", "X"],
                        help="Name of the retrieval strategy (from retrieval.py configs, or 'X' for no retrieval).")
    parser.add_argument("--corpus", "-c", type=str, default="gt_only",
                        choices=["debug", "gt_only", "full"], help="Which corpus to use for the retrieval. Debug uses a small corpus and only 3 qa-pairs for fast pipeline debugging.")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of retrieved snippets to use for context.")
    parser.add_argument("--eval-embedding-model", type=str, default="text-embedding-3-large",
                        help="Name of the embedding model for answer similarity evaluation.")

    # Execution Config
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Maximum number of questions to process. If None, all questions are used.")
    parser.add_argument("--results-dir", type=str, default="./alqa_results",
                        help="Directory to save the output CSV file.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging for debugging.")

    # LLM Generator Config
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for local models.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for LLM generation.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens for LLM generation.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for local LLM inference.")

    args = parser.parse_args()
    asyncio.run(main(args))
