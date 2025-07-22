import json
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.sac_rag.data_models import Document
from benchmarks.alqa.src.result_models import ALQATestItem, ALQAGroundTruthInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_alqa_corpus(corpus_path: str) -> List[Document]:
    """
    Loads the Open Australian Legal Corpus from a .jsonl file.

    Args:
        corpus_path (str): The path to the corpus .jsonl file.

    Returns:
        A list of Document objects.
    """
    logger.info(f"Loading ALQA corpus from: {corpus_path}")
    corpus_docs = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Corpus"):
            data = json.loads(line)
            # Use 'version_id' as the unique identifier for the document.
            doc = Document(file_path=f"alqa/{data['version_id']}", content=data['text'])
            corpus_docs.append(doc)
    logger.info(f"Successfully loaded {len(corpus_docs)} documents from the corpus.")
    return corpus_docs


def load_alqa_test_set(qa_path: str, corpus_map: Dict[str, str]) -> Tuple[List[ALQATestItem], Dict[str, str]]:
    """
    Loads the ALQA test set, finds ground-truth snippet spans, and prepares test items.
    This version is more robust, handling cases where documents or snippets are not found.

    Args:
        qa_path (str): The path to the QA dataset .jsonl file.
        corpus_map (Dict[str, str]): A map of document_id to document content.

    Returns:
        A tuple containing:
        - A list of ALQATestItem objects ready for benchmarking.
        - The potentially updated corpus_map.
    """
    logger.info(f"Loading and processing ALQA test set from: {qa_path}")
    test_items: List[ALQATestItem] = []
    skipped_multi_occurrence = 0
    dummy_span_count = 0
    added_to_corpus_count = 0

    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = [json.loads(line) for line in f]

    for i, qa_pair in enumerate(tqdm(qa_data, desc="Processing QA Pairs")):
        source_info = qa_pair.get('source', {})
        doc_id = f"alqa/{source_info.get('version_id')}"
        gt_snippet_text = source_info.get('text')

        if not doc_id or not gt_snippet_text:
            logger.warning(f"Skipping QA item at index {i} due to missing 'version_id' or 'text' in source.")
            continue

        full_doc_text = corpus_map.get(doc_id)
        if not full_doc_text:
            logger.warning(f"Source document '{doc_id}' for item {i} not in corpus. Using ground-truth snippet as the document.")
            full_doc_text = gt_snippet_text
            corpus_map[doc_id] = gt_snippet_text
            added_to_corpus_count += 1

        # Find the span of the ground-truth snippet in the full document
        start_index = full_doc_text.find(gt_snippet_text)

        # Handle case where snippet is not found
        if start_index == -1:
            logger.warning(f"Ground-truth snippet for item {i} not found in document '{doc_id}'. Using dummy span (-1, -1).")
            dummy_span_count += 1
            ground_truth_info = ALQAGroundTruthInfo(
                doc_id=doc_id,
                span=(-1, -1),  # Use dummy values
                text=gt_snippet_text
            )
            test_item = ALQATestItem(
                index=i,
                question=qa_pair['question'],
                answer=qa_pair['answer'],
                ground_truth_info=ground_truth_info
            )
            test_items.append(test_item)
            continue

        # Check for multiple occurrences, which creates ambiguity (still skip these)
        if full_doc_text.find(gt_snippet_text, start_index + 1) != -1:
            logger.warning(
                f"Skipping QA item at index {i}: Ground-truth snippet has multiple occurrences in document '{doc_id}'.")
            skipped_multi_occurrence += 1
            continue

        # Happy path: snippet found exactly once
        end_index = start_index + len(gt_snippet_text)
        ground_truth_info = ALQAGroundTruthInfo(
            doc_id=doc_id,
            span=(start_index, end_index),
            text=gt_snippet_text
        )
        test_item = ALQATestItem(
            index=i,
            question=qa_pair['question'],
            answer=qa_pair['answer'],
            ground_truth_info=ground_truth_info
        )
        test_items.append(test_item)

    logger.info(f"Finished processing test set. Total items to be tested: {len(test_items)}")
    if dummy_span_count > 0:
        logger.warning(f"Created {dummy_span_count} items with dummy spans due to missing snippets.")
    if skipped_multi_occurrence > 0:
        logger.warning(f"Skipped {skipped_multi_occurrence} items due to multiple snippet occurrences.")
    if added_to_corpus_count > 0:
        logger.warning(f"Added {added_to_corpus_count} ground-truth snippets to the corpus as standalone documents.")

    return test_items, corpus_map
