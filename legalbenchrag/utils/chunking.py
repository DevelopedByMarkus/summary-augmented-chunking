from typing import List, Optional, cast  # Added Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import logging  # For logging

from legalbenchrag.benchmark_types import Document
from legalbenchrag.utils.ai import AIModel, generate_document_summary  # Import new types/functions

logger = logging.getLogger(__name__)  # Setup logger for this module


class Chunk(BaseModel):
    """Represents a text chunk with its source and span."""
    file_path: str
    span: tuple[int, int]  # Span refers to the original content within the document
    content: str  # Final content, possibly prepended with summary


def _chunk_naive(document_content: str, chunk_size: int) -> List[tuple[int, int]]:
    """Creates chunks of fixed character size from the given content."""
    spans = []
    if chunk_size <= 0:
        # logger.warning(f"Naive chunking called with non-positive chunk_size ({chunk_size}). Returning no spans.")
        return spans
    for i in range(0, len(document_content), chunk_size):
        spans.append((i, min(i + chunk_size, len(document_content))))
    return spans


def _chunk_recursive(document_content: str, chunk_size: int, chunk_overlap_ratio: float = 0.0) -> List[tuple[int, int]]:
    """Creates chunks using RecursiveCharacterTextSplitter from the given content."""
    if chunk_size <= 0:
        # logger.warning(f"Recursive chunking called with non-positive chunk_size ({chunk_size}). Returning no spans.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n", "\n", "!", "?", ".", ":", ";", ",", " ", ""
        ],
        chunk_size=chunk_size,  # This size is for the content part
        chunk_overlap=int(chunk_size * chunk_overlap_ratio),  # Overlap for the content part
        length_function=len,
        is_separator_regex=False,
        strip_whitespace=True,  # Changed from False to True as per common Langchain practice
    )
    splits = splitter.split_text(document_content)
    spans = []
    current_search_pos = 0
    for split_text in splits:
        if not split_text.strip():
            continue
        try:
            start_index = document_content.index(split_text, current_search_pos)
            end_index = start_index + len(split_text)
            spans.append((start_index, end_index))
            current_search_pos = start_index + 1
        except ValueError:
            logger.warning(
                f"Could not find exact match for RCTS split text. "  # type: ignore
                f"Split: '{split_text[:50]}...'. This split will be skipped."
            )
            # Attempting to advance current_search_pos naively if a split fails can lead to
            # missing subsequent valid splits or incorrect span calculations.
            # It's safer to skip the problematic split and log it.
            # A more sophisticated approach might use fuzzy matching or diff algorithms,
            # but that adds significant complexity.
    return spans


async def get_chunks(  # Made async
        document: Document,
        strategy_name: str,
        chunk_size: int,  # For summary strategies, this is TOTAL target chunk size
        **kwargs
) -> List[Chunk]:
    """
    Splits a document into chunks based on the specified strategy.
    Supports new strategies that prepend a document summary.
    """
    final_chunks: List[Chunk] = []
    document_summary = ""  # Initialize summary
    is_summary_strategy = strategy_name.startswith("summary_")

    if not is_summary_strategy and cast(Optional[AIModel], kwargs.get("summarization_model")):
        logger.warning("Couldn't find 'summary_' in the chunking strategy name, but contains a summarization_model!")

    if is_summary_strategy:
        summarization_model = cast(Optional[AIModel], kwargs.get("summarization_model"))
        summary_prompt_template = cast(Optional[str], kwargs.get("summary_prompt_template"))
        prompt_target_char_length = cast(int, kwargs.get("prompt_target_char_length", 150))
        summary_truncation_length = cast(int, kwargs.get("summary_truncation_length", 170))
        summaries_base_dir = "./data/summaries"

        if not summarization_model or not summary_prompt_template:
            logger.error(
                f"Summarization model or prompt template not provided for strategy '{strategy_name}' "
                f"on document {document.file_path}. Summary step will be skipped (fallback in generate_document_summary)."
            )
            # generate_document_summary has its own fallback, so we can still call it.
            # It will use the first N chars of the document.
            document_summary = await generate_document_summary(  # Fallback will be used
                document_file_path=document.file_path,
                document_content=document.content,
                # Passing None or incorrect model would cause issues; ensure fallback in generate_document_summary is robust
                summarization_model=AIModel(company="openai", model="gpt-4o-mini"),
                # Dummy, will trigger fallback if openai only
                summary_prompt_template="FALLBACK_DUE_TO_MISSING_CONFIG",  # Indicates issue
                prompt_target_char_length=prompt_target_char_length,
                truncate_char_length=summary_truncation_length,
                summaries_output_dir_base=summaries_base_dir
            )

        else:
            # logger.info(f"Generating summary for document: {document.file_path} using strategy {strategy_name}")
            document_summary = await generate_document_summary(
                document_file_path=document.file_path,
                document_content=document.content,
                summarization_model=summarization_model,
                summary_prompt_template=summary_prompt_template,
                prompt_target_char_length=prompt_target_char_length,
                truncate_char_length=summary_truncation_length,
                summaries_output_dir_base=summaries_base_dir
            )

    # Determine content chunking details
    content_strategy_name = strategy_name.split("summary_")[-1] if is_summary_strategy else strategy_name

    # Define the format for summary prefix
    summary_prefix = f"[document summary] {document_summary}\n[content] " if is_summary_strategy and document_summary else ""
    # If not a summary strategy, or if summary is empty (e.g. after fallback from failed LLM), prefix is empty.
    # If summary strategy AND summary is present, then prefix is used.

    actual_summary_component_len = len(
        summary_prefix)  # This includes the "[document summary]..." and "[content]" parts

    # chunk_size is the TOTAL desired length. Content part should be total minus summary component.
    content_chunk_target_size = chunk_size - actual_summary_component_len

    if is_summary_strategy and content_chunk_target_size <= 0:
        logger.warning(
            f"For document {document.file_path} with strategy {strategy_name}, "
            f"summary component length ({actual_summary_component_len}) "
            f"is >= total chunk_size ({chunk_size}). Resulting content chunks will be empty or not generated. "
            f"Content target size: {content_chunk_target_size}."
        )
        # Allow chunking functions to receive <=0, they should handle it by returning no spans.

    # Perform content chunking based on the original document content
    content_spans: List[tuple[int, int]] = []
    if content_strategy_name == "naive":
        content_spans = _chunk_naive(document.content, content_chunk_target_size)
    elif content_strategy_name == "rcts":
        overlap_ratio = cast(float, kwargs.get("chunk_overlap_ratio", 0.0))  # Use 0.0 as default if not summary_rcts
        content_spans = _chunk_recursive(document.content, content_chunk_target_size, chunk_overlap_ratio=overlap_ratio)
    else:
        raise ValueError(f"Unknown base chunking strategy derived: {content_strategy_name} from {strategy_name}")

    # Construct final Chunk objects
    for span_start, span_end in content_spans:
        if span_end <= span_start:
            continue  # Should be handled by _chunk_xxx if target_size <=0, but defensive check.

        original_content_text = document.content[span_start:span_end]

        # Prepend summary if it's a summary strategy and summary text exists
        if is_summary_strategy and document_summary:
            # Using the refined format from user
            final_chunk_text = f"[document summary] {document_summary}\n[content] {original_content_text}"
        else:
            final_chunk_text = original_content_text

        final_chunks.append(
            Chunk(
                file_path=document.file_path,
                span=(span_start, span_end),
                content=final_chunk_text
            )
        )

    if not final_chunks and len(document.content) > 0 and content_chunk_target_size > 0:  # Added condition
        logger.warning(
            f"No chunks were generated for document {document.file_path} (len: {len(document.content)}) with strategy {strategy_name}, total_chunk_size {chunk_size}, content_target_size {content_chunk_target_size}.")

    return final_chunks
