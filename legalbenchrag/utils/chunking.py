from typing import List, Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from legalbenchrag.benchmark_types import Document


class Chunk(BaseModel):
    """Represents a text chunk with its source and span."""
    file_path: str
    span: tuple[int, int]
    content: str


def _chunk_naive(document_content: str, chunk_size: int) -> List[tuple[int, int]]:
    """Creates chunks of fixed character size."""
    spans = []
    for i in range(0, len(document_content), chunk_size):
        spans.append((i, min(i + chunk_size, len(document_content))))
    return spans


def _chunk_recursive(document_content: str, chunk_size: int, chunk_overlap_ratio: float = 0) -> List[tuple[int, int]]:
    """Creates chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n", "\n", "!", "?", ".", ":", ";", ",", " ", ""
        ],
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * chunk_overlap_ratio),
        length_function=len,
        is_separator_regex=False,
        strip_whitespace=False,
    )
    # Splitter returns text content, we need to map back to spans
    splits = splitter.split_text(document_content)
    spans = []
    current_pos = 0
    for split in splits:
        start_index = document_content.find(split, current_pos)
        if start_index == -1:
            # Handle cases where split might slightly differ due to stripping or internal logic
            # This is a fallback, ideally the splitter maintains original content slices
            # A more robust approach might involve diffing or tracking indices during splitting
            # For now, we assume direct finding works or we approximate
            # Let's try a search that ignores leading/trailing whitespace differences if find fails
            approx_start_index = document_content.find(split.strip(), current_pos)
            if approx_start_index != -1:
                # Adjust start based on leading whitespace difference
                original_slice = document_content[approx_start_index:]
                leading_ws_orig = len(original_slice) - len(original_slice.lstrip())
                leading_ws_split = len(split) - len(split.lstrip())
                start_index = approx_start_index + (leading_ws_orig - leading_ws_split)
                # Ensure start index is not negative
                start_index = max(0, start_index)
            else:
                # If still not found, raise an error or log a warning
                print(f"Warning: Could not reliably find chunk span for split starting near position {current_pos}. Using approximate span.")
                # Fallback: use previous end + length, might be inaccurate
                start_index = current_pos  # Approximate start

        # Check if the found content actually matches (ignoring potential minor whitespace diffs from splitter)
        end_index = start_index + len(split)
        # Ensure the found text is reasonably close to the split text
        # if document_content[start_index:end_index].strip() != split.strip():
        #     print(f"Warning: Span mismatch detected for chunk. Found '{document_content[start_index:end_index]}' vs Split '{split}'")
            # Decide how to handle mismatch - maybe skip chunk, use approximate, etc.

        # Ensure end_index doesn't exceed document length
        end_index = min(end_index, len(document_content))
        # Ensure start_index doesn't exceed end_index (can happen with empty splits)
        if start_index >= end_index and len(split) > 0:
             print(f"Warning: Calculated invalid span ({start_index}, {end_index}) for non-empty split. Skipping.")
             continue # Skip this potentially problematic chunk
        elif start_index > len(document_content):
             print(f"Warning: Start index {start_index} exceeds document length {len(document_content)}. Skipping.")
             continue


        spans.append((start_index, end_index))
        # Update current_pos for the next search, considering overlap
        # Move forward by chunk size minus overlap to find the next potential start
        current_pos = max(start_index + 1, end_index - int(chunk_size * chunk_overlap_ratio))
        # Ensure current_pos doesn't go backward if chunks are very small/overlap large
        current_pos = max(current_pos, start_index + 1 if len(split) > 0 else start_index)


    # Simple verification: check if total length of spanned content roughly matches original
    # total_spanned_length = sum(end - start for start, end in spans)
    # if abs(total_spanned_length - len(document_content)) > len(document_content) * 0.1: # Allow 10% diff for overlap/approximations
    #      print(f"Warning: Total spanned length {total_spanned_length} differs significantly from original {len(document_content)}")

    return spans


def get_chunks(
    document: Document,
    strategy_name: Literal["naive", "rcts"],
    chunk_size: int,
    **kwargs
) -> List[Chunk]:
    """
    Splits a document into chunks based on the specified strategy.

    Args:
        document: The Document object to chunk.
        strategy_name: The chunking strategy ('naive' or 'rcts').
        chunk_size: The target size for each chunk (characters).
        **kwargs: Additional arguments for specific strategies (e.g., chunk_overlap_ratio for rcts).

    Returns:
        A list of Chunk objects.
    """
    spans: List[tuple[int, int]] = []
    if strategy_name == "naive":
        spans = _chunk_naive(document.content, chunk_size)
    elif strategy_name == "rcts":
        overlap_ratio = kwargs.get("chunk_overlap_ratio", 0.1)  # Default overlap
        spans = _chunk_recursive(document.content, chunk_size, chunk_overlap_ratio=overlap_ratio)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}")

    chunks = [
        Chunk(
            file_path=document.file_path,
            span=span,
            content=document.content[span[0]:span[1]]
        )
        for span in spans if span[1] > span[0]  # Ensure non-empty chunks
    ]

    # Verification (optional but recommended)
    reconstructed_length = sum(len(c.content) for c in chunks)
    original_length = len(document.content)
    # Naive should reconstruct perfectly. Recursive might differ due to overlap/splitting.
    if strategy_name == 'naive' and reconstructed_length != original_length:
        print(f"Warning: Naive chunking reconstruction mismatch. Original: {original_length}, Reconstructed: {reconstructed_length}")

    return chunks
