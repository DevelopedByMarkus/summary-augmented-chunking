from abc import ABC, abstractmethod
from collections.abc import Sequence
import os

from pydantic import BaseModel, computed_field, model_validator
from typing_extensions import Self

from sac_rag.utils.utils import sanitize_filename


# max_bridge_gap_len will merge spans that are within max_bridge_gap_len characters of eachother.
def sort_and_merge_spans(
    spans: list[tuple[int, int]], *, max_bridge_gap_len: int = 0
) -> list[tuple[int, int]]:
    if len(spans) == 0:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged_spans = [spans[0]]
    for span in spans[1:]:
        if span[0] <= merged_spans[-1][1] + max_bridge_gap_len:
            merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], span[1]))
        else:
            merged_spans.append(span)
    return merged_spans


class Snippet(BaseModel):
    file_path: str
    span: tuple[int, int]

    @computed_field  # type: ignore[misc]
    @property
    def answer(self) -> str:
        # Logic to find the actual file path (original or sanitized) for reading
        original_full_path = f"./data/corpus/{self.file_path}"
        sanitized_file_path = sanitize_filename(self.file_path)
        sanitized_full_path = f"./data/corpus/{sanitized_file_path}"

        if os.path.exists(original_full_path):
            path_to_read = original_full_path
        elif os.path.exists(sanitized_full_path):
            path_to_read = sanitized_full_path
        else:
            # This *shouldn't* happen if benchmark.py filtering works correctly,
            # as only loadable documents should result in runnable tests.
            # If it does, it signals an inconsistency.
            raise FileNotFoundError(
                f"FATAL: Neither original '{original_full_path}' nor sanitized "
                f"'{sanitized_full_path}' found for Snippet during serialization. "
                f"File path '{self.file_path}' was expected to be loadable based on benchmark filtering."
            )

        with open(path_to_read, encoding='utf-8') as f:
            return f.read()[self.span[0] : self.span[1]]


def validate_snippet_list(snippets: Sequence[Snippet]) -> None:
    snippets_by_file_path: dict[str, list[Snippet]] = {}
    for snippet in snippets:
        if snippet.file_path not in snippets_by_file_path:
            snippets_by_file_path[snippet.file_path] = [snippet]
        else:
            snippets_by_file_path[snippet.file_path].append(snippet)

    for _file_path, snippets_list in snippets_by_file_path.items(): # Renamed variable
        # Sort snippets by start span before checking for overlap
        sorted_snippets = sorted(snippets_list, key=lambda x: x.span[0])
        for i in range(1, len(sorted_snippets)):
            # Allow spans to touch (end == start), but not overlap (end > start)
            if sorted_snippets[i - 1].span[1] > sorted_snippets[i].span[0]:
                raise ValueError(
                    f"Spans are not disjoint for file '{_file_path}'! "
                    f"{sorted_snippets[i - 1].span} VS {sorted_snippets[i].span}"
                )


def validate_snippet_list(snippets: Sequence[Snippet]) -> None:
    snippets_by_file_path: dict[str, list[Snippet]] = {}
    for snippet in snippets:
        if snippet.file_path not in snippets_by_file_path:
            snippets_by_file_path[snippet.file_path] = [snippet]
        else:
            snippets_by_file_path[snippet.file_path].append(snippet)

    for _file_path, snippets in snippets_by_file_path.items():
        snippets = sorted(snippets, key=lambda x: x.span[0])
        for i in range(1, len(snippets)):
            if snippets[i - 1].span[1] >= snippets[i].span[0]:
                raise ValueError(
                    f"Spans are not disjoint! {snippets[i - 1].span} VS {snippets[i].span}"
                )


class QAGroundTruth(BaseModel):
    query: str
    snippets: list[Snippet]
    tags: list[str] = []

    @model_validator(mode="after")
    def validate_snippet_spans(self) -> Self:
        validate_snippet_list(self.snippets)
        return self


class Benchmark(BaseModel):
    tests: list[QAGroundTruth]


# Types for benchmarking a method


class Document(BaseModel):
    file_path: str
    content: str


class RetrievedSnippet(Snippet):
    score: float
    full_chunk_text: str  # The actual text content of the chunk that was retrieved/reranked


class QueryResponse(BaseModel):
    retrieved_snippets: list[RetrievedSnippet]

    @model_validator(mode="after")
    def validate_snippet_spans(self) -> Self:
        # validate_snippet_list(self.retrieved_snippets)
        return self


class RetrievalMethod(ABC):
    @abstractmethod
    async def ingest_document(self, document: Document) -> None:
        """Ingest a document into the retrieval method."""
        ...

    @abstractmethod
    async def sync_all_documents(self) -> None:
        """Enforce synchronization of the documents before running any retrievals."""
        ...

    @abstractmethod
    async def query(self, query: str) -> QueryResponse:
        """Run the retrieval method on the given dataset."""
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup any resources."""
        ...
