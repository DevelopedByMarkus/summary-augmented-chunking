import asyncio
from collections.abc import Coroutine
from typing import Any
from tqdm import tqdm
import math

from pydantic import BaseModel, computed_field, model_validator
from typing_extensions import Self

from legalbenchrag.benchmark_types import (
    Document,
    QAGroundTruth,
    RetrievalMethod,
    RetrievedSnippet,
)


class QAResult(BaseModel):
    qa_gt: QAGroundTruth
    retrieved_snippets: list[RetrievedSnippet]

    @computed_field  # type: ignore[misc]
    @property
    def precision(self) -> float:
        total_retrieved_len = 0
        relevant_retrieved_len = 0
        for snippet in self.retrieved_snippets:
            total_retrieved_len += snippet.span[1] - snippet.span[0]
            # It's guaranteed that gt_snippets don't overlap
            for gt_snippet in self.qa_gt.snippets:
                if snippet.file_path == gt_snippet.file_path:
                    common_min = max(snippet.span[0], gt_snippet.span[0])
                    common_max = min(snippet.span[1], gt_snippet.span[1])
                    if common_max > common_min:
                        relevant_retrieved_len += common_max - common_min
        if total_retrieved_len == 0:
            return 0
        return relevant_retrieved_len / total_retrieved_len

    @computed_field  # type: ignore[misc]
    @property
    def recall(self) -> float:
        total_relevant_len = 0
        relevant_retrieved_len = 0
        for gt_snippet in self.qa_gt.snippets:
            total_relevant_len += gt_snippet.span[1] - gt_snippet.span[0]
            # It's guaranteed that gt_snippets don't overlap
            for snippet in self.retrieved_snippets:
                if snippet.file_path == gt_snippet.file_path:
                    common_min = max(snippet.span[0], gt_snippet.span[0])
                    common_max = min(snippet.span[1], gt_snippet.span[1])
                    if common_max > common_min:
                        relevant_retrieved_len += common_max - common_min
        if total_relevant_len == 0:
            return 0
        return relevant_retrieved_len / total_relevant_len


def avg(arr: list[float]) -> float:
    if len(arr) == 0:
        return float("nan")
    return sum(arr) / len(arr)


class BenchmarkResult(BaseModel):
    qa_result_list: list[QAResult]
    weights: list[float]

    def get_avg_recall_and_precision(
        self, tag_filter: str | None = None
    ) -> tuple[float, float]:
        indices = [
            i
            for i, qa_result in enumerate(self.qa_result_list)
            if (tag_filter is None or tag_filter in qa_result.qa_gt.tags)
        ]
        filtered_qa_results = [self.qa_result_list[i] for i in indices]
        filtered_weights = [self.weights[i] for i in indices]

        if not filtered_weights: # If no tests match filter
            return float("nan"), float("nan")

        avg_weight = avg(filtered_weights)

        # Avoid division by zero AND handle NaN from avg() if filtered_weights was empty
        if avg_weight == 0 or math.isnan(avg_weight):
            recall_avg = avg([qa_result.recall for qa_result in filtered_qa_results])
            precision_avg = avg([qa_result.precision for qa_result in filtered_qa_results])
            # avg() already returns nan for empty lists, so this is safe
            return recall_avg, precision_avg

        # Calculate weighted averages
        recall_weighted_avg = avg(
                [
                    qa_result.recall * weight / avg_weight
                    for qa_result, weight in zip(filtered_qa_results, filtered_weights)
                ]
            )
        precision_weighted_avg = avg(
                [
                    qa_result.precision * weight / avg_weight
                    for qa_result, weight in zip(filtered_qa_results, filtered_weights)
                ]
            )
        return recall_weighted_avg, precision_weighted_avg

    @computed_field  # type: ignore[misc]
    @property
    def avg_precision(self) -> float:
        return self.get_avg_recall_and_precision()[1]

    @computed_field  # type: ignore[misc]
    @property
    def avg_recall(self) -> float:
        return self.get_avg_recall_and_precision()[0]

    @computed_field  # type: ignore[misc]
    @property
    def avg_f1_score(self) -> float:
        """Calculates the overall average F1-score based on avg_precision and avg_recall."""
        precision = self.avg_precision
        recall = self.avg_recall

        # Handle cases where precision or recall might be NaN (if no tests were run)
        # Check explicitly for float type AND NaN values
        if isinstance(precision, float) and isinstance(recall, float) and not (
                math.isnan(precision) or math.isnan(recall)):
            # Handle the edge case where precision + recall is 0
            if precision + recall == 0:
                return 0.0
            else:
                return 2 * (precision * recall) / (precision + recall)
        else:
            # If either P or R is NaN, F1 is also undefined
            return float('nan')

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        if len(self.qa_result_list) != len(self.weights):
            raise ValueError("length of qa_result_list and weights do not match!")
        return self


async def run_benchmark(
    qa_gt_list: list[QAGroundTruth],
    corpus: list[Document],
    retrieval_method: RetrievalMethod,
    *,
    weights: list[float] | None = None,
) -> BenchmarkResult:
    # Process the documents
    for document in corpus:
        await retrieval_method.ingest_document(document)
    await retrieval_method.sync_all_documents()

    # Run the benchmark
    # Create a tqdm progress bar instance
    pbar = tqdm(total=len(qa_gt_list), desc="Running Queries", ncols=100)

    async def run_query(qa_gt: QAGroundTruth) -> QAResult:
        query_response = await retrieval_method.query(qa_gt.query)
        return QAResult(
            qa_gt=qa_gt, retrieved_snippets=query_response.retrieved_snippets
        )

    # Define a wrapper coroutine to run the query and update the progress bar
    async def run_query_with_progress(qa_gt: QAGroundTruth) -> QAResult:
        try:
            result = await run_query(qa_gt)
            return result
        finally:
            # Ensure the progress bar updates even if an error occurs in run_query
            pbar.update(1)

    # Create the list of tasks using the wrapper
    tasks: list[Coroutine[Any, Any, QAResult]] = [
        run_query_with_progress(qa_gt) for qa_gt in qa_gt_list
    ]

    # Run the benchmark queries concurrently using asyncio.gather (preserves order)
    try:
        results = await asyncio.gather(*tasks)
    finally:
        # Ensure the progress bar is closed even if gather is cancelled or fails
        pbar.close()

    await retrieval_method.cleanup()

    return BenchmarkResult(
        qa_result_list=results,
        weights=weights if weights is not None else [1.0] * len(results),
    )
