from pydantic import BaseModel
from typing import List, Optional, Tuple


class ALRAGGroundTruthInfo(BaseModel):
    """Holds the ground truth context information for an ALRAG item."""
    doc_id: str
    span: Tuple[int, int]
    text: str


class ALRAGTestItem(BaseModel):
    """Represents a single, processed question-answer item from the ALRAG dataset."""
    index: int
    question: str
    answer: str
    ground_truth_info: ALRAGGroundTruthInfo


class BenchmarkResultRow(BaseModel):
    """Defines the structure for a single row in the final results CSV."""
    # Core Item Data
    index: int
    question: str
    ground_truth_answer: str
    generated_answer: str
    full_model_answer: str  # Raw output from the model before cleaning

    # RAG & Prompt
    retrieved_context: List[str]
    final_prompt_to_llm: str

    # Evaluation Metrics
    answer_similarity_score: Optional[float]
    retrieval_precision: Optional[float]
    retrieval_recall: Optional[float]
    retrieval_f1_score: Optional[float]
    ground_truth_snippet_span: str

    # Configuration
    generator_model: str
    retrieval_strategy: Optional[str]
    embedding_model_for_retrieval: Optional[str]
    top_k_retrieval: Optional[int]
    eval_embedding_model: Optional[str]
