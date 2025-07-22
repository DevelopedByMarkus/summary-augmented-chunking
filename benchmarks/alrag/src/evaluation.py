import numpy as np
from typing import List, Dict, Optional, Any

from sac_rag.data_models import RetrievedSnippet
from sac_rag.utils.ai import ai_embedding, AIEmbeddingModel, AIEmbeddingType


async def evaluate_answer_similarity(
        generated_answer: str,
        ground_truth_answer: str,
        eval_model: AIEmbeddingModel
) -> Optional[float]:
    """
    Calculates the cosine similarity between the embeddings of two answers.

    Returns:
        The cosine similarity score, or None if an error occurs.
    """
    if not generated_answer or not ground_truth_answer:
        return 0.0

    try:
        embeddings = await ai_embedding(
            model=eval_model,
            texts=[generated_answer, ground_truth_answer],
            embedding_type=AIEmbeddingType.DOCUMENT  # Use DOCUMENT type for semantic comparison
        )
        if len(embeddings) != 2:
            return None

        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])

        # Cosine similarity = dot(A, B) / (norm(A) * norm(B))
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)

    except Exception as e:
        print(f"Warning: Could not compute answer similarity. Error: {e}")
        return None


def evaluate_retrieval_performance(
        retrieved_snippets: List[RetrievedSnippet],
        ground_truth_doc_id: str
) -> Dict[str, float]:
    """
    Calculates document-level precision, recall, and F1-score for retrieval.
    """
    if not retrieved_snippets:
        # If nothing is retrieved, but something was expected, precision is perfect (vacuously true), but recall is 0.
        return {'retrieval_precision': 1.0, 'retrieval_recall': 0.0, 'retrieval_f1_score': 0.0}

    num_retrieved = len(retrieved_snippets)
    num_correctly_retrieved = sum(1 for s in retrieved_snippets if s.file_path == ground_truth_doc_id)

    precision = num_correctly_retrieved / num_retrieved
    # Recall is 1 if at least one correct document was retrieved, 0 otherwise.
    recall = 1.0 if num_correctly_retrieved > 0 else 0.0

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        'retrieval_precision': precision,
        'retrieval_recall': recall,
        'retrieval_f1_score': f1_score
    }


async def evaluate_single_item(
        generated_answer: str,
        ground_truth_answer: str,
        retrieved_snippets: List[RetrievedSnippet],
        ground_truth_doc_id: str,
        eval_embedding_model: Optional[AIEmbeddingModel]
) -> Dict[str, Any]:
    """
    A wrapper function to perform all evaluations for a single test item.
    """
    results: Dict[str, Any] = {
        'answer_similarity_score': None,
        'retrieval_precision': None,
        'retrieval_recall': None,
        'retrieval_f1_score': None,
    }

    # Generation evaluation
    if eval_embedding_model:
        results['answer_similarity_score'] = await evaluate_answer_similarity(
            generated_answer, ground_truth_answer, eval_embedding_model
        )

    # Retrieval evaluation (only if snippets were retrieved)
    if retrieved_snippets:
        retrieval_metrics = evaluate_retrieval_performance(
            retrieved_snippets, ground_truth_doc_id
        )
        results.update(retrieval_metrics)

    return results
