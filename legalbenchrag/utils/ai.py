import asyncio
import hashlib
import logging
import os
import torch
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any, Literal, cast, Dict, List

import anthropic  # Keep for AIModel definition even if not used for summarization
import cohere
import diskcache as dc
import httpx
import openai
import tiktoken
import voyageai  # type: ignore
import voyageai.error  # type: ignore
from anthropic import NOT_GIVEN, Anthropic, AsyncAnthropic, NotGiven  # Keep
from anthropic.types import MessageParam  # Keep
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, computed_field
from sentence_transformers import SentenceTransformer, CrossEncoder

from legalbenchrag.utils.credentials import credentials

logger = logging.getLogger(__name__)

# --- Globals ---
# Cache for loaded SentenceTransformer models to avoid reloading
local_model_cache: Dict[str, Any] = {}
# Cache for loaded CrossEncoder models
local_reranker_cache: Dict[str, Any] = {}


# AI Types
class AIModel(BaseModel):  # This model is used for summarization
    company: Literal["openai", "anthropic"]  # For summarization, we'll restrict to openai in the function
    model: str

    @computed_field  # type: ignore[misc]
    @property
    def ratelimit_tpm(self) -> float:
        match self.company:
            case "openai":
                # Tier 5
                match self.model:
                    case "gpt-4o-mini":
                        return 150000000
                    case "gpt-4o":
                        return 30000000
                    case m if m.startswith("gpt-4-turbo"):
                        return 2000000
                    case _:
                        return 1000000
            case "anthropic":
                # Tier 4
                return 400000


class AIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AIEmbeddingModel(BaseModel):
    company: Literal["openai", "cohere", "voyageai", "huggingface"]
    model: str

    @computed_field  # type: ignore[misc]
    @property
    def ratelimit_tpm(self) -> float:
        match self.company:
            case "openai":
                return 1000000
            case "cohere":
                # 96 texts per embed
                return 10000 * 96
            case "voyageai":
                # It says 300RPM but I can only get 30 out of it
                return 1000000
            case "huggingface":
                # No external rate limit for local models
                return float('inf')

    @computed_field  # type: ignore[misc]
    @property
    def ratelimit_rpm(self) -> float:
        match self.company:
            case "openai":
                return 5000
            case "cohere":
                return 10000
            case "voyageai":
                # It says 300RPM but I can only get 30 out of it
                return 30
            case "huggingface":
                # No external rate limit for local models
                return float('inf')

    @computed_field  # type: ignore[misc]
    @property
    def max_batch_len(self) -> int:
        match self.company:
            case "openai":
                return 2048
            case "cohere":
                return 96
            case "voyageai":
                return 128
            case "huggingface":
                # Sentence Transformers handles internal batching, this is more informational
                # Or could be used by the caller if they want to manage batches passed to ai_embedding
                return 64  # A reasonable default batch size suggestion


class AIEmbeddingType(Enum):
    DOCUMENT = 1
    QUERY = 2


class AIRerankModel(BaseModel):
    company: Literal["cohere", "voyageai", "huggingface"]
    model: str

    @computed_field  # type: ignore[misc]
    @property
    def ratelimit_rpm(self) -> float:
        match self.company:
            case "cohere":
                return 10000
            case "voyageai":
                # It says 100RPM but I can only get 60 out of it
                return 60
            case "huggingface":
                # Local models have no external rate limit
                return float('inf')


# Cache
os.makedirs("./data/cache", exist_ok=True)
cache = dc.Cache("./data/cache/ai_cache.db")  # Main cache for embeddings, reranks, and now summaries

RATE_LIMIT_RATIO = 0.95


class AIConnection:
    openai_client: AsyncOpenAI
    voyageai_client: voyageai.AsyncClient
    cohere_client: cohere.AsyncClient
    anthropic_client: AsyncAnthropic
    sync_anthropic_client: Anthropic
    # Share one global Semaphore across all threads
    cohere_ratelimit_semaphore = asyncio.Semaphore(1)
    voyageai_ratelimit_semaphore = asyncio.Semaphore(1)
    openai_ratelimit_semaphore = asyncio.Semaphore(1)
    anthropic_ratelimit_semaphore = asyncio.Semaphore(1)

    def __init__(self) -> None:
        self.openai_client = AsyncOpenAI(
            api_key=credentials.ai.openai_api_key.get_secret_value()
        )
        self.anthropic_client = AsyncAnthropic(  # Keep for AIModel consistency
            api_key=credentials.ai.anthropic_api_key.get_secret_value()
        )
        self.sync_anthropic_client = Anthropic(  # Keep for AIModel consistency
            api_key=credentials.ai.anthropic_api_key.get_secret_value()
        )
        self.voyageai_client = voyageai.AsyncClient(
            api_key=credentials.ai.voyageai_api_key.get_secret_value()
        )
        self.cohere_client = cohere.AsyncClient(
            api_key=credentials.ai.cohere_api_key.get_secret_value()
        )


# NOTE: API Clients cannot be called from multiple event loops,
# So every asyncio event loop needs its own API connection
ai_connections: dict[asyncio.AbstractEventLoop, AIConnection] = {}


def get_ai_connection() -> AIConnection:
    event_loop = asyncio.get_event_loop()
    if event_loop not in ai_connections:
        ai_connections[event_loop] = AIConnection()
    return ai_connections[event_loop]


class AIError(Exception):
    """A class for AI Task Errors"""


class AIValueError(AIError, ValueError):
    """A class for AI Value Errors"""


class AITimeoutError(AIError, TimeoutError):
    """A class for AI Task Timeout Errors"""


def ai_num_tokens(model: AIModel | AIEmbeddingModel | AIRerankModel, s: str) -> int:
    if isinstance(model, AIModel):  # This is used by ai_call which is used by summarizer
        if model.company == "anthropic":
            return get_ai_connection().sync_anthropic_client.count_tokens(s)
        elif model.company == "openai":
            encoding = tiktoken.encoding_for_model(model.model)
            num_tokens = len(encoding.encode(s))
            return num_tokens
    if isinstance(model, AIEmbeddingModel):
        if model.company == "openai":
            encoding = tiktoken.encoding_for_model(model.model)
            num_tokens = len(encoding.encode(s))
            return num_tokens
        elif model.company == "voyageai":
            return get_ai_connection().voyageai_client.count_tokens([s], model.model)
        elif model.company == "huggingface":
            return int(len(s) / 4)
    if isinstance(model, AIRerankModel) and model.company == 'huggingface':
        return int(len(s) / 4)

    logger.warning(f"Estimating Tokens for model type {type(model)}!")
    return int(len(s) / 4)


def get_call_cache_key(  # Used by ai_call, which is used by summarizer
        model: AIModel,
        messages: list[AIMessage],
) -> str:
    md5_hasher = hashlib.md5()
    md5_hasher.update(model.model_dump_json().encode())
    for message in messages:
        md5_hasher.update(md5_hasher.hexdigest().encode())
        md5_hasher.update(message.model_dump_json().encode())
    key = md5_hasher.hexdigest()
    return key


async def ai_call(  # Used by summarizer
        model: AIModel,
        messages: list[AIMessage],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        anthropic_initial_message: str | None = "<START>",
        anthropic_combine_delimiter: str = "\n",
        num_ratelimit_retries: int = 10,
        backoff_algo: Callable[[int], float] = lambda i: min(2 ** i, 5),
) -> str:
    cache_key = get_call_cache_key(model, messages)
    cached_call = cache.get(cache_key)

    if cached_call is not None:
        return cached_call

    num_tokens_input: int = sum(
        [ai_num_tokens(model, message.content) for message in messages]
    )

    return_value: str | None = None
    match model.company:
        case "openai":  # This is the only branch used by summarizer
            for i in range(num_ratelimit_retries):
                try:
                    async with get_ai_connection().openai_ratelimit_semaphore:
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = num_tokens_input / (tpm / 60)
                        await asyncio.sleep(expected_wait)

                    def ai_message_to_openai_message_param(
                            message: AIMessage,
                    ) -> ChatCompletionMessageParam:
                        if message.role == "system":
                            return {"role": message.role, "content": message.content}
                        elif message.role == "user":
                            return {"role": message.role, "content": message.content}
                        elif message.role == "assistant":
                            return {"role": message.role, "content": message.content}
                        # Added to handle potential errors if other roles are passed
                        raise AIValueError(f"Unsupported message role for OpenAI: {message.role}")

                    if i > 0:
                        logger.debug("Trying again after RateLimitError...")
                    response = (
                        await get_ai_connection().openai_client.chat.completions.create(
                            model=model.model,
                            messages=[  # type: ignore
                                ai_message_to_openai_message_param(message)  # type: ignore
                                for message in messages
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    )
                    assert response.choices[0].message.content is not None
                    return_value = response.choices[0].message.content
                    break
                except RateLimitError:
                    logger.warning("OpenAI RateLimitError")
                    async with get_ai_connection().openai_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if return_value is None:
                raise AITimeoutError("Cannot overcome OpenAI RateLimitError")

        case "anthropic":  # Keep for other uses of ai_call, even if not for summary
            for i in range(num_ratelimit_retries):
                try:
                    async with get_ai_connection().anthropic_ratelimit_semaphore:
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = num_tokens_input / (tpm / 60)
                        await asyncio.sleep(expected_wait)

                    def ai_message_to_anthropic_message_param(
                            message: AIMessage,
                    ) -> MessageParam:
                        if message.role == "user" or message.role == "assistant":
                            return {"role": message.role, "content": message.content}
                        elif message.role == "system":  # System message handled separately by Anthropic client
                            raise AIValueError(
                                "system not allowed in anthropic message param list, provide separately"
                            )
                        raise AIValueError(f"Unsupported message role for Anthropic: {message.role}")

                    if i > 0:
                        logger.debug("Trying again after RateLimitError...")

                    system_prompt_content: str | NotGiven = NOT_GIVEN
                    processed_messages = list(messages)  # Create a mutable copy

                    if processed_messages and processed_messages[0].role == "system":
                        system_prompt_content = processed_messages[0].content
                        processed_messages = processed_messages[1:]

                    if (anthropic_initial_message is not None and
                            (not processed_messages or processed_messages[0].role != "user")):
                        processed_messages = [
                                                 AIMessage(role="user", content=anthropic_initial_message)
                                             ] + processed_messages

                    combined_messages_for_anthropic: list[MessageParam] = []
                    if processed_messages:
                        current_message_content = processed_messages[0].content
                        current_role = processed_messages[0].role
                        for next_message in processed_messages[1:]:
                            if next_message.role == current_role:
                                current_message_content += anthropic_combine_delimiter + next_message.content
                            else:
                                combined_messages_for_anthropic.append(ai_message_to_anthropic_message_param(
                                    AIMessage(role=current_role, content=current_message_content)))  # type: ignore
                                current_message_content = next_message.content
                                current_role = next_message.role
                        combined_messages_for_anthropic.append(ai_message_to_anthropic_message_param(
                            AIMessage(role=current_role, content=current_message_content)))  # type: ignore

                    response_message = (
                        await get_ai_connection().anthropic_client.messages.create(
                            model=model.model,
                            system=system_prompt_content,  # Pass system prompt here
                            messages=combined_messages_for_anthropic,  # type: ignore
                            temperature=0.0,  # temperature was already a parameter
                            max_tokens=max_tokens,
                        )
                    )
                    assert isinstance(
                        response_message.content[0], anthropic.types.TextBlock
                    )
                    assert isinstance(response_message.content[0].text, str)
                    return_value = response_message.content[0].text
                    break
                except anthropic.RateLimitError as e:
                    logger.warning(f"Anthropic Error: {repr(e)}")
                    async with get_ai_connection().anthropic_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if return_value is None:
                raise AITimeoutError("Cannot overcome Anthropic RateLimitError")

    cache.set(cache_key, return_value)
    return return_value  # type: ignore


def get_embeddings_cache_key(
        model: AIEmbeddingModel, text: str, embedding_type: AIEmbeddingType
) -> str:
    key = f"{model.company}||||{model.model}||||{embedding_type.name}||||{hashlib.md5(text.encode()).hexdigest()}"
    return key


def _encode_local_huggingface(
        model_name: str,
        texts: list[str],
        embedding_type: AIEmbeddingType,
        callback: Callable[[], None],
        trust_remote_code: bool = True
) -> list[list[float]]:
    if SentenceTransformer is None:
        raise ImportError("SentenceTransformer is not installed. Run `pip install sentence-transformers`.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name not in local_model_cache:
        logger.info(f"Loading local SentenceTransformer model: {model_name} onto device: {device}")
        try:
            local_model_cache[model_name] = SentenceTransformer(
                model_name,
                trust_remote_code=trust_remote_code,
                device=device
            )
            logger.info(f"HuggingFace: Finished loading {model_name}")
        except Exception as e:
            logger.error(f"HuggingFace: Failed to load model {model_name} onto device {device}: {e}")
            raise RuntimeError(f"Failed to load SentenceTransformer model {model_name}") from e

    model: SentenceTransformer = local_model_cache[model_name]
    texts_to_encode = texts
    if "bge-" in model_name.lower():
        if embedding_type == AIEmbeddingType.QUERY:
            texts_to_encode = ["Represent this sentence for searching relevant passages: " + text for text in texts]
    logger.debug(f"Encoding {len(texts_to_encode)} texts locally using {model_name}...")
    embeddings = model.encode(
        texts_to_encode,
        show_progress_bar=False,
        batch_size=32
    )
    logger.debug(f"Finished local encoding with {model_name}.")
    embeddings_list = cast(List[List[float]], embeddings.tolist())
    for _ in range(len(texts)):
        callback()
    return embeddings_list


async def ai_embedding(
        model: AIEmbeddingModel,
        texts: list[str],
        embedding_type: AIEmbeddingType,
        *,
        num_ratelimit_retries: int = 10,
        backoff_algo: Callable[[int], float] = lambda i: min(2 ** i, 5),
        callback: Callable[[], None] = lambda: None,
) -> list[list[float]]:
    if not texts:
        return []
    text_embeddings: list[list[float] | None] = [None] * len(texts)
    indices_to_fetch = []
    for i, text in enumerate(texts):
        cache_key = get_embeddings_cache_key(model, text, embedding_type)
        cached_embedding = cache.get(cache_key)
        if cached_embedding is not None:
            text_embeddings[i] = cached_embedding
            callback()
        else:
            indices_to_fetch.append(i)
    if not indices_to_fetch:
        return cast(list[list[float]], text_embeddings)
    required_texts = [texts[i] for i in indices_to_fetch]
    if model.company != 'huggingface' and len(required_texts) > model.max_batch_len:
        tasks: list[Coroutine[Any, Any, list[list[float]]]] = []
        for i in range(0, len(indices_to_fetch), model.max_batch_len):
            batch_indices = indices_to_fetch[i: i + model.max_batch_len]
            batch_texts = [texts[idx] for idx in batch_indices]
            tasks.append(
                ai_embedding(
                    model,
                    batch_texts,
                    embedding_type,
                    num_ratelimit_retries=num_ratelimit_retries,
                    backoff_algo=backoff_algo,
                    callback=callback,
                )
            )
        preflattened_results = await asyncio.gather(*tasks)
        results: list[list[float]] = []
        for embeddings_list in preflattened_results:
            results.extend(embeddings_list)
        assert len(indices_to_fetch) == len(
            results), f"Batch result length mismatch: expected {len(indices_to_fetch)}, got {len(results)}"
        for i, embedding in zip(indices_to_fetch, results):
            text_embeddings[i] = embedding
        assert all(text_embeddings[i] is not None for i in indices_to_fetch)
        return cast(list[list[float]], text_embeddings)
    embeddings_response: list[list[float]] | None = None
    num_tokens_input = sum(ai_num_tokens(model, text) for text in required_texts)
    if model.company == 'huggingface':
        embeddings_response = await asyncio.to_thread(
            _encode_local_huggingface,
            model.model,
            required_texts,
            embedding_type,
            callback
        )
    elif model.company == "openai":
        for i in range(num_ratelimit_retries):
            try:
                async with get_ai_connection().openai_ratelimit_semaphore:
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                    expected_wait = max(60.0 / rpm if rpm != float('inf') else 0,
                                        num_tokens_input / (tpm / 60) if tpm != float('inf') else 0)
                    await asyncio.sleep(expected_wait)
                response = await get_ai_connection().openai_client.embeddings.create(
                    input=required_texts, model=model.model
                )
                embeddings_response = [embedding.embedding for embedding in response.data]
                for _ in range(len(required_texts)):
                    callback()
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
                logger.warning(f"OpenAI Embedding Error: {e}")
                if i == num_ratelimit_retries - 1:
                    raise AITimeoutError("Cannot overcome OpenAI Embedding Error")
                async with get_ai_connection().openai_ratelimit_semaphore:
                    await asyncio.sleep(backoff_algo(i))
    elif model.company == "cohere":
        for i in range(num_ratelimit_retries):
            try:
                async with get_ai_connection().cohere_ratelimit_semaphore:
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                    expected_wait = max(60.0 / rpm if rpm != float('inf') else 0,
                                        num_tokens_input / (tpm / 60) if tpm != float('inf') else 0)
                    await asyncio.sleep(expected_wait)
                result = await get_ai_connection().cohere_client.embed(
                    texts=required_texts, model=model.model,
                    input_type="search_document" if embedding_type == AIEmbeddingType.DOCUMENT else "search_query"
                )
                embeddings_response = cast(List[List[float]], result.embeddings)
                for _ in range(len(required_texts)):
                    callback()
                break
            except (cohere.errors.TooManyRequestsError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
                logger.warning(f"Cohere Embedding Error: {e}")
                if i == num_ratelimit_retries - 1:
                    raise AITimeoutError("Cannot overcome Cohere Embedding Error")
                async with get_ai_connection().cohere_ratelimit_semaphore:
                    await asyncio.sleep(backoff_algo(i))
    elif model.company == "voyageai":
        for i in range(num_ratelimit_retries):
            try:
                async with get_ai_connection().voyageai_ratelimit_semaphore:
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                    expected_wait = max(60.0 / rpm if rpm != float('inf') else 0,
                                        num_tokens_input / (tpm / 60) if tpm != float('inf') else 0)
                    await asyncio.sleep(expected_wait)
                result = await get_ai_connection().voyageai_client.embed(
                    required_texts, model=model.model,
                    input_type="document" if embedding_type == AIEmbeddingType.DOCUMENT else "query"
                )
                embeddings_response = cast(List[List[float]], result.embeddings)
                for _ in range(len(required_texts)):
                    callback()
                break
            except voyageai.error.RateLimitError as e:
                logger.warning(f"VoyageAI Embedding Error: {e}")
                if i == num_ratelimit_retries - 1:
                    raise AITimeoutError("Cannot overcome VoyageAI Embedding Error")
                async with get_ai_connection().voyageai_ratelimit_semaphore:
                    await asyncio.sleep(backoff_algo(i))
    if embeddings_response is None:
        raise AITimeoutError(f"Failed to get embeddings for model {model.model} after retries.")
    assert len(embeddings_response) == len(indices_to_fetch), \
        f"Mismatch between requested ({len(indices_to_fetch)}) and received ({len(embeddings_response)}) embeddings."
    for i, embedding in zip(indices_to_fetch, embeddings_response):
        cache_key = get_embeddings_cache_key(model, texts[i], embedding_type)
        cache.set(cache_key, embedding)
        text_embeddings[i] = embedding
    assert all(embedding is not None for embedding in text_embeddings)
    return cast(list[list[float]], text_embeddings)


def get_rerank_cache_key(
        model: AIRerankModel, query: str, texts: list[str]
) -> str:
    md5_hasher = hashlib.md5()
    md5_hasher.update(query.encode())
    for text in texts:
        md5_hasher.update(md5_hasher.hexdigest().encode())
        md5_hasher.update(text.encode())
    texts_hash = md5_hasher.hexdigest()
    key = f"{model.company}||||{model.model}||||{texts_hash}"
    return key


def _rerank_local_huggingface(
        model_name: str,
        query: str,
        texts: list[str],
) -> list[int]:
    if CrossEncoder is None:
        raise ImportError(
            "CrossEncoder (part of sentence-transformers) is not installed. Run `pip install sentence-transformers`.")
    if model_name not in local_reranker_cache:
        logger.info(f"Loading local CrossEncoder model: {model_name}")
        local_reranker_cache[model_name] = CrossEncoder(model_name, trust_remote_code=True)
        logger.info(f"Finished loading {model_name}")
    model: CrossEncoder = local_reranker_cache[model_name]
    input_pairs = [(query, text) for text in texts]
    logger.debug(f"Reranking {len(input_pairs)} pairs locally using {model_name}...")
    scores = model.predict(
        input_pairs,
        show_progress_bar=False,
        batch_size=8 if "large" in model_name else 32
    )
    logger.debug(f"Finished local reranking with {model_name}.")
    indexed_scores = list(enumerate(scores))
    sorted_indices_scores = sorted(indexed_scores, key=lambda item: item[1], reverse=True)
    reranked_indices = [index for index, score in sorted_indices_scores]
    return reranked_indices


async def ai_rerank(
        model: AIRerankModel,
        query: str,
        texts: list[str],
        *,
        top_k: int | None = None,
        num_ratelimit_retries: int = 10,
        backoff_algo: Callable[[int], float] = lambda i: min(2 ** i, 5),
) -> list[int]:
    if not texts:
        return []
    cache_key = get_rerank_cache_key(model, query, texts)
    cached_full_reranking = cache.get(cache_key)
    full_reranked_indices: list[int] | None = None
    if cached_full_reranking is not None:
        logger.debug(f"Cache hit for rerank key: {cache_key[:20]}...")
        full_reranked_indices = cached_full_reranking
    else:
        logger.debug(f"Cache miss for rerank key: {cache_key[:20]}... Calculating.")
        if model.company == "huggingface":
            try:
                full_reranked_indices = await asyncio.to_thread(
                    _rerank_local_huggingface,
                    model.model,
                    query,
                    texts,
                )
            except Exception as e:
                logger.error(f"HuggingFace Rerank Error ({model.model}): {e}")
                full_reranked_indices = []
        elif model.company == "cohere":
            for i in range(num_ratelimit_retries):
                try:
                    async with get_ai_connection().cohere_ratelimit_semaphore:
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        await asyncio.sleep(60.0 / rpm if rpm != float('inf') else 0)
                    response = await get_ai_connection().cohere_client.rerank(
                        model=model.model, query=query, documents=texts, top_n=len(texts),
                    )
                    full_reranked_indices = [result.index for result in response.results]
                    break
                except (cohere.errors.TooManyRequestsError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    logger.warning(f"Cohere Rerank Error: {e}")
                    if i == num_ratelimit_retries - 1:
                        raise AITimeoutError("Cannot overcome Cohere Rerank Error")
                    async with get_ai_connection().cohere_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if full_reranked_indices is None:
                raise AITimeoutError("Cannot overcome Cohere Rerank Error")
        elif model.company == "voyageai":
            for i in range(num_ratelimit_retries):
                try:
                    async with get_ai_connection().voyageai_ratelimit_semaphore:
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        await asyncio.sleep(60.0 / rpm if rpm != float('inf') else 0)
                    voyageai_response = await get_ai_connection().voyageai_client.rerank(
                        query=query, documents=texts, model=model.model, top_k=len(texts),
                    )
                    full_reranked_indices = [int(result.index) for result in voyageai_response.results]
                    break
                except voyageai.error.RateLimitError as e:
                    logger.warning(f"VoyageAI Rerank Error: {e}")
                    if i == num_ratelimit_retries - 1: raise AITimeoutError("Cannot overcome VoyageAI Rerank Error")
                    async with get_ai_connection().voyageai_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i))
            if full_reranked_indices is None:
                raise AITimeoutError("Cannot overcome VoyageAI Rerank Error")
        if full_reranked_indices is None:
            logger.error(f"Full reranking failed unexpectedly for model {model.company}/{model.model}")
            full_reranked_indices = []
        cache.set(cache_key, full_reranked_indices)
        logger.debug(f"Cached full rerank list for key: {cache_key[:20]}...")
    final_indices = full_reranked_indices
    if top_k is not None:
        top_k = min(top_k, len(final_indices))
        if top_k > 0:
            final_indices = final_indices[:top_k]
        else:
            final_indices = []
    return final_indices


# --- New Summarization Functions ---
def get_document_summary_cache_key(
        document_file_path: str,
        document_content_hash: str,
        summarization_model: AIModel,
        summary_prompt_template_hash: str  # Hash of the prompt template
) -> str:
    """Generates a unique cache key for a document summary."""
    return f"summary|||{document_file_path}|||{document_content_hash}|||{summarization_model.company}|||{summarization_model.model}|||{summary_prompt_template_hash}"


async def generate_document_summary(
        document_file_path: str,
        document_content: str,
        summarization_model: AIModel,  # Expecting OpenAI model as per user
        summary_prompt_template: str,
        prompt_target_char_length: int,
        truncate_char_length: int,
        summaries_output_dir_base: str,  # Base directory for saving .txt summaries
        num_ratelimit_retries: int = 5,
        backoff_algo: Callable[[int], float] = lambda i: min(2 ** i, 5)
) -> str:
    """
    Generates a summary for the given document content using an OpenAI LLM.
    Caches the summary and saves it to a text file.
    Falls back to first N chars if LLM call fails.
    Truncates summary if it exceeds truncate_char_length.
    """
    print("doc file path:", document_file_path)

    if summarization_model.company != "openai":
        logger.error(
            f"Summarization is currently only supported for OpenAI models. Requested: {summarization_model.company} for {document_file_path}. Falling back.")
        return document_content[:truncate_char_length]

    doc_content_hash = hashlib.md5(document_content.encode()).hexdigest()
    prompt_template_hash = hashlib.md5(summary_prompt_template.encode()).hexdigest()

    cache_key = get_document_summary_cache_key(
        document_file_path,
        doc_content_hash,
        summarization_model,
        prompt_template_hash
    )

    cached_summary = cache.get(cache_key)
    if cached_summary is not None:
        logger.debug(f"Cache hit for summary: {document_file_path}")
        return cast(str, cached_summary)

    logger.info(f"Cache miss for summary. Generating for: {document_file_path} using {summarization_model.model}")

    # Max tokens for LLM generation. For 150 chars, ~50-75 tokens. For 170 chars, maybe a bit more.
    # Let's be generous to allow the model to generate around the target and then we truncate.
    # Rule of thumb: desired_chars / 3 (avg chars per token) + buffer.
    # If prompt_target_char_length = 150, max_tokens = 150/3 + 25 = 75.
    # If truncate_char_length = 170, 170/3 + 25 = ~80 tokens. Let's use truncate_char_length for calculation.
    llm_max_output_tokens = (truncate_char_length // 2) + 30  # A bit more generous

    # The prompt template should include {document_content} and {target_char_length}
    final_prompt_content = summary_prompt_template.format(
        document_content=document_content,  # Pass the full document content
        target_char_length=prompt_target_char_length  # Inform LLM of desired character length
    )

    messages = [
        # The prompt template provided by user already includes System/User structure
        AIMessage(role="user", content=final_prompt_content)  # Assuming the template handles System/User roles
    ]
    # If the template is just the user part:
    # messages = [
    #     AIMessage(role="system", content="You are an expert legal document summarizer..."), # Example system
    #     AIMessage(role="user", content=final_prompt_content)
    # ]
    # Based on user's provided prompt, it seems to be a single user message that includes instructions.

    summary_text: str
    try:
        summary_text = await ai_call(
            model=summarization_model,
            messages=messages,
            max_tokens=llm_max_output_tokens,
            temperature=0.2,  # Factual summary
            num_ratelimit_retries=num_ratelimit_retries,
            backoff_algo=backoff_algo
        )
    except Exception as e:
        logger.warning(
            f"LLM summarization failed for {document_file_path}: {e}. Using fallback (first {truncate_char_length} chars).")
        summary_text = document_content[:truncate_char_length]

    if len(summary_text) > truncate_char_length:
        summary_text = summary_text[:truncate_char_length]
        logger.debug(f"Summary for {document_file_path} truncated to {truncate_char_length} characters.")

    cache.set(cache_key, summary_text)
    logger.info(f"Generated and cached summary for: {document_file_path}")

    try:
        path_parts = document_file_path.split('/')
        dataset_name = path_parts[0] if len(path_parts) > 1 else "unknown_dataset"
        summary_file_dir = os.path.join(summaries_output_dir_base, dataset_name)
        os.makedirs(summary_file_dir, exist_ok=True)
        original_filename = os.path.basename(document_file_path)
        summary_filename_txt = os.path.join(summary_file_dir, original_filename)
        with open(summary_filename_txt, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logger.debug(f"Summary for {document_file_path} also saved to {summary_filename_txt}")
    except Exception as e:
        logger.error(f"Failed to save summary text file for {document_file_path}: {e}")

    return summary_text
