import asyncio
import hashlib
import logging
import os
import torch
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any, Literal, cast, Dict, List

import anthropic
import cohere
import diskcache as dc
import httpx
import openai
import tiktoken
import voyageai  # type: ignore
import voyageai.error  # type: ignore
from anthropic import NOT_GIVEN, Anthropic, AsyncAnthropic, NotGiven
from anthropic.types import MessageParam
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, computed_field
from sentence_transformers import SentenceTransformer, CrossEncoder

from legalbenchrag.utils.credentials import credentials

logger = logging.getLogger("uvicorn")

# --- Globals ---
# Cache for loaded SentenceTransformer models to avoid reloading
local_model_cache: Dict[str, Any] = {}
# Cache for loaded CrossEncoder models
local_reranker_cache: Dict[str, Any] = {}


# AI Types
class AIModel(BaseModel):
    company: Literal["openai", "anthropic"]
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
cache = dc.Cache("./data/cache/ai_cache.db")

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
        self.anthropic_client = AsyncAnthropic(
            api_key=credentials.ai.anthropic_api_key.get_secret_value()
        )
        self.sync_anthropic_client = Anthropic(
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
    if isinstance(model, AIModel):
        if model.company == "anthropic":
            # Doesn't actually connect to the network
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
            # Use simple estimate. if needed precisely: use tokenizer from SentenceTransformer model if loaded
            return int(len(s) / 4)
    # Otherwise, estimate
    # Add estimate for local rerankers if needed, though token count isn't usually the limiting factor
    if isinstance(model, AIRerankModel) and model.company == 'huggingface':
        return int(len(s) / 4)

    logger.warning(f"Estimating Tokens for model type {type(model)}!")
    return int(len(s) / 4)


def get_call_cache_key(
    model: AIModel,
    messages: list[AIMessage],
) -> str:
    # Hash the array of texts
    md5_hasher = hashlib.md5()
    md5_hasher.update(model.model_dump_json().encode())
    for message in messages:
        md5_hasher.update(md5_hasher.hexdigest().encode())
        md5_hasher.update(message.model_dump_json().encode())
    key = md5_hasher.hexdigest()

    return key


async def ai_call(
    model: AIModel,
    messages: list[AIMessage],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    # When using anthropic, the first message must be from the user.
    # If the first message is not a User, this message will be prepended to the messages.
    anthropic_initial_message: str | None = "<START>",
    # If two messages of the same role are given to anthropic, they must be concatenated.
    # This is the delimiter between concatenated.
    anthropic_combine_delimiter: str = "\n",
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
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
        case "openai":
            for i in range(num_ratelimit_retries):
                try:
                    # Guard with ratelimit
                    async with get_ai_connection().openai_ratelimit_semaphore:
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = num_tokens_input / (tpm / 60)
                        await asyncio.sleep(expected_wait)

                    def ai_message_to_openai_message_param(
                        message: AIMessage,
                    ) -> ChatCompletionMessageParam:
                        if message.role == "system":  # noqa: SIM114
                            return {"role": message.role, "content": message.content}
                        elif message.role == "user":  # noqa: SIM114
                            return {"role": message.role, "content": message.content}
                        elif message.role == "assistant":
                            return {"role": message.role, "content": message.content}

                    if i > 0:
                        logger.debug("Trying again after RateLimitError...")
                    response = (
                        await get_ai_connection().openai_client.chat.completions.create(
                            model=model.model,
                            messages=[
                                ai_message_to_openai_message_param(message)
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

        case "anthropic":
            for i in range(num_ratelimit_retries):
                try:
                    # Guard with ratelimit
                    async with get_ai_connection().anthropic_ratelimit_semaphore:
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = num_tokens_input / (tpm / 60)
                        await asyncio.sleep(expected_wait)

                    def ai_message_to_anthropic_message_param(
                        message: AIMessage,
                    ) -> MessageParam:
                        if message.role == "user" or message.role == "assistant":
                            return {"role": message.role, "content": message.content}
                        elif message.role == "system":
                            raise AIValueError(
                                "system not allowed in anthropic message param"
                            )

                    if i > 0:
                        logger.debug("Trying again after RateLimitError...")

                    # Extract system message if it exists
                    system: str | NotGiven = NOT_GIVEN
                    if len(messages) > 0 and messages[0].role == "system":
                        system = messages[0].content
                        messages = messages[1:]
                    # Insert initial message if necessary
                    if (
                        anthropic_initial_message is not None
                        and len(messages) > 0
                        and messages[0].role != "user"
                    ):
                        messages = [
                            AIMessage(role="user", content=anthropic_initial_message)
                        ] + messages
                    # Combined messages (By combining consecutive messages of the same role)
                    combined_messages: list[AIMessage] = []
                    for message in messages:
                        if (
                            len(combined_messages) == 0
                            or combined_messages[-1].role != message.role
                        ):
                            combined_messages.append(message)
                        else:
                            # Copy before edit
                            combined_messages[-1] = combined_messages[-1].model_copy(
                                deep=True
                            )
                            # Merge consecutive messages with the same role
                            combined_messages[-1].content += (
                                anthropic_combine_delimiter + message.content
                            )
                    # Get the response
                    response_message = (
                        await get_ai_connection().anthropic_client.messages.create(
                            model=model.model,
                            system=system,
                            messages=[
                                ai_message_to_anthropic_message_param(message)
                                for message in combined_messages
                            ],
                            temperature=0.0,
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
    return return_value


def get_embeddings_cache_key(
    model: AIEmbeddingModel, text: str, embedding_type: AIEmbeddingType
) -> str:
    key = f"{model.company}||||{model.model}||||{embedding_type.name}||||{hashlib.md5(text.encode()).hexdigest()}"
    return key


# --- Internal Synchronous HuggingFace Embedding Function ---
def _encode_local_huggingface(
    model_name: str,
    texts: list[str],
    embedding_type: AIEmbeddingType,
    callback: Callable[[], None],
    trust_remote_code: bool = True
) -> list[list[float]]:
    """Loads and uses a SentenceTransformer model to encode texts synchronously."""
    if SentenceTransformer is None:
        raise ImportError("SentenceTransformer is not installed. Run `pip install sentence-transformers`.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model from cache or disk
    if model_name not in local_model_cache:
        logger.info(f"Loading local SentenceTransformer model: {model_name} onto device: {device}")
        try:
            # Pass the determined device to the constructor
            local_model_cache[model_name] = SentenceTransformer(
                model_name,
                trust_remote_code=trust_remote_code,
                device=device
            )
            logger.info(f"HuggingFace: Finished loading {model_name}")
        except Exception as e:
            # Catch potential loading errors (e.g., out of memory)
            logger.error(f"HuggingFace: Failed to load model {model_name} onto device {device}: {e}")
            raise RuntimeError(f"Failed to load SentenceTransformer model {model_name}") from e

    model: SentenceTransformer = local_model_cache[model_name]

    # Handle document/query type (e.g., BGE models expect specific prefixes)
    # This is a common pattern, but specific prefixes depend on the model.
    # We might need a more configuration-driven way if supporting many models.
    # Example for BGE:
    texts_to_encode = texts
    if "bge-" in model_name.lower():
         if embedding_type == AIEmbeddingType.QUERY:
             # BGE query prefix often mentioned in model cards (but check specific model)
             # Some require a space after ':', some don't. Let's assume no space needed.
             texts_to_encode = ["Represent this sentence for searching relevant passages: " + text for text in texts]
             # For older BGE versions, it might just be adding a trailing space to the query.
             # texts_to_encode = [text + " " for text in texts]
         elif embedding_type == AIEmbeddingType.DOCUMENT:
             # BGE document encoding usually doesn't require a prefix, but check model card.
             pass # No prefix needed for documents based on common BGE usage

    logger.debug(f"Encoding {len(texts_to_encode)} texts locally using {model_name}...")
    embeddings = model.encode(
        texts_to_encode,
        show_progress_bar=False, # Set to True for console progress within the thread
        batch_size=32 # Default batch size, adjust as needed
    )
    logger.debug(f"Finished local encoding with {model_name}.")

    # Convert numpy array to list of lists
    embeddings_list = cast(List[List[float]], embeddings.tolist())

    # Call the callback *after* the whole batch is done in this sync function
    # For finer-grained progress, SentenceTransformer's encode might need modification
    # or we process in smaller loops here.
    for _ in range(len(texts)): # Call callback once per input text for consistency with async path
        callback()

    return embeddings_list


# --- Unified Embedding Function ---
async def ai_embedding(
    model: AIEmbeddingModel,
    texts: list[str],
    embedding_type: AIEmbeddingType,
    *,
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
    # Callback (For tracking progress)
    callback: Callable[[], None] = lambda: None,
) -> list[list[float]]:
    if not texts:
        return []

    # Extract cache miss indices
    text_embeddings: list[list[float] | None] = [None] * len(texts)
    indices_to_fetch = []
    for i, text in enumerate(texts):
        cache_key = get_embeddings_cache_key(model, text, embedding_type)
        cached_embedding = cache.get(cache_key)
        if cached_embedding is not None:
            text_embeddings[i] = cached_embedding
            callback()  # Call callback even for cache hits
        else:
            indices_to_fetch.append(i)

    if not indices_to_fetch:
        return cast(list[list[float]], text_embeddings)  # All needed embeddings were in cache

    required_texts = [texts[i] for i in indices_to_fetch]
    if model.company != 'huggingface' and len(required_texts) > model.max_batch_len:
        # Recursive Batching for APIs
        tasks: list[Coroutine[Any, Any, list[list[float]]]] = []
        for i in range(0, len(indices_to_fetch), model.max_batch_len):
            batch_indices = indices_to_fetch[i : i + model.max_batch_len]
            batch_texts = [texts[idx] for idx in batch_indices]
            tasks.append(
                ai_embedding(
                    model,
                    batch_texts,
                    embedding_type,
                    num_ratelimit_retries=num_ratelimit_retries,
                    backoff_algo=backoff_algo,
                    # Pass callback down, it will be called for cache hits/misses in sub-calls
                    callback=callback,
                )
            )
        preflattened_results = await asyncio.gather(*tasks)
        results: list[list[float]] = []
        for embeddings_list in preflattened_results:
            results.extend(embeddings_list)
        # Merge with cache hits
        assert len(indices_to_fetch) == len(results), f"Batch result length mismatch: expected {len(indices_to_fetch)}, got {len(results)}"
        for i, embedding in zip(indices_to_fetch, results):
            text_embeddings[i] = embedding
            # Cache is handled within the recursive call, no need to set here
        assert all(text_embeddings[i] is not None for i in indices_to_fetch)
        return cast(list[list[float]], text_embeddings)

    # --- Get Embeddings for Cache Misses (Single Batch) ---
    embeddings_response: list[list[float]] | None = None
    num_tokens_input = sum(ai_num_tokens(model, text) for text in required_texts)

    if model.company == 'huggingface':
        # Run local encoding in a separate thread
        embeddings_response = await asyncio.to_thread(
            _encode_local_huggingface,
            model.model,
            required_texts,
            embedding_type,
            callback
        )# Note: The callback within _encode_local_huggingface already accounts for progress.

    elif model.company == "openai":
        for i in range(num_ratelimit_retries):
            try:
                async with get_ai_connection().openai_ratelimit_semaphore:
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                    expected_wait = max(60.0 / rpm if rpm != float('inf') else 0, num_tokens_input / (tpm / 60) if tpm != float('inf') else 0)
                    await asyncio.sleep(expected_wait)
                response = await get_ai_connection().openai_client.embeddings.create(
                    input=required_texts, model=model.model
                )
                embeddings_response = [embedding.embedding for embedding in response.data]
                for _ in range(len(required_texts)):
                    callback()  # Call callback after success
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
                    expected_wait = max(60.0 / rpm if rpm != float('inf') else 0, num_tokens_input / (tpm / 60) if tpm != float('inf') else 0)
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
                    expected_wait = max(60.0 / rpm if rpm != float('inf') else 0, num_tokens_input / (tpm / 60) if tpm != float('inf') else 0)
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

    # --- Process results and update cache ---
    if embeddings_response is None:
        # This should only happen if all retries failed for an API call
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
    model: AIRerankModel, query: str, texts: list[str], top_k: int | None
) -> str:
    # Hash the array of texts
    md5_hasher = hashlib.md5()
    md5_hasher.update(query.encode())
    for text in texts:
        md5_hasher.update(md5_hasher.hexdigest().encode())
        md5_hasher.update(text.encode())
    texts_hash = md5_hasher.hexdigest()
    key = f"{model.company}||||{model.model}||||{top_k}||||{texts_hash}"
    return key


# --- Internal Synchronous HuggingFace Reranking Function ---
def _rerank_local_huggingface(
    model_name: str,
    query: str,
    texts: list[str],
    top_k: int | None = None
) -> list[int]:
    """Loads and uses a CrossEncoder model to rerank texts synchronously."""
    if CrossEncoder is None:
        raise ImportError("CrossEncoder (part of sentence-transformers) is not installed. Run `pip install sentence-transformers`.")

    # Load model from cache or disk
    if model_name not in local_reranker_cache:
        logger.info(f"Loading local CrossEncoder model: {model_name}")
        # CrossEncoder loads from HF Hub, uses cache automatically
        # Specify trust_remote_code=True if needed for the specific model
        local_reranker_cache[model_name] = CrossEncoder(model_name, trust_remote_code=True)
        logger.info(f"Finished loading {model_name}")
    model: CrossEncoder = local_reranker_cache[model_name]

    # Prepare input pairs for the cross-encoder
    input_pairs = [(query, text) for text in texts]

    logger.debug(f"Reranking {len(input_pairs)} pairs locally using {model_name}...")
    # Predict scores
    scores = model.predict(input_pairs, show_progress_bar=False) # Turn off internal progress bar
    logger.debug(f"Finished local reranking with {model_name}.")

    # Combine scores with original indices
    indexed_scores = list(enumerate(scores)) # [(0, score0), (1, score1), ...]

    # Sort by score in descending order
    sorted_indices_scores = sorted(indexed_scores, key=lambda item: item[1], reverse=True)

    # Extract just the original indices in the new sorted order
    reranked_indices = [index for index, score in sorted_indices_scores]

    # Apply top_k if specified
    if top_k is not None:
        reranked_indices = reranked_indices[:top_k]

    return reranked_indices


# Gets the list of indices that reranks the original texts
async def ai_rerank(
    model: AIRerankModel,
    query: str,
    texts: list[str],
    *,
    top_k: int | None = None,
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
) -> list[int]:
    if not texts:
        return []

    # Ensure top_k is sensible relative to the number of texts
    if top_k is not None:
        top_k = min(top_k, len(texts))
        if top_k <= 0:  # If top_k becomes 0 or negative, return empty list
            return []

    cache_key = get_rerank_cache_key(model, query, texts, top_k)
    cached_reranking = cache.get(cache_key)
    if cached_reranking is not None:
        return cached_reranking

    indices: list[int] | None = None
    if model.company == "huggingface":
        # Run local reranking in a separate thread to avoid blocking asyncio loop
        try:
            indices = await asyncio.to_thread(
                _rerank_local_huggingface,
                model.model,
                query,
                texts,
                top_k
            )
        except Exception as e:
            logger.error(f"HuggingFace Rerank Error ({model.model}): {e}")
            # Handle specific errors if needed, otherwise re-raise or return empty
            # For simplicity, let's return empty list on error, but could raise AIError
            indices = []  # Or raise AIError(f"HuggingFace reranking failed: {e}")

    elif model.company == "cohere":
        for i in range(num_ratelimit_retries):
            try:
                async with get_ai_connection().cohere_ratelimit_semaphore:
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    await asyncio.sleep(60.0 / rpm if rpm != float('inf') else 0)
                response = await get_ai_connection().cohere_client.rerank(
                    model=model.model, query=query, documents=texts, top_n=top_k,
                )
                indices = [result.index for result in response.results]
                break
            except (cohere.errors.TooManyRequestsError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                logger.warning(f"Cohere Rerank Error: {e}")
                if i == num_ratelimit_retries - 1:
                    raise AITimeoutError("Cannot overcome Cohere Rerank Error")
                async with get_ai_connection().cohere_ratelimit_semaphore:
                    await asyncio.sleep(backoff_algo(i))
        if indices is None:
            raise AITimeoutError("Cannot overcome Cohere Rerank Error")

    elif model.company == "voyageai":
        for i in range(num_ratelimit_retries):
            try:
                async with get_ai_connection().voyageai_ratelimit_semaphore:
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    await asyncio.sleep(60.0 / rpm if rpm != float('inf') else 0)
                voyageai_response = await get_ai_connection().voyageai_client.rerank(
                     query=query, documents=texts, model=model.model, top_k=top_k,
                )
                indices = [int(result.index) for result in voyageai_response.results]
                break
            except voyageai.error.RateLimitError as e:
                logger.warning(f"VoyageAI Rerank Error: {e}")
                if i == num_ratelimit_retries - 1: raise AITimeoutError("Cannot overcome VoyageAI Rerank Error")
                async with get_ai_connection().voyageai_ratelimit_semaphore:
                    await asyncio.sleep(backoff_algo(i))
        if indices is None:
            raise AITimeoutError("Cannot overcome VoyageAI Rerank Error")

    if indices is None:
        # Should ideally not happen if all branches handle errors/success
        logger.error(f"Reranking failed unexpectedly for model {model.company}/{model.model}")
        indices = []  # Return empty list as a fallback

    cache.set(cache_key, indices)
    return indices
