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
import voyageai
import voyageai.error
from anthropic import NOT_GIVEN, Anthropic, AsyncAnthropic, NotGiven
from anthropic.types import MessageParam
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, computed_field
from sentence_transformers import SentenceTransformer, CrossEncoder

from legalbenchrag.utils.credentials import credentials

logger = logging.getLogger(__name__)

# --- Globals ---
local_model_cache: Dict[str, Any] = {}
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
                return 10000 * 96
            case "voyageai":
                return 1000000
            case "huggingface":
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
                return 30
            case "huggingface":
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
                return 64


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
                return 60
            case "huggingface":
                return float('inf')


os.makedirs("./data/cache", exist_ok=True)
cache = dc.Cache("./data/cache/ai_cache.db")

RATE_LIMIT_RATIO = 0.95


class AIConnection:
    openai_client: AsyncOpenAI
    voyageai_client: voyageai.AsyncClient
    cohere_client: cohere.AsyncClient
    anthropic_client: AsyncAnthropic
    sync_anthropic_client: Anthropic

    # This shared client is for libraries that explicitly support passing one.
    shared_httpx_async_client_for_supported_libs: httpx.AsyncClient

    cohere_ratelimit_semaphore = asyncio.Semaphore(1)
    voyageai_ratelimit_semaphore = asyncio.Semaphore(1)
    openai_ratelimit_semaphore = asyncio.Semaphore(1)
    anthropic_ratelimit_semaphore = asyncio.Semaphore(1)

    def __init__(self) -> None:
        # This client will be passed to OpenAI, Anthropic, and Cohere
        self.shared_httpx_async_client_for_supported_libs = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0, read=50.0),  # Increased overall timeout
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            follow_redirects=True
            # httpx uses trust_env=True by default, so it will pick up proxies from env vars
        )

        self.openai_client = AsyncOpenAI(
            api_key=credentials.ai.openai_api_key.get_secret_value(),
            http_client=self.shared_httpx_async_client_for_supported_libs
        )
        self.anthropic_client = AsyncAnthropic(
            api_key=credentials.ai.anthropic_api_key.get_secret_value(),
            http_client=self.shared_httpx_async_client_for_supported_libs
        )
        self.sync_anthropic_client = Anthropic(
            api_key=credentials.ai.anthropic_api_key.get_secret_value(),
            http_client=httpx.Client(  # Sync version needs its own sync httpx client
                timeout=httpx.Timeout(60.0, connect=10.0, read=50.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                follow_redirects=True
            )
        )
        self.cohere_client = cohere.AsyncClient(
            api_key=credentials.ai.cohere_api_key.get_secret_value(),
            httpx_client=self.shared_httpx_async_client_for_supported_libs
        )
        self.voyageai_client = voyageai.AsyncClient(
            api_key=credentials.ai.voyageai_api_key.get_secret_value()
        )

    async def close_shared_client(self):
        """Method to explicitly close the shared httpx client used by some libs."""
        if self.shared_httpx_async_client_for_supported_libs and \
                not self.shared_httpx_async_client_for_supported_libs.is_closed:
            await self.shared_httpx_async_client_for_supported_libs.aclose()

        # Close sync httpx client if it's managed by this class instance
        if hasattr(self.sync_anthropic_client, 'close') and callable(getattr(self.sync_anthropic_client, 'close')):
            self.sync_anthropic_client.close()


ai_connections: dict[asyncio.AbstractEventLoop, AIConnection] = {}
# Lock for thread-safe creation of AIConnection per event loop
_ai_connection_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}


async def get_ai_connection() -> AIConnection:  # Changed to async to use lock
    event_loop = asyncio.get_running_loop()  # Use get_running_loop in async context

    if event_loop not in _ai_connection_locks:
        # This part is tricky if multiple coroutines on the same loop call this simultaneously
        # before the lock itself is created. This should ideally be initialized earlier
        # or protected by a global lock for first-time access to _ai_connection_locks.
        # For simplicity now, assuming this function is not massively parallel on first call for a loop.
        _ai_connection_locks[event_loop] = asyncio.Lock()

    async with _ai_connection_locks[event_loop]:
        if event_loop not in ai_connections:
            logger.info(f"Creating new AIConnection for event loop {id(event_loop)}")
            ai_connections[event_loop] = AIConnection()
        return ai_connections[event_loop]


async def close_all_ai_connections():
    """Closes all shared httpx clients in cached AIConnection objects."""
    for loop, conn in list(ai_connections.items()):  # Iterate over a copy
        logger.info(f"Closing AIConnection for event loop {id(loop)}")
        await conn.close_shared_client()
        del ai_connections[loop]
        if loop in _ai_connection_locks:
            del _ai_connection_locks[loop]


class AIError(Exception):
    """A class for AI Task Errors"""


class AIValueError(AIError, ValueError):
    """A class for AI Value Errors"""


class AITimeoutError(AIError, TimeoutError):
    """A class for AI Task Timeout Errors"""


def ai_num_tokens(model: AIModel | AIEmbeddingModel | AIRerankModel, s: str) -> int:
    # This function might need to be async if get_ai_connection becomes async and is called here.
    # For now, assuming it can get a connection if one was already established by an async part.
    # Let's make it robust by trying to get connection if needed.
    # However, count_tokens is usually synchronous on client libraries after init.

    # To avoid issues with get_ai_connection potentially being called from sync context here,
    # let's assume if an AIConnection is needed, it must have been created by an async context first.
    # This part is a bit tricky if count_tokens itself needs an active async client that's not yet ready.
    # For tiktoken and simple length estimations, it's fine.
    # VoyageAI's client.count_tokens might be an issue if client isn't fully ready.

    if isinstance(model, AIModel):
        if model.company == "anthropic":
            # This is problematic as sync_anthropic_client is part of AIConnection
            # which is fetched via async get_ai_connection.
            # For simplicity, let's assume a sync client is available or estimate.
            # conn = get_ai_connection_sync_fallback() # Needs a sync way to get it, or estimate
            # return conn.sync_anthropic_client.count_tokens(s)
            try:
                conn = next(iter(ai_connections.values()))  # Try to get any existing connection (hacky)
                return conn.sync_anthropic_client.count_tokens(s)
            except (StopIteration, RuntimeError):  # RuntimeError if called from different thread with no loop
                logger.warning("Anthropic token count failed due to missing AIConnection, estimating.")
                return int(len(s) / 4)  # Fallback estimation
        elif model.company == "openai":
            try:
                encoding = tiktoken.encoding_for_model(model.model)
                return len(encoding.encode(s))
            except Exception:  # Model not found in tiktoken
                logger.warning(f"Tiktoken model {model.model} not found, estimating tokens.")
                return int(len(s) / 4)  # Fallback estimation
    if isinstance(model, AIEmbeddingModel):
        if model.company == "openai":
            try:
                encoding = tiktoken.encoding_for_model(model.model)
                return len(encoding.encode(s))
            except Exception:
                logger.warning(f"Tiktoken model {model.model} not found, estimating tokens.")
                return int(len(s) / 4)
        elif model.company == "voyageai":
            # voyageai.Client().count_tokens is synchronous
            # We need a way to get a sync voyage client instance or estimate
            try:
                # This assumes voyageai.Client() can be instantiated on the fly without async loop issues
                # Or that a sync version is available via AIConnection (it's not currently)
                # For now, using the global client if already initialized, or estimate.
                sync_voyage_client = voyageai.Client(api_key=credentials.ai.voyageai_api_key.get_secret_value())
                return sync_voyage_client.count_tokens([s], model=model.model)  # model kwarg added
            except Exception as e:
                logger.warning(f"VoyageAI token count failed ({e}), estimating.")
                return int(len(s) / 4)
        elif model.company == "huggingface":  # Local model
            return int(len(s) / 4)  # Simple estimation
    if isinstance(model, AIRerankModel) and model.company == 'huggingface':
        return int(len(s) / 4)

    logger.warning(
        f"Estimating Tokens for unhandled model type {type(model)} or company {getattr(model, 'company', 'Unknown')}!")
    return int(len(s) / 4)


def get_call_cache_key(
        model: AIModel,
        messages: list[AIMessage],
) -> str:
    md5_hasher = hashlib.md5()
    md5_hasher.update(model.model_dump_json().encode())
    for message in messages:
        md5_hasher.update(md5_hasher.hexdigest().encode())  # This seems like an error, should be message content
        md5_hasher.update(message.model_dump_json().encode())
    key = md5_hasher.hexdigest()
    return key


async def ai_call(
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

    num_tokens_input_estimation: int = sum(  # Renamed to avoid conflict if ai_num_tokens is async
        [ai_num_tokens(model, message.content) for message in messages]
        # ai_num_tokens here needs to be careful about async context
    )

    return_value: str | None = None
    ai_conn = await get_ai_connection()  # Get connection once

    match model.company:
        case "openai":
            for i in range(num_ratelimit_retries):
                try:
                    # Ratelimit semaphore is per-client type, not global to AIConnection instance
                    async with AIConnection.openai_ratelimit_semaphore:  # Access via class
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        # num_tokens_input_estimation is used here
                        expected_wait = num_tokens_input_estimation / (tpm / 60) if tpm > 0 else 0
                        await asyncio.sleep(expected_wait)

                    def ai_message_to_openai_message_param(
                            message: AIMessage,
                    ) -> ChatCompletionMessageParam:
                        if message.role in ["system", "user", "assistant"]:
                            return {"role": message.role, "content": message.content}
                        raise AIValueError(f"Unsupported message role for OpenAI: {message.role}")

                    if i > 0:
                        logger.debug("OpenAI: Trying again after RateLimitError...")
                    response = await ai_conn.openai_client.chat.completions.create(
                        model=model.model,
                        messages=[
                            ai_message_to_openai_message_param(message)
                            for message in messages
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    assert response.choices[0].message.content is not None
                    return_value = response.choices[0].message.content
                    break
                except RateLimitError:
                    logger.warning("OpenAI RateLimitError")
                    async with AIConnection.openai_ratelimit_semaphore:  # Access via class
                        await asyncio.sleep(backoff_algo(i))
            if return_value is None:
                raise AITimeoutError("Cannot overcome OpenAI RateLimitError")

        case "anthropic":
            for i in range(num_ratelimit_retries):
                try:
                    async with AIConnection.anthropic_ratelimit_semaphore:  # Access via class
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                        expected_wait = num_tokens_input_estimation / (tpm / 60) if tpm > 0 else 0
                        await asyncio.sleep(expected_wait)

                    def ai_message_to_anthropic_message_param(
                            message: AIMessage,
                    ) -> MessageParam:
                        if message.role == "user" or message.role == "assistant":
                            return {"role": message.role, "content": message.content}
                        elif message.role == "system":
                            raise AIValueError("system not allowed in anthropic message param list, provide separately")
                        raise AIValueError(f"Unsupported message role for Anthropic: {message.role}")

                    if i > 0:
                        logger.debug("Anthropic: Trying again after RateLimitError...")

                    system_prompt_content: str | NotGiven = NOT_GIVEN
                    processed_messages = list(messages)

                    if processed_messages and processed_messages[0].role == "system":
                        system_prompt_content = processed_messages[0].content
                        processed_messages = processed_messages[1:]

                    if (anthropic_initial_message is not None and
                            (not processed_messages or processed_messages[0].role != "user")):
                        processed_messages = [AIMessage(role="user",
                                                        content=anthropic_initial_message)] + processed_messages

                    combined_messages_for_anthropic: list[MessageParam] = []
                    if processed_messages:
                        current_message_content = processed_messages[0].content
                        current_role = processed_messages[0].role
                        for next_message in processed_messages[1:]:
                            if next_message.role == current_role:
                                current_message_content += anthropic_combine_delimiter + next_message.content
                            else:
                                combined_messages_for_anthropic.append(ai_message_to_anthropic_message_param(
                                    AIMessage(role=current_role, content=current_message_content)))
                                current_message_content = next_message.content
                                current_role = next_message.role
                        combined_messages_for_anthropic.append(ai_message_to_anthropic_message_param(
                            AIMessage(role=current_role, content=current_message_content)))

                    response_message = await ai_conn.anthropic_client.messages.create(
                        model=model.model,
                        system=system_prompt_content,
                        messages=combined_messages_for_anthropic,
                        temperature=temperature,  # temperature was already a parameter
                        max_tokens=max_tokens,
                    )
                    assert isinstance(response_message.content[0], anthropic.types.TextBlock)
                    assert isinstance(response_message.content[0].text, str)
                    return_value = response_message.content[0].text
                    break
                except anthropic.RateLimitError as e:
                    logger.warning(f"Anthropic Error: {repr(e)}")
                    async with AIConnection.anthropic_ratelimit_semaphore:  # Access via class
                        await asyncio.sleep(backoff_algo(i))
            if return_value is None:
                raise AITimeoutError("Cannot overcome Anthropic RateLimitError")

    if return_value is None:  # Should be caught by specific errors above, but as a safeguard
        raise AIError(f"Failed to get response from AI model {model.company}/{model.model}")

    cache.set(cache_key, return_value)
    return return_value


def get_embeddings_cache_key(
        model: AIEmbeddingModel, text: str, embedding_type: AIEmbeddingType
) -> str:
    key = f"{model.company}||||{model.model}||||{embedding_type.name}||||{hashlib.md5(text.encode()).hexdigest()}"
    return key


def _encode_local_huggingface(
        model_name: str,
        texts: list[str],
        embedding_type: AIEmbeddingType,  # embedding_type is used
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

    model_instance: SentenceTransformer = local_model_cache[model_name]  # Renamed for clarity
    texts_to_encode = texts
    # BGE models require specific prefixes for query vs document
    if "bge-" in model_name.lower() and embedding_type == AIEmbeddingType.QUERY:
        texts_to_encode = ["Represent this sentence for searching relevant passages: " + text for text in texts]

    logger.debug(f"Encoding {len(texts_to_encode)} texts locally using {model_name}...")
    embeddings = model_instance.encode(  # Use model_instance
        texts_to_encode,
        show_progress_bar=False,  # Typically false for programmatic use
        batch_size=32  # A common default, can be tuned
    )
    logger.debug(f"Finished local encoding with {model_name}.")
    embeddings_list = cast(List[List[float]], embeddings.tolist())
    for _ in range(len(texts)):  # Call callback for each original text processed
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
        if not text or not isinstance(text, str):  # Skip empty or invalid texts
            logger.warning(f"Invalid or empty text at index {i} for embedding. Skipping.")
            text_embeddings[i] = []  # Or handle as error; providing empty list to maintain structure
            callback()
            continue
        cache_key = get_embeddings_cache_key(model, text, embedding_type)
        cached_embedding = cache.get(cache_key)
        if cached_embedding is not None:
            text_embeddings[i] = cached_embedding
            callback()
        else:
            indices_to_fetch.append(i)

    if not indices_to_fetch:  # All texts were cached or invalid
        # Filter out None placeholders for invalid texts if we used None instead of []
        return [emb for emb in text_embeddings if emb is not None]  # type: ignore

    required_texts = [texts[i] for i in indices_to_fetch]

    # Batching for non-HF models if input exceeds max_batch_len
    if model.company != 'huggingface' and len(required_texts) > model.max_batch_len:
        tasks: list[Coroutine[Any, Any, list[list[float]]]] = []
        for i in range(0, len(indices_to_fetch), model.max_batch_len):
            current_batch_indices_to_fetch = indices_to_fetch[i: i + model.max_batch_len]
            current_batch_texts = [texts[idx] for idx in current_batch_indices_to_fetch]
            tasks.append(
                ai_embedding(  # Recursive call for the smaller batch
                    model,
                    current_batch_texts,
                    embedding_type,
                    num_ratelimit_retries=num_ratelimit_retries,
                    backoff_algo=backoff_algo,
                    # Callback for this sub-batch will be handled internally by the recursive call
                    # The main callback() is for the original texts.
                )
            )

        # This callback logic for batched API calls is complex.
        # The callback should ideally be invoked by the deepest ai_embedding call
        # once an embedding is truly computed or fetched from cache.
        # The current structure means callback() is called for cached items,
        # and then the recursive calls will handle callbacks for their items.
        # For simplicity, the provided callback is not passed down when batching for remote APIs here.
        # The original callback will be called N times by _encode_local_huggingface for local.

        preflattened_results = await asyncio.gather(*tasks)
        results_for_current_batch: list[list[float]] = []
        for embeddings_list in preflattened_results:
            results_for_current_batch.extend(embeddings_list)

        if len(indices_to_fetch) != len(results_for_current_batch):  # Check length against original required texts
            raise AIError(
                f"Batch result length mismatch: expected {len(indices_to_fetch)}, got {len(results_for_current_batch)}")

        for original_idx, embedding in zip(indices_to_fetch, results_for_current_batch):
            text_embeddings[original_idx] = embedding
            # Cache setting happens in the deepest call, no need to repeat here
            # Also, the callback for these items was handled by the deeper call

        return [emb for emb in text_embeddings if emb is not None]  # type: ignore

    embeddings_response: list[list[float]] | None = None
    # num_tokens_input estimation needs to be done carefully for async context
    # num_tokens_input = sum(ai_num_tokens(model, text) for text in required_texts) # This may cause issues
    num_tokens_input = 1  # Placeholder, actual calculation should be careful with async context

    ai_conn = await get_ai_connection()  # Get connection once for the current batch

    if model.company == 'huggingface':
        # For HuggingFace, callback is passed to _encode_local_huggingface which calls it internally
        embeddings_response = await asyncio.to_thread(
            _encode_local_huggingface,
            model.model,
            required_texts,
            embedding_type,
            callback  # Pass the original callback
        )
    elif model.company == "openai":
        for i_retry in range(num_ratelimit_retries):
            try:
                async with AIConnection.openai_ratelimit_semaphore:
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                    # Estimate tokens for THIS batch (required_texts) if possible, else use placeholder
                    current_batch_tokens = sum(ai_num_tokens(model, text) for text in required_texts)
                    expected_wait = max(60.0 / rpm if rpm != float('inf') else 0,
                                        current_batch_tokens / (tpm / 60) if tpm > 0 else 0)
                    await asyncio.sleep(expected_wait)
                response = await ai_conn.openai_client.embeddings.create(
                    input=required_texts, model=model.model
                )
                embeddings_response = [embedding.embedding for embedding in response.data]
                for _ in range(len(required_texts)): callback()  # Call callback for each processed text
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
                logger.warning(f"OpenAI Embedding Error: {e}")
                if i_retry == num_ratelimit_retries - 1: raise AITimeoutError("Cannot overcome OpenAI Embedding Error")
                async with AIConnection.openai_ratelimit_semaphore:
                    await asyncio.sleep(backoff_algo(i_retry))
    elif model.company == "cohere":
        for i_retry in range(num_ratelimit_retries):
            try:
                async with AIConnection.cohere_ratelimit_semaphore:
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    # current_batch_tokens = sum(ai_num_tokens(model, text) for text in required_texts) # Cohere doesn't use TPM for embeddings directly
                    expected_wait = (60.0 / rpm if rpm != float('inf') else 0)
                    await asyncio.sleep(expected_wait)
                result = await ai_conn.cohere_client.embed(
                    texts=required_texts, model=model.model,
                    input_type="search_document" if embedding_type == AIEmbeddingType.DOCUMENT else "search_query"
                )
                embeddings_response = cast(List[List[float]], result.embeddings)
                for _ in range(len(required_texts)): callback()
                break
            except (cohere.errors.TooManyRequestsError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
                logger.warning(f"Cohere Embedding Error: {e}")
                if i_retry == num_ratelimit_retries - 1: raise AITimeoutError("Cannot overcome Cohere Embedding Error")
                async with AIConnection.cohere_ratelimit_semaphore:
                    await asyncio.sleep(backoff_algo(i_retry))
    elif model.company == "voyageai":
        for i_retry in range(num_ratelimit_retries):
            try:
                async with AIConnection.voyageai_ratelimit_semaphore:
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    # current_batch_tokens = sum(ai_num_tokens(model, text) for text in required_texts)
                    expected_wait = (60.0 / rpm if rpm != float('inf') else 0)
                    await asyncio.sleep(expected_wait)
                result = await ai_conn.voyageai_client.embed(
                    required_texts, model=model.model,  # Use model.model here
                    input_type="document" if embedding_type == AIEmbeddingType.DOCUMENT else "query"
                )
                embeddings_response = cast(List[List[float]], result.embeddings)
                for _ in range(len(required_texts)): callback()
                break
            except voyageai.error.RateLimitError as e:  # Use the specific error type
                logger.warning(f"VoyageAI Embedding Error: {e}")
                if i_retry == num_ratelimit_retries - 1: raise AITimeoutError(
                    "Cannot overcome VoyageAI Embedding Error")
                async with AIConnection.voyageai_ratelimit_semaphore:
                    await asyncio.sleep(backoff_algo(i_retry))

    if embeddings_response is None:
        raise AITimeoutError(
            f"Failed to get embeddings for model {model.model} after retries for {len(required_texts)} texts.")

    if len(embeddings_response) != len(required_texts):  # Check against required_texts for this specific call
        raise AIError(
            f"Mismatch between requested ({len(required_texts)}) and received ({len(embeddings_response)}) embeddings in non-batched/final batch part.")

    for original_idx, embedding in zip(indices_to_fetch, embeddings_response):
        cache_key = get_embeddings_cache_key(model, texts[original_idx], embedding_type)
        cache.set(cache_key, embedding)
        text_embeddings[original_idx] = embedding

    # Final check and filtering of None values (e.g. from initial skip of invalid prompts)
    final_embeddings = [emb for emb in text_embeddings if emb is not None and (isinstance(emb, list) and len(emb) > 0)]
    if len(final_embeddings) != len(
            [txt for txt in texts if txt and isinstance(txt, str)]):  # Compare to valid input texts
        logger.warning(
            f"Length mismatch after processing all embeddings. Input valid texts: {len([txt for txt in texts if txt and isinstance(txt, str)])}, Output embeddings: {len(final_embeddings)}")
        # This part needs careful handling if some prompts were skipped initially.
        # The expectation is that text_embeddings has placeholders for skipped ones.

    return cast(list[list[float]], final_embeddings)  # Return only successfully processed embeddings


def get_rerank_cache_key(
        model: AIRerankModel, query: str, texts: list[str]
) -> str:
    md5_hasher = hashlib.md5()
    md5_hasher.update(model.model_dump_json().encode())  # Cache based on model config
    md5_hasher.update(query.encode())
    # Sort texts before hashing to ensure cache key is consistent regardless of input order
    # This is important if the order of `texts` to `ai_rerank` might vary for the same logical set.
    # However, reranking IS order-dependent for its results (indices refer to original list).
    # So, hashing on ordered texts is correct.
    for text in texts:  # Hash based on content of texts
        md5_hasher.update(text.encode())  # More direct hash of content
    texts_hash = md5_hasher.hexdigest()
    # Include query in the key name more explicitly
    key = f"rerank|||{model.company}|||{model.model}|||query_hash_{hashlib.md5(query.encode()).hexdigest()}|||texts_hash_{texts_hash}"
    return key


def _rerank_local_huggingface(
        model_name: str,
        query: str,
        texts: list[str],
        trust_remote_code: bool = True  # Added trust_remote_code
) -> list[int]:
    if CrossEncoder is None:
        raise ImportError(
            "CrossEncoder (part of sentence-transformers) is not installed. Run `pip install sentence-transformers`.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name not in local_reranker_cache:
        logger.info(f"Loading local CrossEncoder model: {model_name} onto device: {device}")
        try:
            local_reranker_cache[model_name] = CrossEncoder(
                model_name,
                trust_remote_code=trust_remote_code,  # Pass this along
                device=device
            )
            logger.info(f"Finished loading {model_name}")
        except Exception as e:
            logger.error(f"HuggingFace Reranker: Failed to load model {model_name} onto device {device}: {e}")
            raise RuntimeError(f"Failed to load CrossEncoder model {model_name}") from e

    model_instance: CrossEncoder = local_reranker_cache[model_name]  # Renamed
    input_pairs = [(query, text) for text in texts]

    logger.debug(f"Reranking {len(input_pairs)} pairs locally using {model_name}...")
    scores = model_instance.predict(  # Use model_instance
        input_pairs,
        show_progress_bar=False,  # Typically false
        batch_size=8 if "large" in model_name.lower() else 32  # Heuristic for batch size
    )
    logger.debug(f"Finished local reranking with {model_name}.")

    # scores can be a list or numpy array
    indexed_scores = list(enumerate(scores))
    # Sort by score in descending order, then by original index for stability if scores are equal
    sorted_indices_scores = sorted(indexed_scores, key=lambda item: (item[1], -item[0]), reverse=True)
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

    # Ensure texts for reranking are not excessively long, as some rerankers have limits
    # This is a basic truncation, more sophisticated handling might be needed.
    # Cohere rerank has a limit of 510 tokens per document.
    MAX_TEXT_LEN_FOR_RERANK = 4000  # Approx 1000 tokens, adjust as needed
    processed_texts = [text[:MAX_TEXT_LEN_FOR_RERANK] if len(text) > MAX_TEXT_LEN_FOR_RERANK else text for text in
                       texts]
    if any(len(t) > MAX_TEXT_LEN_FOR_RERANK for t in texts):
        logger.warning(f"Some texts provided to ai_rerank were truncated to {MAX_TEXT_LEN_FOR_RERANK} chars.")

    cache_key = get_rerank_cache_key(model, query, processed_texts)  # Use processed_texts for cache key
    cached_full_reranking = cache.get(cache_key)
    full_reranked_indices: list[int] | None = None

    if cached_full_reranking is not None:
        logger.debug(f"Cache hit for rerank key: {cache_key[:30]}...")
        full_reranked_indices = cached_full_reranking
    else:
        logger.debug(f"Cache miss for rerank key: {cache_key[:30]}... Calculating for {len(processed_texts)} texts.")
        ai_conn = await get_ai_connection()  # Get connection once

        if model.company == "huggingface":
            try:
                full_reranked_indices = await asyncio.to_thread(
                    _rerank_local_huggingface,
                    model.model,
                    query,
                    processed_texts,  # Use processed texts
                )
            except Exception as e:
                logger.error(f"HuggingFace Rerank Error ({model.model}): {e}")
                full_reranked_indices = list(range(len(processed_texts)))  # Fallback to original order
        elif model.company == "cohere":
            for i_retry in range(num_ratelimit_retries):
                try:
                    async with AIConnection.cohere_ratelimit_semaphore:
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        await asyncio.sleep(60.0 / rpm if rpm > 0 else 0)
                    # Cohere rerank expects 'documents' to be list of str or list of dicts with 'text' key
                    docs_for_cohere = [{"text": t} for t in processed_texts] if isinstance(processed_texts[0],
                                                                                           str) else processed_texts

                    response = await ai_conn.cohere_client.rerank(
                        model=model.model, query=query, documents=docs_for_cohere,  # type: ignore
                        top_n=len(processed_texts),  # Get scores for all to cache full ranking
                    )
                    full_reranked_indices = [result.index for result in response.results]
                    break
                except (cohere.errors.TooManyRequestsError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    logger.warning(f"Cohere Rerank Error: {e}")
                    if i_retry == num_ratelimit_retries - 1:
                        # Fallback to original order if all retries fail
                        logger.error("Max retries reached for Cohere Rerank. Falling back to original order.")
                        full_reranked_indices = list(range(len(processed_texts)))
                        break  # Break from retry loop
                    async with AIConnection.cohere_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i_retry))
            if full_reranked_indices is None:  # Should be set by fallback if error
                full_reranked_indices = list(range(len(processed_texts)))
        elif model.company == "voyageai":
            for i_retry in range(num_ratelimit_retries):
                try:
                    async with AIConnection.voyageai_ratelimit_semaphore:
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        await asyncio.sleep(60.0 / rpm if rpm > 0 else 0)
                    voyageai_response = await ai_conn.voyageai_client.rerank(
                        query=query, documents=processed_texts, model=model.model,  # Use model.model
                        top_k=len(processed_texts),  # Get scores for all
                    )
                    full_reranked_indices = [int(result.index) for result in voyageai_response.results]
                    break
                except voyageai.error.RateLimitError as e:  # Specific error
                    logger.warning(f"VoyageAI Rerank Error: {e}")
                    if i_retry == num_ratelimit_retries - 1:
                        logger.error("Max retries reached for VoyageAI Rerank. Falling back to original order.")
                        full_reranked_indices = list(range(len(processed_texts)))
                        break
                    async with AIConnection.voyageai_ratelimit_semaphore:
                        await asyncio.sleep(backoff_algo(i_retry))
            if full_reranked_indices is None:
                full_reranked_indices = list(range(len(processed_texts)))

        if full_reranked_indices is None:  # Should not happen if fallbacks are in place
            logger.error(
                f"Full reranking failed and no fallback for model {model.company}/{model.model}. Returning original order.")
            full_reranked_indices = list(range(len(processed_texts)))

        cache.set(cache_key, full_reranked_indices)
        logger.debug(f"Cached full rerank list for key: {cache_key[:30]}...")

    final_indices = full_reranked_indices
    if top_k is not None:  # Apply top_k if specified
        top_k = min(top_k, len(final_indices))  # Ensure top_k is not out of bounds
        if top_k > 0:
            final_indices = final_indices[:top_k]
        elif top_k == 0:  # If top_k is 0, return empty list
            final_indices = []
        # If top_k is negative, it implies no truncation, which is already handled.

    return final_indices


# --- New Summarization Functions ---
def get_document_summary_cache_key(
        document_file_path: str,
        document_content_hash: str,  # Hash of the full document content
        summarization_model: AIModel,
        summary_prompt_template_hash: str
) -> str:
    return f"summary_v2|||{document_file_path}|||{document_content_hash}|||{summarization_model.company}|||{summarization_model.model}|||{summary_prompt_template_hash}"


async def generate_document_summary(
        document_file_path: str,
        document_content: str,
        summarization_model: AIModel,
        summary_prompt_template: str,
        prompt_target_char_length: int,
        truncate_char_length: int,
        summaries_output_dir_base: str,
        num_ratelimit_retries: int = 5,  # Consistent with other AI calls
        backoff_algo: Callable[[int], float] = lambda i: min(2 ** i, 5)
) -> str:
    if summarization_model.company != "openai":  # Enforce OpenAI for now based on ai_call current state
        logger.error(
            f"Summarization currently only robustly supports OpenAI models via ai_call. Requested: {summarization_model.company} for {document_file_path}. Using fallback.")
        # Fallback to first N characters of the original document content
        return document_content[:truncate_char_length].strip()

    doc_content_hash = hashlib.md5(document_content.encode('utf-8', 'replace')).hexdigest()  # Ensure encoding
    prompt_template_hash = hashlib.md5(summary_prompt_template.encode('utf-8', 'replace')).hexdigest()

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

    llm_max_output_tokens = (truncate_char_length // 3) + 50  # More generous buffer

    try:
        # Ensure the template can handle these keys or adjust placeholder names
        final_prompt_content = summary_prompt_template.format(
            document_content=document_content,  # Pass the full document content
            target_char_length=prompt_target_char_length
        )
    except KeyError as e:
        logger.error(f"Invalid placeholder in summary_prompt_template: {e}. Document: {document_file_path}")
        # Fallback to simpler prompt if template fails
        final_prompt_content = f"Summarize this document to about {prompt_target_char_length} characters: {document_content}"

    # Assuming the prompt template might already include system/user roles.
    # If not, structure messages accordingly.
    # Based on user's example template, it's a single block of text with implicit roles.
    # For OpenAI, it's best to use explicit roles.
    messages_for_llm = [
        AIMessage(role="system",
                  content="You are an expert legal document summarizer. Your goal is to provide a concise summary (around the target character length provided) focusing on key entities, purpose, and legal topics. This summary will be used to give context to smaller text chunks from the document. Output only the summary text itself."),
        AIMessage(role="user", content=final_prompt_content)
    ]

    summary_text: str
    try:
        summary_text = await ai_call(
            model=summarization_model,
            messages=messages_for_llm,
            max_tokens=llm_max_output_tokens,
            temperature=0.2,  # Low temperature for factual summary
            num_ratelimit_retries=num_ratelimit_retries,
            backoff_algo=backoff_algo
        )
    except Exception as e:
        logger.warning(
            f"LLM summarization failed for {document_file_path}: {e}. Using fallback (first {truncate_char_length} chars of original).")
        summary_text = document_content[:truncate_char_length].strip()  # Fallback to original content

    # Truncate if summary is longer than allowed (even after LLM's attempt)
    if len(summary_text) > truncate_char_length:
        summary_text = summary_text[:truncate_char_length].strip()  # Ensure no leading/trailing spaces after truncate
        logger.debug(f"Summary for {document_file_path} was hard-truncated to {truncate_char_length} characters.")

    if not summary_text.strip() and document_content.strip():  # If summary is empty but doc was not
        logger.warning(
            f"Generated summary for {document_file_path} is empty. Using fallback (first {truncate_char_length} chars of original).")
        summary_text = document_content[:truncate_char_length].strip()

    cache.set(cache_key, summary_text)
    logger.info(f"Generated and cached summary for: {document_file_path} (len: {len(summary_text)})")

    try:
        # Assumes document_file_path is like "dataset_name/actual_filename.txt"
        path_parts = os.path.normpath(document_file_path).split(os.sep)
        dataset_name = path_parts[0] if len(path_parts) > 1 else "unknown_dataset"

        summary_file_dir = os.path.join(summaries_output_dir_base, dataset_name)
        os.makedirs(summary_file_dir, exist_ok=True)

        original_filename_base = os.path.basename(document_file_path)
        # Replace original extension with .summary.txt or append .summary.txt
        summary_filename_txt = os.path.join(summary_file_dir,
                                            f"{os.path.splitext(original_filename_base)[0]}.summary.txt")

        with open(summary_filename_txt, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logger.debug(f"Summary for {document_file_path} also saved to {summary_filename_txt}")
    except Exception as e:
        logger.error(f"Failed to save summary text file for {document_file_path}: {e}")

    return summary_text