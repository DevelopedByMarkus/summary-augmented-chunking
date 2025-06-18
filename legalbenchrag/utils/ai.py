import asyncio
import hashlib
import logging
import os
import torch
import random
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any, Literal, cast, Dict, List
import re

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


# Define characters typically illegal in Windows filenames and replacement
ILLEGAL_FILENAME_CHARS = r'[\\/:*?"<>|]'
REPLACEMENT_CHAR = '_'


def sanitize_filename(filename: str) -> str:
    """Replaces characters illegal in Windows filenames with underscores."""
    return re.sub(ILLEGAL_FILENAME_CHARS, REPLACEMENT_CHAR, filename)


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

    shared_httpx_async_client_for_supported_libs: httpx.AsyncClient
    _sync_anthropic_http_client: httpx.Client

    # Instance-level semaphores
    openai_semaphore_instance: asyncio.Semaphore
    cohere_semaphore_instance: asyncio.Semaphore
    voyageai_semaphore_instance: asyncio.Semaphore
    anthropic_semaphore_instance: asyncio.Semaphore

    def __init__(self) -> None:
        self.shared_httpx_async_client_for_supported_libs = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0, read=50.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            follow_redirects=True
        )
        self._sync_anthropic_http_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0, read=50.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            follow_redirects=True
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
            http_client=self._sync_anthropic_http_client
        )
        self.cohere_client = cohere.AsyncClient(
            api_key=credentials.ai.cohere_api_key.get_secret_value(),
            httpx_client=self.shared_httpx_async_client_for_supported_libs
        )
        self.voyageai_client = voyageai.AsyncClient(
            api_key=credentials.ai.voyageai_api_key.get_secret_value()
        )

        # These semaphores are now instance variables, created when AIConnection is instantiated.
        # The get_ai_connection function ensures this happens within an active event loop.
        self.openai_semaphore_instance = asyncio.Semaphore(1)
        self.cohere_semaphore_instance = asyncio.Semaphore(1)
        self.voyageai_semaphore_instance = asyncio.Semaphore(1)
        self.anthropic_semaphore_instance = asyncio.Semaphore(1)

    async def close_shared_client(self):
        """Method to explicitly close the shared httpx client used by some libs."""
        if self.shared_httpx_async_client_for_supported_libs and \
                not self.shared_httpx_async_client_for_supported_libs.is_closed:
            logger.debug("Closing shared async httpx client.")
            await self.shared_httpx_async_client_for_supported_libs.aclose()

        if self._sync_anthropic_http_client and \
                not self._sync_anthropic_http_client.is_closed:
            logger.debug("Closing sync httpx client for Anthropic.")
            self._sync_anthropic_http_client.close()


ai_connections: dict[asyncio.AbstractEventLoop, AIConnection] = {}
# Lock for thread-safe creation of AIConnection per event loop
_ai_connection_locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}


async def get_ai_connection() -> AIConnection:
    event_loop = asyncio.get_running_loop()
    if event_loop not in _ai_connection_locks:
        # This lock creation itself is not thread-safe if multiple threads hit this for the first time for the same new loop
        # However, in typical asyncio usage with a single main thread managing the loop, this is okay.
        _ai_connection_locks[event_loop] = asyncio.Lock()

    async with _ai_connection_locks[event_loop]:
        if event_loop not in ai_connections:
            logger.info(f"Creating new AIConnection for event loop {id(event_loop)}")
            ai_connections[event_loop] = AIConnection()
        return ai_connections[event_loop]


async def close_all_ai_connections():
    """Closes all shared httpx clients in cached AIConnection objects."""
    for loop_key in list(ai_connections.keys()):  # Iterate over copy of keys
        conn = ai_connections.pop(loop_key, None)
        if conn:
            logger.info(f"Closing AIConnection for event loop {id(loop_key)}")
            await conn.close_shared_client()
        if loop_key in _ai_connection_locks:
            # The lock itself doesn't need explicit closing, it's just a sync primitive
            del _ai_connection_locks[loop_key]


class AIError(Exception):
    """A class for AI Task Errors"""


class AIValueError(AIError, ValueError):
    """A class for AI Value Errors"""


class AITimeoutError(AIError, TimeoutError):
    """A class for AI Task Timeout Errors"""


def ai_num_tokens(model: AIModel | AIEmbeddingModel | AIRerankModel, s: str) -> int:
    if isinstance(model, AIModel):
        if model.company == "anthropic":
            try:
                # Try to get an existing sync client if AIConnection was initialized
                # This is still a bit of a hack for a function that might be called synchronously
                # Ideally, if this function needs a client, the client should be passed or be available synchronously.
                if ai_connections:
                    conn = next(iter(ai_connections.values()))
                    return conn.sync_anthropic_client.count_tokens(s)
                else:  # Fallback if no AIConnection instance exists yet
                    client = Anthropic(api_key=credentials.ai.anthropic_api_key.get_secret_value())
                    return client.count_tokens(s)
            except Exception as e:
                logger.warning(f"Anthropic token count failed ({e}), estimating.")
                return int(len(s) / 4)
        elif model.company == "openai":
            try:
                encoding = tiktoken.encoding_for_model(model.model)
                return len(encoding.encode(s))
            except Exception:
                logger.warning(f"Tiktoken model {model.model} not found, estimating tokens.")
                return int(len(s) / 4)
    if isinstance(model, AIEmbeddingModel):
        if model.company == "openai":
            try:
                encoding = tiktoken.encoding_for_model(model.model)
                return len(encoding.encode(s))
            except Exception:
                logger.warning(f"Tiktoken model {model.model} not found, estimating tokens.")
                return int(len(s) / 4)
        elif model.company == "voyageai":
            try:
                # VoyageAI's sync client for count_tokens
                sync_voyage_client = voyageai.Client(api_key=credentials.ai.voyageai_api_key.get_secret_value())
                return sync_voyage_client.count_tokens([s], model=model.model)
            except Exception as e:
                logger.warning(f"VoyageAI token count failed ({e}), estimating.")
                return int(len(s) / 4)
        elif model.company == "huggingface":
            return int(len(s) / 4)
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
        # Corrected hashing: hash the message content or its dump, not the running hash
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

    num_tokens_input_estimation: int = sum(
        [ai_num_tokens(model, message.content) for message in messages]
    )

    return_value: str | None = None
    ai_conn = await get_ai_connection()

    match model.company:
        case "openai":
            for i in range(num_ratelimit_retries):
                try:
                    async with ai_conn.openai_semaphore_instance:  # Use instance semaphore
                        tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
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
                    content = response.choices[0].message.content
                    if content is None:
                        raise AIError(f"OpenAI response content is None for model {model.model}")
                    return_value = content
                    break
                except RateLimitError:
                    logger.warning("OpenAI RateLimitError")
                    async with ai_conn.openai_semaphore_instance:  # Use instance semaphore
                        await asyncio.sleep(backoff_algo(i))
            if return_value is None:
                raise AITimeoutError("Cannot overcome OpenAI RateLimitError")

        case "anthropic":
            for i in range(num_ratelimit_retries):
                try:
                    async with ai_conn.anthropic_semaphore_instance:  # Use instance semaphore
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
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if not response_message.content or not isinstance(response_message.content[0],
                                                                      anthropic.types.TextBlock) or not isinstance(
                            response_message.content[0].text, str):
                        raise AIError(f"Anthropic response content is invalid for model {model.model}")
                    return_value = response_message.content[0].text
                    break
                except anthropic.RateLimitError as e:
                    logger.warning(f"Anthropic Error: {repr(e)}")
                    async with ai_conn.anthropic_semaphore_instance:  # Use instance semaphore
                        await asyncio.sleep(backoff_algo(i))
            if return_value is None:
                raise AITimeoutError("Cannot overcome Anthropic RateLimitError")

    if return_value is None:
        raise AIError(f"Failed to get response from AI model {model.company}/{model.model}")

    cache.set(cache_key, return_value)
    return return_value


def get_embeddings_cache_key(
        model: AIEmbeddingModel, text: str, embedding_type: AIEmbeddingType
) -> str:
    key = f"{model.company}||||{model.model}||||{embedding_type.name}||||{hashlib.md5(text.encode('utf-8', 'replace')).hexdigest()}"
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

    model_instance: SentenceTransformer = local_model_cache[model_name]
    texts_to_encode = texts
    if "bge-" in model_name.lower() and embedding_type == AIEmbeddingType.QUERY:
        texts_to_encode = ["Represent this sentence for searching relevant passages: " + text for text in texts]

    logger.debug(f"Encoding {len(texts_to_encode)} texts locally using {model_name}...")
    embeddings = model_instance.encode(
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
        backoff_algo: Callable[[int], float] = lambda i: min(2 ** i, 60) + random.uniform(0, 1),
        callback: Callable[[], None] = lambda: None,
) -> list[list[float]]:
    if not texts:
        return []

    valid_texts_with_original_indices: List[tuple[int, str]] = []
    for i, text in enumerate(texts):
        if text and isinstance(text, str):
            valid_texts_with_original_indices.append((i, text))
        else:
            logger.warning(f"Invalid or empty text at original index {i} for embedding. Skipping.")
            # No placeholder needed in text_embeddings if we reconstruct later

    if not valid_texts_with_original_indices:
        return []  # No valid texts to process

    # Prepare for fetching: map valid text index to its original index
    text_embeddings_map: Dict[int, list[float] | None] = {orig_idx: None for orig_idx, _ in
                                                          valid_texts_with_original_indices}
    indices_to_fetch_map: Dict[int, int] = {}  # Maps new_idx (for required_texts) to original_idx
    required_texts_list: List[str] = []

    current_fetch_idx = 0
    for original_idx, valid_text in valid_texts_with_original_indices:
        cache_key = get_embeddings_cache_key(model, valid_text, embedding_type)
        cached_embedding = cache.get(cache_key)
        if cached_embedding is not None:
            text_embeddings_map[original_idx] = cached_embedding
            callback()
        else:
            indices_to_fetch_map[current_fetch_idx] = original_idx
            required_texts_list.append(valid_text)
            current_fetch_idx += 1

    if not required_texts_list:  # All valid texts were cached
        final_result_from_cache = [None] * len(texts)
        for orig_idx, emb in text_embeddings_map.items():
            if emb is not None:
                final_result_from_cache[orig_idx] = emb
        return [e for e in final_result_from_cache if
                e is not None and (isinstance(e, list) and len(e) > 0)]  # type: ignore

    # Batching for non-HF models if input exceeds max_batch_len
    if model.company != 'huggingface' and len(required_texts_list) > model.max_batch_len:
        tasks: list[Coroutine[Any, Any, list[list[float]]]] = []
        # Create batches from required_texts_list and their corresponding original indices
        # This part needs to map results back to the original text_embeddings_map structure

        # Store results from sub-batches mapped by their index within required_texts_list
        temp_batch_results: Dict[int, list[float]] = {}

        async def process_sub_batch(sub_batch_texts: List[str], sub_batch_orig_indices_in_req_texts: List[int]):
            sub_embeddings = await ai_embedding(  # Recursive call
                model, sub_batch_texts, embedding_type,
                num_ratelimit_retries=num_ratelimit_retries, backoff_algo=backoff_algo
                # Callback is handled by the deepest call for non-cached items
            )
            for i, emb in enumerate(sub_embeddings):
                # map back to index in required_texts_list, not original overall index yet
                temp_batch_results[sub_batch_orig_indices_in_req_texts[i]] = emb

        sub_batch_tasks = []
        for i in range(0, len(required_texts_list), model.max_batch_len):
            batch_texts_for_api = required_texts_list[i: i + model.max_batch_len]
            # These are indices *within* required_texts_list for this sub-batch
            batch_indices_in_required_texts_list = list(range(i, i + len(batch_texts_for_api)))
            sub_batch_tasks.append(process_sub_batch(batch_texts_for_api, batch_indices_in_required_texts_list))

        await asyncio.gather(*sub_batch_tasks)

        # Populate text_embeddings_map using temp_batch_results and indices_to_fetch_map
        for req_text_idx, embedding in temp_batch_results.items():
            original_idx = indices_to_fetch_map[req_text_idx]
            text_embeddings_map[original_idx] = embedding
            cache_key = get_embeddings_cache_key(model, texts[original_idx], embedding_type)
            cache.set(cache_key, embedding)
            callback()  # Call callback now that it's processed and cached

        # Reconstruct final list based on original `texts` length
        final_result = [None] * len(texts)
        for i, _ in enumerate(texts):
            final_result[i] = text_embeddings_map.get(i)
        return [e for e in final_result if e is not None and (isinstance(e, list) and len(e) > 0)]  # type: ignore

    # ---- Process a single batch (or the only batch if not exceeding max_batch_len) ----
    embeddings_response_current_batch: list[list[float]] | None = None
    ai_conn = await get_ai_connection()

    if model.company == 'huggingface':
        embeddings_response_current_batch = await asyncio.to_thread(
            _encode_local_huggingface, model.model, required_texts_list, embedding_type, callback
        )
    elif model.company == "openai":
        for i_retry in range(num_ratelimit_retries):
            try:
                async with ai_conn.openai_semaphore_instance:  # Use instance semaphore
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    tpm = model.ratelimit_tpm * RATE_LIMIT_RATIO
                    current_batch_tokens = sum(ai_num_tokens(model, text) for text in required_texts_list)
                    expected_wait = max(60.0 / rpm if rpm != float('inf') else 0,
                                        current_batch_tokens / (tpm / 60) if tpm > 0 else 0)
                    await asyncio.sleep(expected_wait)
                response = await ai_conn.openai_client.embeddings.create(
                    input=required_texts_list, model=model.model
                )
                if response and response.data:
                    embeddings_response_current_batch = [e.embedding for e in response.data]
                    for _ in range(len(required_texts_list)):
                        callback()
                    break
            except (RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
                logger.warning(f"OpenAI Embedding Error: {e}")
                if i_retry == num_ratelimit_retries - 1: raise AITimeoutError("Cannot overcome OpenAI Embedding Error")
                async with ai_conn.openai_semaphore_instance:
                    await asyncio.sleep(backoff_algo(i_retry))
    elif model.company == "cohere":
        for i_retry in range(num_ratelimit_retries):
            try:
                async with ai_conn.cohere_semaphore_instance:  # Use instance semaphore
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    expected_wait = (60.0 / rpm if rpm != float('inf') else 0)
                    await asyncio.sleep(expected_wait)
                result = await ai_conn.cohere_client.embed(
                    texts=required_texts_list, model=model.model,
                    input_type="search_document" if embedding_type == AIEmbeddingType.DOCUMENT else "search_query"
                )
                embeddings_response_current_batch = cast(List[List[float]], result.embeddings)
                for _ in range(len(required_texts_list)): callback()
                break
            except (cohere.errors.TooManyRequestsError, httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
                logger.warning(f"Cohere Embedding Error: {e}")
                if i_retry == num_ratelimit_retries - 1: raise AITimeoutError("Cannot overcome Cohere Embedding Error")
                async with ai_conn.cohere_semaphore_instance:
                    await asyncio.sleep(backoff_algo(i_retry))
    elif model.company == "voyageai":
        for i_retry in range(num_ratelimit_retries):
            try:
                async with ai_conn.voyageai_semaphore_instance:  # Use instance semaphore
                    rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                    expected_wait = (60.0 / rpm if rpm != float('inf') else 0)
                    await asyncio.sleep(expected_wait)
                result = await ai_conn.voyageai_client.embed(
                    required_texts_list, model=model.model,
                    input_type="document" if embedding_type == AIEmbeddingType.DOCUMENT else "query"
                )
                embeddings_response_current_batch = cast(List[List[float]], result.embeddings)
                for _ in range(len(required_texts_list)): callback()
                break
            except voyageai.error.RateLimitError as e:
                logger.warning(f"VoyageAI Embedding Error: {e}")
                if i_retry == num_ratelimit_retries - 1: raise AITimeoutError(
                    "Cannot overcome VoyageAI Embedding Error")
                async with ai_conn.voyageai_semaphore_instance:
                    await asyncio.sleep(backoff_algo(i_retry))

    if embeddings_response_current_batch is None:
        raise AITimeoutError(f"Failed to get embeddings for model {model.model} (batch part).")

    if len(embeddings_response_current_batch) != len(required_texts_list):
        raise AIError(
            f"Mismatch in embeddings for required_texts_list: got {len(embeddings_response_current_batch)}, expected {len(required_texts_list)}")

    for i, embedding in enumerate(embeddings_response_current_batch):
        original_idx = indices_to_fetch_map[i]  # Map back to original index in `texts`
        text_embeddings_map[original_idx] = embedding
        cache_key = get_embeddings_cache_key(model, texts[original_idx], embedding_type)
        cache.set(cache_key, embedding)
        # Callback was already called for HF, or for each text in API success block.

    # Reconstruct final list based on original `texts` length
    final_result = [None] * len(texts)
    for i, _ in enumerate(texts):
        final_result[i] = text_embeddings_map.get(i)  # Get from map which includes cached and newly fetched

    return [e for e in final_result if e is not None and (isinstance(e, list) and len(e) > 0)]  # type: ignore


def get_rerank_cache_key(
        model: AIRerankModel, query: str, texts: list[str]
) -> str:
    md5_hasher = hashlib.md5()
    md5_hasher.update(model.model_dump_json().encode())
    md5_hasher.update(query.encode('utf-8', 'replace'))
    for text in texts:
        md5_hasher.update(text.encode('utf-8', 'replace'))
    texts_hash = md5_hasher.hexdigest()
    key = f"rerank_v2|||{model.company}|||{model.model}|||q_hash_{hashlib.md5(query.encode('utf-8', 'replace')).hexdigest()}|||texts_hash_{texts_hash}"
    return key


def _rerank_local_huggingface(
        model_name: str,
        query: str,
        texts: list[str],
        trust_remote_code: bool = True
) -> list[int]:
    if CrossEncoder is None:
        raise ImportError("CrossEncoder not installed.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name not in local_reranker_cache:
        logger.info(f"Loading local CrossEncoder model: {model_name} onto device: {device}")
        try:
            local_reranker_cache[model_name] = CrossEncoder(
                model_name, trust_remote_code=trust_remote_code, device=device
            )
            logger.info(f"Finished loading {model_name}")
        except Exception as e:
            logger.error(f"HF Reranker: Load failed for {model_name}: {e}")
            raise RuntimeError(f"Failed to load CrossEncoder {model_name}") from e

    model_instance: CrossEncoder = local_reranker_cache[model_name]
    input_pairs = [(query, text) for text in texts]

    logger.debug(f"Reranking {len(input_pairs)} pairs locally with {model_name}...")
    try:
        scores = model_instance.predict(
            input_pairs, show_progress_bar=False,
            batch_size=8 if "large" in model_name.lower() else 32
        )
    except Exception as e:
        logger.error(f"Error during CrossEncoder predict for {model_name}: {e}")
        return list(range(len(texts)))  # Fallback to original order on error

    logger.debug(f"Finished local reranking with {model_name}.")

    indexed_scores = list(enumerate(scores))
    sorted_indices_scores = sorted(indexed_scores, key=lambda item: (item[1], -item[0]), reverse=True)
    return [index for index, score in sorted_indices_scores]


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

    MAX_TEXT_LEN_FOR_RERANK = 4000
    processed_texts = [text[:MAX_TEXT_LEN_FOR_RERANK] if len(text) > MAX_TEXT_LEN_FOR_RERANK else text for text in
                       texts]
    if any(len(t) > MAX_TEXT_LEN_FOR_RERANK for t in texts):
        logger.warning(f"Some texts for ai_rerank truncated to {MAX_TEXT_LEN_FOR_RERANK} chars.")

    cache_key = get_rerank_cache_key(model, query, processed_texts)
    cached_full_reranking = cache.get(cache_key)
    full_reranked_indices: list[int] | None = None

    if cached_full_reranking is not None:
        logger.debug(f"Cache hit for rerank key: {cache_key[:30]}...")
        full_reranked_indices = cached_full_reranking
    else:
        logger.debug(f"Cache miss for rerank key: {cache_key[:30]}... Calculating for {len(processed_texts)} texts.")
        ai_conn = await get_ai_connection()

        if model.company == "huggingface":
            try:
                full_reranked_indices = await asyncio.to_thread(
                    _rerank_local_huggingface, model.model, query, processed_texts
                )
            except Exception as e:
                logger.error(f"HuggingFace Rerank Error ({model.model}): {e}")
                full_reranked_indices = list(range(len(processed_texts)))
        elif model.company == "cohere":
            for i_retry in range(num_ratelimit_retries):
                try:
                    async with ai_conn.cohere_semaphore_instance:  # Use instance semaphore
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        await asyncio.sleep(60.0 / rpm if rpm > 0 else 0)

                    docs_for_cohere = [{"text": t} for t in processed_texts]  # No more isinstance check needed
                    response = await ai_conn.cohere_client.rerank(
                        model=model.model, query=query, documents=docs_for_cohere,  # type: ignore
                        top_n=len(processed_texts),
                    )
                    full_reranked_indices = [result.index for result in response.results]
                    break
                except (cohere.errors.TooManyRequestsError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    logger.warning(f"Cohere Rerank Error: {e}")
                    if i_retry == num_ratelimit_retries - 1:
                        logger.error("Max retries for Cohere Rerank. Fallback to original order.")
                        full_reranked_indices = list(range(len(processed_texts)))
                        break
                    async with ai_conn.cohere_semaphore_instance:
                        await asyncio.sleep(backoff_algo(i_retry))
            if full_reranked_indices is None: full_reranked_indices = list(range(len(processed_texts)))
        elif model.company == "voyageai":
            for i_retry in range(num_ratelimit_retries):
                try:
                    async with ai_conn.voyageai_semaphore_instance:  # Use instance semaphore
                        rpm = model.ratelimit_rpm * RATE_LIMIT_RATIO
                        await asyncio.sleep(60.0 / rpm if rpm > 0 else 0)
                    voyageai_response = await ai_conn.voyageai_client.rerank(
                        query=query, documents=processed_texts, model=model.model,
                        top_k=len(processed_texts),
                    )
                    full_reranked_indices = [int(result.index) for result in voyageai_response.results]
                    break
                except voyageai.error.RateLimitError as e:
                    logger.warning(f"VoyageAI Rerank Error: {e}")
                    if i_retry == num_ratelimit_retries - 1:
                        logger.error("Max retries for VoyageAI Rerank. Fallback to original order.")
                        full_reranked_indices = list(range(len(processed_texts)))
                        break
                    async with ai_conn.voyageai_semaphore_instance:
                        await asyncio.sleep(backoff_algo(i_retry))
            if full_reranked_indices is None: full_reranked_indices = list(range(len(processed_texts)))

        if full_reranked_indices is None:
            logger.error(f"Reranking failed. Fallback to original order for {model.company}/{model.model}.")
            full_reranked_indices = list(range(len(processed_texts)))

        cache.set(cache_key, full_reranked_indices)
        logger.debug(f"Cached full rerank list for key: {cache_key[:30]}...")

    final_indices = full_reranked_indices
    if top_k is not None:
        top_k = min(top_k, len(final_indices))
        if top_k >= 0:  # Allow top_k=0 to return empty list
            final_indices = final_indices[:top_k]
        # If top_k is negative, implies no truncation from the full list.
    return final_indices


# --- New Summarization Functions ---
def get_document_summary_cache_key(
        document_file_path: str,
        document_content_hash: str,
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
        num_ratelimit_retries: int = 5,
        backoff_algo: Callable[[int], float] = lambda i: min(2 ** i, 5)
) -> str:
    if summarization_model.company != "openai":
        logger.error(f"Summarization only supports OpenAI. Requested: {summarization_model.company}. Fallback.")
        return document_content[:truncate_char_length].strip()

    doc_content_hash = hashlib.md5(document_content.encode('utf-8', 'replace')).hexdigest()
    prompt_template_hash = hashlib.md5(summary_prompt_template.encode('utf-8', 'replace')).hexdigest()

    cache_key = get_document_summary_cache_key(
        document_file_path, doc_content_hash, summarization_model, prompt_template_hash
    )

    cached_summary = cache.get(cache_key)
    if cached_summary is not None:
        logger.debug(f"Cache hit for summary: {document_file_path}")
        return cast(str, cached_summary)

    logger.info(f"Cache miss for summary. Generating for: {document_file_path} with {summarization_model.model}")

    llm_max_output_tokens = (truncate_char_length // 3) + 50

    try:
        final_prompt_content = summary_prompt_template.format(
            document_content=document_content, target_char_length=prompt_target_char_length
        )
    except KeyError as e:
        logger.error(
            f"Invalid placeholder in summary_prompt_template: {e} for {document_file_path}. Using basic prompt.")
        final_prompt_content = f"Summarize this to about {prompt_target_char_length} chars: {document_content}"

    messages_for_llm = [
        AIMessage(role="system",
                  content="You are an expert legal document summarizer... Output only the summary text."),  # Shortened
        AIMessage(role="user", content=final_prompt_content)
    ]

    summary_text: str
    try:
        summary_text = await ai_call(
            model=summarization_model, messages=messages_for_llm, max_tokens=llm_max_output_tokens,
            temperature=0.2, num_ratelimit_retries=num_ratelimit_retries, backoff_algo=backoff_algo
        )
    except Exception as e:
        logger.warning(f"LLM summarization failed for {document_file_path}: {e}. Fallback.")
        summary_text = document_content[:truncate_char_length].strip()

    if len(summary_text) > truncate_char_length:
        summary_text = summary_text[:truncate_char_length].strip()
        logger.debug(f"Summary for {document_file_path} hard-truncated to {truncate_char_length} chars.")

    if not summary_text.strip() and document_content.strip():
        logger.warning(f"Empty summary for {document_file_path}. Fallback.")
        summary_text = document_content[:truncate_char_length].strip()

    cache.set(cache_key, summary_text)
    logger.info(f"Generated/cached summary for: {document_file_path} (len: {len(summary_text)})")

    try:
        path_parts = os.path.normpath(document_file_path).split(os.sep)
        dataset_name = path_parts[0] if len(path_parts) > 0 else "unknown_dataset"
        summary_file_dir = os.path.join(summaries_output_dir_base, dataset_name)
        os.makedirs(summary_file_dir, exist_ok=True)
        original_filename_base = os.path.relpath(document_file_path, dataset_name)
        doc_file_name = os.path.splitext(original_filename_base)[0]
        sanitized_file_name = sanitize_filename(doc_file_name)
        summary_filename_txt = os.path.join(summary_file_dir,
                                            f"{sanitized_file_name}_summary.txt")
        print("document_file_path:", document_file_path)
        print("summary_filename_txt:", summary_filename_txt)
        with open(summary_filename_txt, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logger.debug(f"Summary for {document_file_path} saved to {summary_filename_txt}")
    except Exception as e:
        logger.error(f"Failed to save summary file for {document_file_path}: {e}")

    return summary_text
