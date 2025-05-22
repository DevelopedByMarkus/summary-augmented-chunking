from abc import ABC, abstractmethod
import asyncio
import os
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AsyncOpenAI, APIError


class BaseGenerator(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        # kwargs might include temperature, max_new_tokens if common to all
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_new_tokens = kwargs.get("max_new_tokens", 256)

    @abstractmethod
    async def generate(self, prompts: list[str], **generation_kwargs) -> list[str]:
        pass


class LocalGenerator(BaseGenerator, ABC):
    def __init__(self, model_name: str, batch_size: int, device: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.batch_size = batch_size
        self.device = device


class APIGenerator(BaseGenerator, ABC):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        DEFAULT_CONCURRENCY_LIMIT = 10
        self.concurrency_limit = DEFAULT_CONCURRENCY_LIMIT


class OpenAIGenerator(APIGenerator):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)

    async def _get_completion(self, prompt: str, semaphore: asyncio.Semaphore, temperature: float,
                              max_new_tokens: int) -> str | None:
        async with semaphore:
            try:
                chat_completion = await self.client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                response_content = chat_completion.choices[0].message.content
                return response_content.strip() if response_content else None
            except APIError as e:
                # TODO: Use index and task name in the error print
                print(f"OpenAI API Error for prompt '{prompt[:50]}...': {e}")
                return f"ERROR: OpenAI API Error - {e}"
            except Exception as e:
                # TODO: Use index and task name in the error print
                print(f"An unexpected error occurred with OpenAI for prompt '{prompt[:50]}...': {e}")
                return f"ERROR: Unexpected error - {e}"

    async def generate(self, prompts: list[str], **generation_kwargs) -> list[str]:
        temperature = generation_kwargs.get("temperature", self.temperature)
        max_new_tokens = generation_kwargs.get("max_new_tokens", self.max_new_tokens)
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        tasks = []
        for prompt_text in prompts:
            if not prompt_text or not isinstance(prompt_text, str):
                print(f"Warning: Invalid or empty prompt skipped: {prompt_text}")
                tasks.append(
                    asyncio.create_task(asyncio.sleep(0, result="ERROR: Invalid prompt")))  # Ensure future has a result
                continue
            tasks.append(self._get_completion(prompt_text, semaphore, temperature, max_new_tokens))

        results = await asyncio.gather(*tasks, return_exceptions=False)  # Exceptions handled in _get_completion

        # Ensure all results are strings, even if None was returned or an error string.
        # The calling code expects a list of strings of the same length as prompts.
        final_results = []
        for i, res in enumerate(results):
            if res is None:
                final_results.append(f"ERROR: No response for prompt {i}")
            else:
                final_results.append(res)
        return final_results


class LlamaLocalGenerator(LocalGenerator):
    def __init__(self, model_name: str, batch_size: int, device: str, **kwargs):
        super().__init__(model_name, batch_size=batch_size, device=device, **kwargs)
        try:
            # Determine torch_dtype based on device capability
            if self.device == "cuda" and torch.cuda.is_available():
                dtype = torch.float16
            elif self.device == "mps" and torch.backends.mps.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32  # Default for CPU

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                # device_map="auto" can be useful for multi-GPU or very large models.
                # For single explicit device, to(self.device) is more direct.
                trust_remote_code=True
            ).to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id  # Also update model config

            self.model.eval()  # Set model to evaluation mode
        except Exception as e:
            print(f"Error initializing LlamaLocalGenerator for {model_name}: {e}")
            raise  # Re-raise exception to signal failure to initialize

    async def generate(self, prompts: list[str], **generation_kwargs) -> list[str]:
        temperature = generation_kwargs.get("temperature", self.temperature)
        max_new_tokens = generation_kwargs.get("max_new_tokens", self.max_new_tokens)

        print(
            f"Llama generating {len(prompts)} prompts for model {self.model_name} with batch_size {self.batch_size}...")
        all_generations = []

        # This method is async to match BaseGenerator, but internal HF logic is synchronous.
        # No actual `await` on I/O or true async operations here, but the signature matches.
        # If this were part of a larger asyncio application, long-running sync code should
        # ideally be run in a thread pool executor to avoid blocking the event loop.
        for i in tqdm(range(0, len(prompts), self.batch_size), desc=f"Generating {self.model_name}"):
            batch_prompts = prompts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Adjust as needed, consider model's max context
            ).to(self.device)

            # Store length of input_ids to slice off the prompt from the generated output
            input_ids_lengths = [len(ids) for ids in inputs.input_ids]

            try:
                with torch.no_grad():
                    # Ensure do_sample is True if temperature is not 1.0 or top_p/top_k are used
                    do_sample = True if temperature < 1.0 or temperature > 1.0 else False

                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        # top_p=0.9, # Example: add other generation params if needed
                    )
            except Exception as e:
                print(f"Error during model.generate for a batch: {e}")
                # Add error placeholders for this batch
                all_generations.extend([f"ERROR: Generation failed - {e}" for _ in batch_prompts])
                continue

            # Decode and slice off the prompt part
            # generated_ids shape: (batch_size, sequence_length)
            batch_results = []
            for j, output_ids in enumerate(generated_ids):
                # The generated sequence includes the input prompt.
                # We need to decode only the newly generated tokens.
                prompt_length = input_ids_lengths[j]
                # Check if generated sequence is longer than prompt (i.e., something was generated)
                if len(output_ids) > prompt_length:
                    newly_generated_ids = output_ids[prompt_length:]
                    decoded_text = self.tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
                    batch_results.append(decoded_text.strip())
                else:
                    # Nothing new was generated, or generation was shorter than expected (e.g., only EOS)
                    batch_results.append("")  # Or some indicator of no new generation

            all_generations.extend(batch_results)

            # Clean up GPU memory for large models if processing many batches, though PyTorch usually handles it.
            # del inputs, generated_ids
            # if self.device == "cuda":
            #     torch.cuda.empty_cache()

        return all_generations


def create_generator(args) -> BaseGenerator:
    """Factory function to create a generator instance."""
    common_kwargs = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    model_name_lower = args.model_name.lower()
    if "gpt-" in model_name_lower or "openai/" in model_name_lower:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via OPENAI_API_KEY environment variable for GPT models.")
        return OpenAIGenerator(model_name=args.model_name, api_key=api_key, **common_kwargs)

    elif "llama" in model_name_lower:
        return LlamaLocalGenerator(
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            **common_kwargs
        )
    else:
        raise ValueError(
            f"Unsupported model_name or unable to determine model type: {args.model_name}. "
            "Please use a full Hugging Face path (e.g., 'meta-llama/Llama-2-7b-chat-hf') for local models, "
            "or ensure it's a recognized OpenAI model name (e.g., 'gpt-4o')."
        )
