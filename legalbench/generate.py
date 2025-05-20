from abc import ABC, abstractmethod
import asyncio
import os


class BaseGenerator(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    async def generate(self, prompts: list[str], **generation_kwargs) -> list[str]:
        pass


class LocalGenerator(BaseGenerator, ABC):
    def __init__(self, model_name: str, batch_size: int, device: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.batch_size = batch_size
        self.device = device


class APIGenerator(BaseGenerator, ABC):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key


class OpenAIGenerator(APIGenerator):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.api_key)
        print(f"Initialized OpenAIGenerator for model: {self.model_name}")

    async def generate(self, prompts: list[str], temperature: float = 0.7, max_new_tokens: int = 256, **kwargs) -> list[
        str]:
        print(f"OpenAI generating {len(prompts)} prompts (placeholder)...")
        # Actual implementation:
        # - Use asyncio.Semaphore for concurrency control
        # - Loop through prompts, create tasks for self.client.chat.completions.create(...)
        # - Use asyncio.gather to run tasks concurrently
        # - Handle responses and errors
        # For placeholder:
        await asyncio.sleep(0.1)  # Simulate async work
        return [f"OpenAI response for: {p[:30]}..." for p in prompts]


class LlamaLocalGenerator(LocalGenerator):
    def __init__(self, model_name: str, device: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.device = device
        # Initialize Hugging Face model and tokenizer here
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Initialized LlamaLocalGenerator for model: {self.model_name} on device: {self.device}")

    async def generate(self, prompts: list[str], batch_size: int = 8, temperature: float = 0.7,
                       max_new_tokens: int = 256, **kwargs) -> list[str]:
        print(f"Llama generating {len(prompts)} prompts with batch_size {batch_size} (placeholder)...")
        # Actual implementation:
        # - Loop through prompts in batches
        # - Tokenize batch: self.tokenizer(batch_prompts, ..., return_tensors="pt").to(self.device)
        # - Generate: with torch.no_grad(): self.model.generate(...)
        # - Decode: self.tokenizer.batch_decode(...)
        # - This method is async to match BaseGenerator, but internal logic is synchronous.
        all_generations = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            # Simulate processing
            await asyncio.sleep(0.05 * len(batch_prompts))  # Simulate async work (though actual work is sync)
            all_generations.extend([f"Llama response for: {p[:30]}..." for p in batch_prompts])
        return all_generations


def create_generator(args) -> BaseGenerator:
    """Factory function to create a generator instance."""
    common_kwargs = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    if "gpt" in args.model_name.lower():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via OPENAI_API_KEY environment variable for GPT models.")
        return OpenAIGenerator(model_name=args.model_name, api_key=api_key, **common_kwargs)
    elif "llama" in args.model_name.lower():
        return LlamaLocalGenerator(model_name=args.model_name, batch_size=args.batch_size, device=args.device, **common_kwargs)
    else:
        # Fallback or error for unsupported models
        # For now, let's try a generic local model load attempt if not OpenAI
        print(
            f"Warning: Model '{args.model_name}' not explicitly handled. Attempting generic local load (Llama-like structure).")
        print(f"Ensure '{args.model_name}' is a valid Hugging Face model path for causal LM.")
        return LlamaLocalGenerator(model_name=args.model_name, device=args.device, **common_kwargs)
