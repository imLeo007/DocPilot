from deepeval.models.base_model import DeepEvalBaseLLM

from openai import OpenAI, AsyncOpenAI

import asyncio

from app.core.config import settings


class CustomLLMJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.openai_model
        
        client_kwargs = {
            "api_key": settings.openai_api_key,
            "base_url": "https://aicredits.in/v1",
            "timeout": 30.0,
            "max_retries": 5,
        }

        self.client = OpenAI(**client_kwargs)
        
        self.aclient = AsyncOpenAI(**client_kwargs)

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(1)

        async with self._semaphore:
            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            await asyncio.sleep(0.5)
            return response.choices[0].message.content or ""


judge = CustomLLMJudge()