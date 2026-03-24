from __future__ import annotations

from typing import Type

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import SMART_MODEL

load_dotenv()

client = instructor.from_litellm(litellm.completion)


def _is_retryable_error(exc: Exception) -> bool:
    retryable_class_names = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
        "ServiceUnavailableError",
    }
    if exc.__class__.__name__ in retryable_class_names:
        return True

    message = str(exc).lower()
    retryable_keywords = [
        "rate limit",
        "too many requests",
        "429",
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporarily unavailable",
    ]
    return any(keyword in message for keyword in retryable_keywords)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception(_is_retryable_error),
)
def generate_structured_data(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[BaseModel],
    model: str = SMART_MODEL,
    temperature: float = 0.3,
) -> BaseModel:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        messages=messages,
        temperature=temperature,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception(_is_retryable_error),
)
def generate_text(
    system_prompt: str,
    user_prompt: str,
    model: str = SMART_MODEL,
    temperature: float = 0.7,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content if isinstance(content, str) else str(content or "")
