from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Type

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.config import SMART_MODEL

load_dotenv()

_mode_json = getattr(instructor.Mode, "JSON", None)
if _mode_json is None:
    _mode_json = getattr(instructor.Mode, "MD_JSON", None)
if _mode_json is not None:
    client = instructor.from_litellm(litellm.completion, mode=_mode_json)
else:
    client = instructor.from_litellm(litellm.completion)
logger = logging.getLogger(__name__)


def _resolve_api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY", "").strip().strip('"').strip("'")



def _is_retryable_error(exc: Exception) -> bool:
    retryable_class_names = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
        "ServiceUnavailableError",
        "ValidationError",
        "JSONDecodeError",
        "JSONParseError",
        "InstructorRetryException",
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
        "invalid json",
        "json decode",
        "json parsing",
        "pydantic",
        "expecting",
        "unterminated string",
    ]
    return any(keyword in message for keyword in retryable_keywords)


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
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
    start = perf_counter()
    response_model_name = getattr(response_model, "__name__", str(response_model))
    logger.info(
        "llm.structured.start model=%s response_model=%s temperature=%.2f user_chars=%s",
        model,
        response_model_name,
        temperature,
        len(user_prompt),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        api_key = _resolve_api_key()
        create_kwargs = {}
        if "deepseek" in model.lower():
            create_kwargs["response_format"] = {"type": "json_object"}
        result = client.chat.completions.create(
            model=model,
            response_model=response_model,
            messages=messages,
            temperature=temperature,
            api_key=api_key or None,
            max_retries=5,
            **create_kwargs,
        )
        duration_ms = int((perf_counter() - start) * 1000)
        logger.info(
            "llm.structured.done model=%s response_model=%s duration_ms=%s",
            model,
            response_model_name,
            duration_ms,
        )
        return result
    except Exception:
        duration_ms = int((perf_counter() - start) * 1000)
        logger.exception(
            "llm.structured.error model=%s response_model=%s duration_ms=%s",
            model,
            response_model_name,
            duration_ms,
        )
        raise


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
    start = perf_counter()
    logger.info(
        "llm.text.start model=%s temperature=%.2f user_chars=%s",
        model,
        temperature,
        len(user_prompt),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        api_key = _resolve_api_key()
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            api_key=api_key or None,
        )
        content = response.choices[0].message.content
        text = content if isinstance(content, str) else str(content or "")
        duration_ms = int((perf_counter() - start) * 1000)
        logger.info(
            "llm.text.done model=%s duration_ms=%s output_chars=%s",
            model,
            duration_ms,
            len(text),
        )
        return text
    except Exception:
        duration_ms = int((perf_counter() - start) * 1000)
        logger.exception(
            "llm.text.error model=%s duration_ms=%s",
            model,
            duration_ms,
        )
        raise
