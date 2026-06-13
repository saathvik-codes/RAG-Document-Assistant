from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from rag_assistant.config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.strip().lower()

    if provider == "ollama":
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=openai requires langchain-openai. Install optional dependencies first."
            ) from exc
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=anthropic requires langchain-anthropic. Install optional dependencies first."
            ) from exc
        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=google requires langchain-google-genai. Install optional dependencies first."
            ) from exc
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use one of: ollama, openai, anthropic, google."
    )


def active_model_name(settings: Settings) -> str:
    provider = settings.llm_provider.strip().lower()
    if provider == "ollama":
        return settings.ollama_model
    if provider == "openai":
        return settings.openai_model
    if provider == "anthropic":
        return settings.anthropic_model
    if provider == "google":
        return settings.google_model
    return provider
