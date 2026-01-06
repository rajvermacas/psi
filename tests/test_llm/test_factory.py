"""
Tests for LLM factory.
"""

import pytest

from psi.config import Config, LLMProvider
from psi.llm import (
    AzureOpenAIProvider,
    GeminiProvider,
    LLMFactory,
    OpenRouterProvider,
    create_llm_provider,
)


class TestLLMFactory:
    """Tests for LLMFactory class."""

    def test_create_azure_provider(self, azure_config: None) -> None:
        """Test creating Azure provider."""
        provider = LLMFactory.create()

        assert isinstance(provider, AzureOpenAIProvider)
        assert provider.provider_name == "azure"

    def test_create_openrouter_provider(self, openrouter_config: None) -> None:
        """Test creating OpenRouter provider."""
        provider = LLMFactory.create()

        assert isinstance(provider, OpenRouterProvider)
        assert provider.provider_name == "openrouter"

    def test_create_gemini_provider(self, gemini_config: None) -> None:
        """Test creating Gemini provider."""
        provider = LLMFactory.create()

        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"

    def test_create_with_explicit_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test creating provider with explicit config."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")

        config = Config()
        provider = LLMFactory.create(config)

        assert isinstance(provider, GeminiProvider)
        assert provider.model_name == "gemini-1.5-pro"


class TestCreateLLMProvider:
    """Tests for create_llm_provider function."""

    def test_create_llm_provider_azure(self, azure_config: None) -> None:
        """Test create_llm_provider with Azure."""
        provider = create_llm_provider()

        assert isinstance(provider, AzureOpenAIProvider)

    def test_create_llm_provider_openrouter(self, openrouter_config: None) -> None:
        """Test create_llm_provider with OpenRouter."""
        provider = create_llm_provider()

        assert isinstance(provider, OpenRouterProvider)

    def test_create_llm_provider_gemini(self, gemini_config: None) -> None:
        """Test create_llm_provider with Gemini."""
        provider = create_llm_provider()

        assert isinstance(provider, GeminiProvider)
