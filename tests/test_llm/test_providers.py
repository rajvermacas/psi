"""
Tests for LLM providers (mocked).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from psi.exceptions import InvalidConfigError, LLMProviderError, LLMResponseParseError
from psi.llm import (
    AzureOpenAIProvider,
    BaseLLMProvider,
    GeminiProvider,
    OpenRouterProvider,
    PSIClassification,
)


class TestPSIClassification:
    """Tests for PSIClassification dataclass."""

    def test_creation(self) -> None:
        """Test creating PSIClassification."""
        result = PSIClassification(
            is_psi=True,
            reasoning="This is PSI because...",
        )

        assert result.is_psi is True
        assert result.reasoning == "This is PSI because..."

    def test_not_psi(self) -> None:
        """Test creating non-PSI classification."""
        result = PSIClassification(
            is_psi=False,
            reasoning="This is NOT PSI because...",
        )

        assert result.is_psi is False


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider interface."""

    def test_build_system_prompt(self) -> None:
        """Test building system prompt."""
        # Use a concrete implementation for testing
        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        prompt = provider.build_system_prompt("SEBI", "India")

        assert "SEBI" in prompt
        assert "India" in prompt
        assert "JSON" in prompt
        assert "is_psi" in prompt
        assert "reasoning" in prompt

    def test_repr(self) -> None:
        """Test provider repr."""
        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        repr_str = repr(provider)
        assert "azure" in repr_str
        assert "gpt-4" in repr_str


class TestAzureOpenAIProvider:
    """Tests for AzureOpenAIProvider."""

    def test_provider_name(self) -> None:
        """Test provider name."""
        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        assert provider.provider_name == "azure"

    def test_model_name(self) -> None:
        """Test model name."""
        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4-turbo",
        )

        assert provider.model_name == "gpt-4-turbo"

    def test_validate_config_missing_endpoint(self) -> None:
        """Test validation with missing endpoint."""
        provider = AzureOpenAIProvider(
            endpoint=None,
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        with pytest.raises(InvalidConfigError) as exc_info:
            provider.validate_config()

        assert "endpoint" in str(exc_info.value).lower()

    def test_validate_config_missing_api_key(self) -> None:
        """Test validation with missing API key."""
        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key=None,
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        with pytest.raises(InvalidConfigError) as exc_info:
            provider.validate_config()

        assert "api key" in str(exc_info.value).lower()

    @patch("psi.llm.azure.AzureChatOpenAI")
    def test_classify_success(self, mock_client_class: MagicMock) -> None:
        """Test successful classification."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.content = '{"is_psi": true, "reasoning": "Test reasoning"}'
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        result = provider.classify(
            news="Company announces Q4 earnings",
            framework_context="SEBI PSI criteria...",
        )

        assert result.is_psi is True
        assert result.reasoning == "Test reasoning"

    @patch("psi.llm.azure.AzureChatOpenAI")
    def test_classify_with_markdown_response(self, mock_client_class: MagicMock) -> None:
        """Test classification with markdown-wrapped JSON response."""
        mock_response = MagicMock()
        mock_response.content = '```json\n{"is_psi": false, "reasoning": "Not PSI"}\n```'
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        result = provider.classify(
            news="CEO attends conference",
            framework_context="SEBI PSI criteria...",
        )

        assert result.is_psi is False

    @patch("psi.llm.azure.AzureChatOpenAI")
    def test_classify_api_error(self, mock_client_class: MagicMock) -> None:
        """Test API error handling."""
        mock_client = MagicMock()
        mock_client.invoke.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        with pytest.raises(LLMProviderError) as exc_info:
            provider.classify(
                news="Test news",
                framework_context="Context",
            )

        assert exc_info.value.provider == "azure"

    @patch("psi.llm.azure.AzureChatOpenAI")
    def test_classify_invalid_json(self, mock_client_class: MagicMock) -> None:
        """Test invalid JSON response handling."""
        mock_response = MagicMock()
        mock_response.content = "This is not JSON"
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        with pytest.raises(LLMResponseParseError):
            provider.classify(
                news="Test news",
                framework_context="Context",
            )

    @patch("psi.llm.azure.AzureChatOpenAI")
    def test_classify_missing_fields(self, mock_client_class: MagicMock) -> None:
        """Test response with missing required fields."""
        mock_response = MagicMock()
        mock_response.content = '{"is_psi": true}'  # Missing reasoning
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        with pytest.raises(LLMResponseParseError) as exc_info:
            provider.classify(
                news="Test news",
                framework_context="Context",
            )

        assert "reasoning" in exc_info.value.missing_fields


class TestOpenRouterProvider:
    """Tests for OpenRouterProvider."""

    def test_provider_name(self) -> None:
        """Test provider name."""
        provider = OpenRouterProvider(
            api_key="test-key",
            model="anthropic/claude-3-sonnet",
        )

        assert provider.provider_name == "openrouter"

    def test_model_name(self) -> None:
        """Test model name."""
        provider = OpenRouterProvider(
            api_key="test-key",
            model="anthropic/claude-3-opus",
        )

        assert provider.model_name == "anthropic/claude-3-opus"

    def test_validate_config_missing_api_key(self) -> None:
        """Test validation with missing API key."""
        provider = OpenRouterProvider(
            api_key=None,
            model="anthropic/claude-3-sonnet",
        )

        with pytest.raises(InvalidConfigError) as exc_info:
            provider.validate_config()

        assert "api key" in str(exc_info.value).lower()

    def test_validate_config_missing_model(self) -> None:
        """Test validation with missing model."""
        provider = OpenRouterProvider(
            api_key="test-key",
            model=None,
        )

        with pytest.raises(InvalidConfigError) as exc_info:
            provider.validate_config()

        assert "model" in str(exc_info.value).lower()

    @patch("psi.llm.openrouter.httpx.Client")
    def test_classify_success(self, mock_client_class: MagicMock) -> None:
        """Test successful classification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"is_psi": true, "reasoning": "Test reasoning"}'
                    }
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OpenRouterProvider(
            api_key="test-key",
            model="anthropic/claude-3-sonnet",
        )

        result = provider.classify(
            news="Company announces Q4 earnings",
            framework_context="SEBI PSI criteria...",
        )

        assert result.is_psi is True
        assert result.reasoning == "Test reasoning"

    @patch("psi.llm.openrouter.httpx.Client")
    def test_classify_api_error(self, mock_client_class: MagicMock) -> None:
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = OpenRouterProvider(
            api_key="test-key",
            model="anthropic/claude-3-sonnet",
        )

        with pytest.raises(LLMProviderError) as exc_info:
            provider.classify(
                news="Test news",
                framework_context="Context",
            )

        assert exc_info.value.status_code == 401


class TestGeminiProvider:
    """Tests for GeminiProvider."""

    def test_provider_name(self) -> None:
        """Test provider name."""
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-1.5-pro",
        )

        assert provider.provider_name == "gemini"

    def test_model_name(self) -> None:
        """Test model name."""
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-1.5-pro",
        )

        assert provider.model_name == "gemini-1.5-pro"

    def test_validate_config_missing_api_key(self) -> None:
        """Test validation with missing API key."""
        provider = GeminiProvider(
            api_key=None,
            model="gemini-1.5-pro",
        )

        with pytest.raises(InvalidConfigError) as exc_info:
            provider.validate_config()

        assert "api key" in str(exc_info.value).lower()

    def test_validate_config_missing_model(self) -> None:
        """Test validation with missing model."""
        provider = GeminiProvider(
            api_key="test-key",
            model=None,
        )

        with pytest.raises(InvalidConfigError) as exc_info:
            provider.validate_config()

        assert "model" in str(exc_info.value).lower()

    @patch("psi.llm.gemini.ChatGoogleGenerativeAI")
    def test_classify_success(self, mock_client_class: MagicMock) -> None:
        """Test successful classification."""
        mock_response = MagicMock()
        mock_response.content = '{"is_psi": false, "reasoning": "Not PSI because..."}'
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-1.5-pro",
        )

        result = provider.classify(
            news="CEO attends conference",
            framework_context="SEBI PSI criteria...",
        )

        assert result.is_psi is False
        assert "Not PSI" in result.reasoning

    @patch("psi.llm.gemini.ChatGoogleGenerativeAI")
    def test_classify_api_error(self, mock_client_class: MagicMock) -> None:
        """Test API error handling."""
        mock_client = MagicMock()
        mock_client.invoke.side_effect = Exception("Gemini API Error")
        mock_client_class.return_value = mock_client

        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-1.5-pro",
        )

        with pytest.raises(LLMProviderError) as exc_info:
            provider.classify(
                news="Test news",
                framework_context="Context",
            )

        assert exc_info.value.provider == "gemini"


class TestResponseParsing:
    """Tests for response parsing across all providers."""

    @patch("psi.llm.azure.AzureChatOpenAI")
    def test_string_boolean_conversion(self, mock_client_class: MagicMock) -> None:
        """Test that string 'true'/'false' is converted to boolean."""
        mock_response = MagicMock()
        mock_response.content = '{"is_psi": "true", "reasoning": "Test"}'
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = AzureOpenAIProvider(
            endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            api_version="2024-02-15",
            deployment="gpt-4",
            model="gpt-4",
        )

        result = provider.classify(
            news="Test news",
            framework_context="Context",
        )

        assert result.is_psi is True  # String "true" converted to boolean
