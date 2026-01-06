"""
Tests for PSI Classifier configuration module.

Tests cover configuration loading, validation, and provider-specific settings.
"""

import os

import pytest

from psi.config import Config, LLMProvider, LogLevel, get_config, setup_logging
from psi.exceptions import InvalidConfigError


class TestLLMProviderEnum:
    """Tests for LLMProvider enum."""

    def test_enum_values(self) -> None:
        """Test that all expected provider values exist."""
        assert LLMProvider.AZURE.value == "azure"
        assert LLMProvider.OPENROUTER.value == "openrouter"
        assert LLMProvider.GEMINI.value == "gemini"

    def test_enum_from_string(self) -> None:
        """Test creating enum from string value."""
        assert LLMProvider("azure") == LLMProvider.AZURE
        assert LLMProvider("openrouter") == LLMProvider.OPENROUTER
        assert LLMProvider("gemini") == LLMProvider.GEMINI


class TestLogLevelEnum:
    """Tests for LogLevel enum."""

    def test_enum_values(self) -> None:
        """Test that all expected log level values exist."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestConfigAzure:
    """Tests for Azure OpenAI configuration."""

    def test_valid_azure_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test valid Azure configuration."""
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4")

        config = Config()

        assert config.llm_provider == LLMProvider.AZURE
        assert config.azure_openai_endpoint == "https://test.openai.azure.com/"
        assert config.azure_openai_api_key == "test-key"
        assert config.azure_openai_api_version == "2024-02-15-preview"
        assert config.azure_openai_deployment == "gpt-4"
        assert config.azure_openai_model == "gpt-4"

    def test_azure_missing_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Azure config missing endpoint raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)
        assert exc_info.value.missing_key == "AZURE_OPENAI_ENDPOINT"

    def test_azure_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Azure config missing API key raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "AZURE_OPENAI_API_KEY" in str(exc_info.value)

    def test_azure_missing_api_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Azure config missing API version raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "AZURE_OPENAI_API_VERSION" in str(exc_info.value)

    def test_azure_missing_deployment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Azure config missing deployment raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "AZURE_OPENAI_DEPLOYMENT" in str(exc_info.value)

    def test_azure_missing_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Azure config missing model raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "AZURE_OPENAI_MODEL" in str(exc_info.value)


class TestConfigOpenRouter:
    """Tests for OpenRouter configuration."""

    def test_valid_openrouter_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test valid OpenRouter configuration."""
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3-sonnet")

        config = Config()

        assert config.llm_provider == LLMProvider.OPENROUTER
        assert config.openrouter_api_key == "test-key"
        assert config.openrouter_model == "anthropic/claude-3-sonnet"

    def test_openrouter_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test OpenRouter config missing API key raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3-sonnet")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "OPENROUTER_API_KEY" in str(exc_info.value)
        assert exc_info.value.missing_key == "OPENROUTER_API_KEY"

    def test_openrouter_missing_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test OpenRouter config missing model raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "OPENROUTER_MODEL" in str(exc_info.value)


class TestConfigGemini:
    """Tests for Gemini configuration."""

    def test_valid_gemini_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test valid Gemini configuration."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")

        config = Config()

        assert config.llm_provider == LLMProvider.GEMINI
        assert config.gemini_api_key == "test-key"
        assert config.gemini_model == "gemini-1.5-pro"

    def test_gemini_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Gemini config missing API key raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "GEMINI_API_KEY" in str(exc_info.value)
        assert exc_info.value.missing_key == "GEMINI_API_KEY"

    def test_gemini_missing_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Gemini config missing model raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "GEMINI_MODEL" in str(exc_info.value)


class TestConfigProviderValidation:
    """Tests for LLM provider validation."""

    def test_missing_provider_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test missing LLM_PROVIDER raises error via get_config."""
        # Clear any existing LLM_PROVIDER
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        # Use get_config which wraps pydantic's ValidationError into InvalidConfigError
        with pytest.raises(InvalidConfigError) as exc_info:
            get_config()

        assert "llm_provider" in str(exc_info.value).lower() or "LLM_PROVIDER" in str(exc_info.value)

    def test_invalid_provider_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test invalid LLM_PROVIDER value raises error."""
        monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")

        with pytest.raises(InvalidConfigError) as exc_info:
            Config()

        assert "invalid_provider" in str(exc_info.value)
        assert "azure, openrouter, gemini" in str(exc_info.value)

    def test_provider_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test provider name is case insensitive."""
        monkeypatch.setenv("LLM_PROVIDER", "AZURE")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4")

        config = Config()
        assert config.llm_provider == LLMProvider.AZURE


class TestConfigLogLevel:
    """Tests for log level configuration."""

    def test_default_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default log level is INFO."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")

        config = Config()
        assert config.log_level == LogLevel.INFO

    def test_custom_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test custom log level."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        config = Config()
        assert config.log_level == LogLevel.DEBUG


class TestConfigGetModelName:
    """Tests for get_model_name method."""

    def test_get_model_name_azure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_model_name returns Azure model."""
        monkeypatch.setenv("LLM_PROVIDER", "azure")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4-turbo")

        config = Config()
        assert config.get_model_name() == "gpt-4-turbo"

    def test_get_model_name_openrouter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_model_name returns OpenRouter model."""
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3-opus")

        config = Config()
        assert config.get_model_name() == "anthropic/claude-3-opus"

    def test_get_model_name_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_model_name returns Gemini model."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")

        config = Config()
        assert config.get_model_name() == "gemini-1.5-pro"


class TestGetConfigFunction:
    """Tests for get_config function."""

    def test_get_config_caching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_config is cached."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")

        # Clear cache before test
        get_config.cache_clear()

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_get_config_raises_on_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_config raises InvalidConfigError on invalid config."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        # Clear cache before test
        get_config.cache_clear()

        with pytest.raises(InvalidConfigError):
            get_config()


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_with_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test setup_logging configures logging correctly."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        config = Config()
        setup_logging(config)

        import logging

        # Root logger should have DEBUG level set
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG
