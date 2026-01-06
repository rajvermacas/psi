"""
Configuration loader for PSI Classifier.

Loads configuration from environment variables using pydantic-settings.
Follows fail-fast principle - raises InvalidConfigError if required
configuration is missing.
"""

import logging
import os
from enum import Enum
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

from psi.exceptions import InvalidConfigError

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM provider types."""

    AZURE = "azure"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Config(BaseSettings):
    """
    Application configuration loaded from environment variables.

    This class uses pydantic-settings to load and validate configuration
    from environment variables. Required fields will raise an exception
    if not provided (fail-fast principle).

    Attributes:
        llm_provider: The LLM provider to use (azure/openrouter/gemini).
        log_level: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL).

        # Azure OpenAI settings (required if llm_provider == azure)
        azure_openai_endpoint: Azure OpenAI endpoint URL.
        azure_openai_api_key: Azure OpenAI API key.
        azure_openai_api_version: Azure OpenAI API version.
        azure_openai_deployment: Azure OpenAI deployment name.
        azure_openai_model: Azure OpenAI model name.

        # OpenRouter settings (required if llm_provider == openrouter)
        openrouter_api_key: OpenRouter API key.
        openrouter_model: OpenRouter model identifier.

        # Gemini settings (required if llm_provider == gemini)
        gemini_api_key: Google Gemini API key.
        gemini_model: Gemini model name.
    """

    # Required
    llm_provider: LLMProvider

    # Logging
    log_level: LogLevel = LogLevel.INFO

    # Azure OpenAI Configuration
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_model: str | None = None

    # OpenRouter Configuration
    openrouter_api_key: str | None = None
    openrouter_model: str | None = None

    # Gemini Configuration
    gemini_api_key: str | None = None
    gemini_model: str | None = None

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("llm_provider", mode="before")
    @classmethod
    def validate_llm_provider(cls, v: Any) -> LLMProvider:
        """
        Validate and convert LLM provider value.

        Args:
            v: Input value for llm_provider.

        Returns:
            Validated LLMProvider enum value.

        Raises:
            InvalidConfigError: If provider is not one of azure/openrouter/gemini.
        """
        if v is None:
            raise InvalidConfigError(
                message=(
                    "LLM_PROVIDER environment variable is required. "
                    "Set it to one of: azure, openrouter, gemini"
                ),
                missing_key="LLM_PROVIDER",
            )
        if isinstance(v, LLMProvider):
            return v
        try:
            return LLMProvider(v.lower())
        except ValueError:
            raise InvalidConfigError(
                message=(
                    f"Invalid LLM_PROVIDER value: '{v}'. "
                    "Must be one of: azure, openrouter, gemini"
                ),
                missing_key="LLM_PROVIDER",
            )

    @model_validator(mode="after")
    def validate_provider_config(self) -> "Config":
        """
        Validate that required provider-specific configuration is present.

        Raises:
            InvalidConfigError: If required provider-specific config is missing.

        Returns:
            Validated Config instance.
        """
        logger.debug(f"Validating configuration for provider: {self.llm_provider}")

        if self.llm_provider == LLMProvider.AZURE:
            self._validate_azure_config()
        elif self.llm_provider == LLMProvider.OPENROUTER:
            self._validate_openrouter_config()
        elif self.llm_provider == LLMProvider.GEMINI:
            self._validate_gemini_config()

        logger.info(
            f"Configuration validated successfully for provider: {self.llm_provider.value}"
        )
        return self

    def _validate_azure_config(self) -> None:
        """
        Validate Azure OpenAI configuration.

        Raises:
            InvalidConfigError: If required Azure config is missing.
        """
        required_fields = [
            ("azure_openai_endpoint", "AZURE_OPENAI_ENDPOINT"),
            ("azure_openai_api_key", "AZURE_OPENAI_API_KEY"),
            ("azure_openai_api_version", "AZURE_OPENAI_API_VERSION"),
            ("azure_openai_deployment", "AZURE_OPENAI_DEPLOYMENT"),
            ("azure_openai_model", "AZURE_OPENAI_MODEL"),
        ]

        for field_name, env_name in required_fields:
            value = getattr(self, field_name)
            if not value:
                raise InvalidConfigError(
                    message=(
                        f"{env_name} is required when LLM_PROVIDER=azure. "
                        "Please set this environment variable."
                    ),
                    missing_key=env_name,
                )

        logger.debug(
            f"Azure config validated: endpoint={self.azure_openai_endpoint}, "
            f"deployment={self.azure_openai_deployment}, "
            f"model={self.azure_openai_model}"
        )

    def _validate_openrouter_config(self) -> None:
        """
        Validate OpenRouter configuration.

        Raises:
            InvalidConfigError: If required OpenRouter config is missing.
        """
        required_fields = [
            ("openrouter_api_key", "OPENROUTER_API_KEY"),
            ("openrouter_model", "OPENROUTER_MODEL"),
        ]

        for field_name, env_name in required_fields:
            value = getattr(self, field_name)
            if not value:
                raise InvalidConfigError(
                    message=(
                        f"{env_name} is required when LLM_PROVIDER=openrouter. "
                        "Please set this environment variable."
                    ),
                    missing_key=env_name,
                )

        logger.debug(f"OpenRouter config validated: model={self.openrouter_model}")

    def _validate_gemini_config(self) -> None:
        """
        Validate Gemini configuration.

        Raises:
            InvalidConfigError: If required Gemini config is missing.
        """
        required_fields = [
            ("gemini_api_key", "GEMINI_API_KEY"),
            ("gemini_model", "GEMINI_MODEL"),
        ]

        for field_name, env_name in required_fields:
            value = getattr(self, field_name)
            if not value:
                raise InvalidConfigError(
                    message=(
                        f"{env_name} is required when LLM_PROVIDER=gemini. "
                        "Please set this environment variable."
                    ),
                    missing_key=env_name,
                )

        logger.debug(f"Gemini config validated: model={self.gemini_model}")

    def get_model_name(self) -> str:
        """
        Get the model name for the configured provider.

        Returns:
            Model name string based on the configured provider.

        Raises:
            InvalidConfigError: If model name cannot be determined.
        """
        if self.llm_provider == LLMProvider.AZURE:
            if not self.azure_openai_model:
                raise InvalidConfigError(
                    message="AZURE_OPENAI_MODEL is not configured",
                    missing_key="AZURE_OPENAI_MODEL",
                )
            return self.azure_openai_model
        elif self.llm_provider == LLMProvider.OPENROUTER:
            if not self.openrouter_model:
                raise InvalidConfigError(
                    message="OPENROUTER_MODEL is not configured",
                    missing_key="OPENROUTER_MODEL",
                )
            return self.openrouter_model
        elif self.llm_provider == LLMProvider.GEMINI:
            if not self.gemini_model:
                raise InvalidConfigError(
                    message="GEMINI_MODEL is not configured",
                    missing_key="GEMINI_MODEL",
                )
            return self.gemini_model
        else:
            raise InvalidConfigError(
                message=f"Unknown LLM provider: {self.llm_provider}",
                missing_key="LLM_PROVIDER",
            )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get the application configuration (cached).

    This function loads configuration from environment variables
    and .env file. The result is cached for subsequent calls.

    Returns:
        Validated Config instance.

    Raises:
        InvalidConfigError: If configuration is invalid or incomplete.
    """
    logger.info("Loading configuration from environment")

    # Load .env file if present
    env_file = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_file):
        logger.debug(f"Loading environment from: {env_file}")
        load_dotenv(env_file)
    else:
        logger.debug("No .env file found, using environment variables only")

    try:
        config = Config()
        logger.info(
            f"Configuration loaded: provider={config.llm_provider.value}, "
            f"log_level={config.log_level.value}"
        )
        return config
    except Exception as e:
        # Re-raise InvalidConfigError as-is
        if isinstance(e, InvalidConfigError):
            raise
        # Wrap other validation errors
        raise InvalidConfigError(
            message=f"Failed to load configuration: {str(e)}",
            missing_key=None,
        ) from e


def setup_logging(config: Config | None = None) -> None:
    """
    Configure logging based on application configuration.

    Args:
        config: Optional Config instance. If not provided, uses get_config().
    """
    if config is None:
        config = get_config()

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Get the root logger and set its level explicitly
    root_logger = logging.getLogger()

    # Remove existing handlers and add a fresh one
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(handler)
    root_logger.setLevel(config.log_level.value)

    logger.info(f"Logging configured with level: {config.log_level.value}")
