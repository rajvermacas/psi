"""
LLM provider factory.

Creates LLM provider instances based on configuration.
"""

import logging

from psi.config import Config, LLMProvider, get_config
from psi.exceptions import InvalidConfigError
from psi.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory for creating LLM provider instances.

    Creates the appropriate provider based on configuration,
    ensuring all required credentials are available.
    """

    @staticmethod
    def create(config: Config | None = None) -> BaseLLMProvider:
        """
        Create an LLM provider instance based on configuration.

        Args:
            config: Configuration instance. If None, uses get_config().

        Returns:
            BaseLLMProvider instance for the configured provider.

        Raises:
            InvalidConfigError: If provider type is invalid or config is missing.
        """
        if config is None:
            config = get_config()

        logger.info(f"Creating LLM provider for: {config.llm_provider.value}")

        if config.llm_provider == LLMProvider.AZURE:
            return LLMFactory._create_azure_provider(config)
        elif config.llm_provider == LLMProvider.OPENROUTER:
            return LLMFactory._create_openrouter_provider(config)
        elif config.llm_provider == LLMProvider.GEMINI:
            return LLMFactory._create_gemini_provider(config)
        else:
            raise InvalidConfigError(
                message=f"Unknown LLM provider: {config.llm_provider}",
                missing_key="LLM_PROVIDER",
            )

    @staticmethod
    def _create_azure_provider(config: Config) -> BaseLLMProvider:
        """
        Create Azure OpenAI provider.

        Args:
            config: Configuration with Azure credentials.

        Returns:
            AzureOpenAIProvider instance.
        """
        from psi.llm.azure import AzureOpenAIProvider

        logger.debug(
            f"Creating Azure provider: endpoint={config.azure_openai_endpoint}, "
            f"deployment={config.azure_openai_deployment}"
        )

        return AzureOpenAIProvider(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            deployment=config.azure_openai_deployment,
            model=config.azure_openai_model,
        )

    @staticmethod
    def _create_openrouter_provider(config: Config) -> BaseLLMProvider:
        """
        Create OpenRouter provider.

        Args:
            config: Configuration with OpenRouter credentials.

        Returns:
            OpenRouterProvider instance.
        """
        from psi.llm.openrouter import OpenRouterProvider

        logger.debug(f"Creating OpenRouter provider: model={config.openrouter_model}")

        return OpenRouterProvider(
            api_key=config.openrouter_api_key,
            model=config.openrouter_model,
        )

    @staticmethod
    def _create_gemini_provider(config: Config) -> BaseLLMProvider:
        """
        Create Gemini provider.

        Args:
            config: Configuration with Gemini credentials.

        Returns:
            GeminiProvider instance.
        """
        from psi.llm.gemini import GeminiProvider

        logger.debug(f"Creating Gemini provider: model={config.gemini_model}")

        return GeminiProvider(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
        )


def create_llm_provider(config: Config | None = None) -> BaseLLMProvider:
    """
    Convenience function to create LLM provider.

    Args:
        config: Configuration instance. If None, uses get_config().

    Returns:
        BaseLLMProvider instance for the configured provider.
    """
    return LLMFactory.create(config)
