"""
Base LLM provider interface.

Defines the abstract interface that all LLM providers must implement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PSIClassification:
    """
    Result of PSI classification from LLM.

    Attributes:
        is_psi: Whether the news is classified as PSI.
        reasoning: LLM's explanation for the classification.
    """

    is_psi: bool
    reasoning: str


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM provider implementations (Azure, OpenRouter, Gemini) must
    inherit from this class and implement the abstract methods.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the provider name.

        Returns:
            Provider identifier (e.g., 'azure', 'openrouter', 'gemini').
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the model name being used.

        Returns:
            Model identifier (e.g., 'gpt-4', 'claude-3-sonnet').
        """
        pass

    @abstractmethod
    def classify(self, news: str, framework_context: str) -> PSIClassification:
        """
        Classify news as PSI or non-PSI.

        Args:
            news: The news content to classify.
            framework_context: Regulatory framework context for classification.

        Returns:
            PSIClassification with is_psi boolean and reasoning.

        Raises:
            LLMProviderError: If the LLM API call fails.
            LLMResponseParseError: If the response cannot be parsed.
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate that provider configuration is complete.

        Raises:
            InvalidConfigError: If configuration is invalid.
        """
        pass

    def build_system_prompt(self, framework_name: str, jurisdiction: str) -> str:
        """
        Build the system prompt for PSI classification.

        Args:
            framework_name: Name of the regulatory framework (e.g., 'SEBI').
            jurisdiction: Jurisdiction of the framework (e.g., 'India').

        Returns:
            System prompt string.
        """
        return f"""You are a financial regulatory compliance expert specializing in \
{framework_name} regulations for {jurisdiction}.

Your task is to determine whether a given piece of news constitutes \
Price Sensitive Information (PSI) under {framework_name} regulations.

You must respond with ONLY a valid JSON object in this exact format:
{{
    "is_psi": <true or false>,
    "reasoning": "<detailed explanation referencing specific criteria>"
}}

Do not include any text outside the JSON object.
The "is_psi" field must be a boolean (true or false, not strings).
The "reasoning" field must explain why the news is or is not PSI, \
referencing specific regulatory criteria."""

    def build_user_prompt(
        self,
        news: str,
        date: str,
        ticker: str,
        exchange: str,
        region: str,
        framework_context: str,
    ) -> str:
        """
        Build the user prompt for PSI classification.

        Args:
            news: The news content to classify.
            date: Date of the news.
            ticker: Stock ticker symbol.
            exchange: Stock exchange name.
            region: Geographic region.
            framework_context: Full regulatory framework context.

        Returns:
            User prompt string.
        """
        return f"""{framework_context}

---

## News to Classify

**Date:** {date}
**Ticker:** {ticker}
**Exchange:** {exchange}
**Region:** {region}

**News Content:**
{news}

---

## Instructions

Analyze the news content against the regulatory framework's PSI criteria and \
provide your classification.

Respond with ONLY a valid JSON object:
{{
    "is_psi": <true or false>,
    "reasoning": "<detailed explanation>"
}}"""

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(provider={self.provider_name!r}, model={self.model_name!r})"
