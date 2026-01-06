"""
Google Gemini LLM provider implementation.
"""

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from psi.exceptions import InvalidConfigError, LLMProviderError, LLMResponseParseError
from psi.llm.base import BaseLLMProvider, PSIClassification

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider.

    Uses Google's Gemini models for PSI classification.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str | None,
    ) -> None:
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key.
            model: Gemini model name (e.g., 'gemini-1.5-pro').
        """
        self._api_key = api_key
        self._model = model
        self._client: ChatGoogleGenerativeAI | None = None

        logger.debug(f"GeminiProvider initialized: model={model}")

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "gemini"

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model or "unknown"

    def validate_config(self) -> None:
        """Validate provider configuration."""
        if not self._api_key:
            raise InvalidConfigError(
                message="Gemini API key is required",
                missing_key="GEMINI_API_KEY",
            )
        if not self._model:
            raise InvalidConfigError(
                message="Gemini model is required",
                missing_key="GEMINI_MODEL",
            )

        logger.info("Gemini configuration validated successfully")

    def _get_client(self) -> ChatGoogleGenerativeAI:
        """Get or create the Gemini client."""
        if self._client is None:
            self.validate_config()

            logger.debug("Creating ChatGoogleGenerativeAI client")

            self._client = ChatGoogleGenerativeAI(
                google_api_key=self._api_key,
                model=self._model,
                temperature=0.0,  # Deterministic output for classification
            )

        return self._client

    def classify(self, news: str, framework_context: str) -> PSIClassification:
        """
        Classify news using Gemini.

        Args:
            news: News content to classify.
            framework_context: Regulatory framework context.

        Returns:
            PSIClassification result.

        Raises:
            LLMProviderError: If API call fails.
            LLMResponseParseError: If response cannot be parsed.
        """
        logger.info(f"Classifying news using Gemini ({self._model})")
        logger.debug(f"News content: {news[:200]}...")

        client = self._get_client()

        # Build messages for chat
        system_prompt = self.build_system_prompt("the regulatory framework", "the jurisdiction")
        user_prompt = f"""{framework_context}

---

## News to Classify

**News Content:**
{news}

---

## Instructions

Analyze the news content against the regulatory framework's PSI criteria.

Respond with ONLY a valid JSON object:
{{
    "is_psi": <true or false>,
    "reasoning": "<detailed explanation>"
}}"""

        # Gemini uses a combined prompt approach
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            logger.debug("Sending request to Gemini")
            response = client.invoke(combined_prompt)
            response_text = response.content

            logger.debug(f"Received response: {response_text[:500]}...")

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise LLMProviderError(
                message=f"Gemini API call failed: {str(e)}",
                provider=self.provider_name,
                original_error=e,
            ) from e

        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> PSIClassification:
        """
        Parse LLM response into PSIClassification.

        Args:
            response_text: Raw response from LLM.

        Returns:
            PSIClassification object.

        Raises:
            LLMResponseParseError: If response cannot be parsed.
        """
        logger.debug(f"Parsing response: {response_text[:200]}...")

        # Clean up the response - sometimes LLMs add markdown code blocks
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(
                message=f"Failed to parse JSON response: {str(e)}",
                raw_response=response_text,
            ) from e

        # Validate required fields
        missing_fields = []
        if "is_psi" not in data:
            missing_fields.append("is_psi")
        if "reasoning" not in data:
            missing_fields.append("reasoning")

        if missing_fields:
            raise LLMResponseParseError(
                message=f"Response missing required fields: {missing_fields}",
                raw_response=response_text,
                missing_fields=missing_fields,
            )

        # Validate is_psi is boolean
        is_psi = data["is_psi"]
        if isinstance(is_psi, str):
            is_psi = is_psi.lower() == "true"
        elif not isinstance(is_psi, bool):
            raise LLMResponseParseError(
                message=f"'is_psi' must be a boolean, got: {type(is_psi).__name__}",
                raw_response=response_text,
            )

        result = PSIClassification(
            is_psi=is_psi,
            reasoning=str(data["reasoning"]),
        )

        logger.info(f"Classification result: is_psi={result.is_psi}")
        logger.debug(f"Reasoning: {result.reasoning[:200]}...")

        return result
