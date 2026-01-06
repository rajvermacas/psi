"""
OpenRouter LLM provider implementation.

Uses OpenRouter API to access various LLM models.
"""

import json
import logging

import httpx

from psi.exceptions import InvalidConfigError, LLMProviderError, LLMResponseParseError
from psi.llm.base import BaseLLMProvider, PSIClassification

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter LLM provider.

    Uses OpenRouter API for PSI classification, supporting various models
    like Claude, GPT-4, etc.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str | None,
    ) -> None:
        """
        Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key.
            model: Model identifier (e.g., 'anthropic/claude-3-sonnet').
        """
        self._api_key = api_key
        self._model = model

        logger.debug(f"OpenRouterProvider initialized: model={model}")

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "openrouter"

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model or "unknown"

    def validate_config(self) -> None:
        """Validate provider configuration."""
        if not self._api_key:
            raise InvalidConfigError(
                message="OpenRouter API key is required",
                missing_key="OPENROUTER_API_KEY",
            )
        if not self._model:
            raise InvalidConfigError(
                message="OpenRouter model is required",
                missing_key="OPENROUTER_MODEL",
            )

        logger.info("OpenRouter configuration validated successfully")

    def classify(self, news: str, framework_context: str) -> PSIClassification:
        """
        Classify news using OpenRouter.

        Args:
            news: News content to classify.
            framework_context: Regulatory framework context.

        Returns:
            PSIClassification result.

        Raises:
            LLMProviderError: If API call fails.
            LLMResponseParseError: If response cannot be parsed.
        """
        self.validate_config()

        logger.info(f"Classifying news using OpenRouter ({self._model})")
        logger.debug(f"News content: {news[:200]}...")

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

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/psi-classifier",
            "X-Title": "PSI Classifier",
        }

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }

        try:
            logger.debug("Sending request to OpenRouter")

            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                )

            if response.status_code != 200:
                raise LLMProviderError(
                    message=f"OpenRouter API returned status {response.status_code}: {response.text}",
                    provider=self.provider_name,
                    status_code=response.status_code,
                )

            data = response.json()
            response_text = data["choices"][0]["message"]["content"]

            logger.debug(f"Received response: {response_text[:500]}...")

        except httpx.RequestError as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise LLMProviderError(
                message=f"OpenRouter request failed: {str(e)}",
                provider=self.provider_name,
                original_error=e,
            ) from e
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse OpenRouter response structure: {e}")
            raise LLMProviderError(
                message=f"Invalid OpenRouter response structure: {str(e)}",
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
