"""
Azure OpenAI LLM provider implementation.
"""

import json
import logging

from langchain_openai import AzureChatOpenAI

from psi.exceptions import InvalidConfigError, LLMProviderError, LLMResponseParseError
from psi.llm.base import BaseLLMProvider, PSIClassification

logger = logging.getLogger(__name__)


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI LLM provider.

    Uses Azure OpenAI Service for PSI classification.
    """

    def __init__(
        self,
        endpoint: str | None,
        api_key: str | None,
        api_version: str | None,
        deployment: str | None,
        model: str | None,
    ) -> None:
        """
        Initialize Azure OpenAI provider.

        Args:
            endpoint: Azure OpenAI endpoint URL.
            api_key: Azure OpenAI API key.
            api_version: Azure OpenAI API version.
            deployment: Azure OpenAI deployment name.
            model: Azure OpenAI model name.
        """
        self._endpoint = endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._deployment = deployment
        self._model = model
        self._client: AzureChatOpenAI | None = None

        logger.debug(
            f"AzureOpenAIProvider initialized: endpoint={endpoint}, "
            f"deployment={deployment}, model={model}"
        )

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "azure"

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model or "unknown"

    def validate_config(self) -> None:
        """Validate provider configuration."""
        if not self._endpoint:
            raise InvalidConfigError(
                message="Azure endpoint is required",
                missing_key="AZURE_OPENAI_ENDPOINT",
            )
        if not self._api_key:
            raise InvalidConfigError(
                message="Azure API key is required",
                missing_key="AZURE_OPENAI_API_KEY",
            )
        if not self._api_version:
            raise InvalidConfigError(
                message="Azure API version is required",
                missing_key="AZURE_OPENAI_API_VERSION",
            )
        if not self._deployment:
            raise InvalidConfigError(
                message="Azure deployment is required",
                missing_key="AZURE_OPENAI_DEPLOYMENT",
            )
        if not self._model:
            raise InvalidConfigError(
                message="Azure model is required",
                missing_key="AZURE_OPENAI_MODEL",
            )

        logger.info("Azure OpenAI configuration validated successfully")

    def _get_client(self) -> AzureChatOpenAI:
        """Get or create the Azure OpenAI client."""
        if self._client is None:
            self.validate_config()

            logger.debug("Creating AzureChatOpenAI client")

            self._client = AzureChatOpenAI(
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
                api_version=self._api_version,
                azure_deployment=self._deployment,
                model=self._model,
                temperature=0.0,  # Deterministic output for classification
            )

        return self._client

    def classify(self, news: str, framework_context: str) -> PSIClassification:
        """
        Classify news using Azure OpenAI.

        Args:
            news: News content to classify.
            framework_context: Regulatory framework context.

        Returns:
            PSIClassification result.

        Raises:
            LLMProviderError: If API call fails.
            LLMResponseParseError: If response cannot be parsed.
        """
        logger.info(f"Classifying news using Azure OpenAI ({self._model})")
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

        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]

        try:
            logger.debug("Sending request to Azure OpenAI")
            response = client.invoke(messages)
            response_text = response.content

            logger.debug(f"Received response: {response_text[:500]}...")

        except Exception as e:
            logger.error(f"Azure OpenAI API call failed: {e}")
            raise LLMProviderError(
                message=f"Azure OpenAI API call failed: {str(e)}",
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
