"""
Custom exception classes for the PSI Classifier.

Exception Hierarchy:
    PSIException (Base)
    ├── ConfigError
    │   └── InvalidConfigError
    ├── ValidationError
    │   ├── CSVValidationError
    │   └── StateValidationError
    └── ProcessingError
        ├── UnknownRegionError
        ├── LLMProviderError
        └── LLMResponseParseError

All exceptions follow the fail-fast principle - no fallbacks, no defaults.
"""

import logging

logger = logging.getLogger(__name__)


class PSIException(Exception):
    """
    Base exception class for all PSI Classifier exceptions.

    All custom exceptions in this module inherit from this base class,
    enabling consistent exception handling and logging.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the PSI exception.

        Args:
            message: Human-readable error message describing the exception.
        """
        self.message = message
        logger.error(f"{self.__class__.__name__}: {message}")
        super().__init__(message)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigError(PSIException):
    """
    Base exception for configuration-related errors.

    Raised when there are issues with environment configuration,
    missing required settings, or invalid configuration values.
    """

    pass


class InvalidConfigError(ConfigError):
    """
    Raised when configuration is missing or invalid.

    This exception is raised during application startup when required
    environment variables are not set or have invalid values.

    Examples:
        - Missing LLM_PROVIDER environment variable
        - Invalid provider name (not azure/openrouter/gemini)
        - Missing provider-specific credentials
    """

    def __init__(self, message: str, missing_key: str | None = None) -> None:
        """
        Initialize the invalid configuration exception.

        Args:
            message: Human-readable error message.
            missing_key: Optional key name that is missing or invalid.
        """
        self.missing_key = missing_key
        super().__init__(message)


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(PSIException):
    """
    Base exception for validation-related errors.

    Raised when input data fails validation checks, such as
    malformed CSV files or invalid state transitions.
    """

    pass


class CSVValidationError(ValidationError):
    """
    Raised when CSV file validation fails.

    This exception is raised when the input CSV file is malformed,
    missing required columns, or contains invalid data.

    Examples:
        - Missing required columns (date, news, ticker, region, exchange)
        - Empty CSV file
        - Malformed CSV syntax
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        missing_columns: list[str] | None = None,
    ) -> None:
        """
        Initialize the CSV validation exception.

        Args:
            message: Human-readable error message.
            file_path: Path to the CSV file that failed validation.
            missing_columns: List of column names that are missing.
        """
        self.file_path = file_path
        self.missing_columns = missing_columns or []
        super().__init__(message)


class StateValidationError(ValidationError):
    """
    Raised when state validation fails during graph execution.

    This exception is raised when the LangGraph state transitions
    to an invalid state or contains inconsistent data.

    Examples:
        - current_index out of bounds
        - Missing required state fields
        - Inconsistent state after node execution
    """

    def __init__(
        self,
        message: str,
        current_state: dict | None = None,
        expected_fields: list[str] | None = None,
    ) -> None:
        """
        Initialize the state validation exception.

        Args:
            message: Human-readable error message.
            current_state: Dictionary representation of current state (sanitized).
            expected_fields: List of expected fields that are missing or invalid.
        """
        self.current_state = current_state
        self.expected_fields = expected_fields or []
        super().__init__(message)


# =============================================================================
# Processing Errors
# =============================================================================


class ProcessingError(PSIException):
    """
    Base exception for processing-related errors.

    Raised when errors occur during the classification pipeline,
    such as LLM failures or unknown regions.
    """

    pass


class UnknownRegionError(ProcessingError):
    """
    Raised when region/exchange combination is not in the registry.

    This exception is raised during framework resolution when the
    provided region and exchange combination does not map to any
    known regulatory framework. Follows fail-fast principle.

    Examples:
        - region="Antarctica", exchange="ICE"
        - Unsupported region/exchange combinations
    """

    def __init__(
        self,
        message: str,
        region: str | None = None,
        exchange: str | None = None,
        supported_combinations: list[str] | None = None,
    ) -> None:
        """
        Initialize the unknown region exception.

        Args:
            message: Human-readable error message.
            region: The region that was not found.
            exchange: The exchange that was not found.
            supported_combinations: List of supported region/exchange combinations.
        """
        self.region = region
        self.exchange = exchange
        self.supported_combinations = supported_combinations or []
        super().__init__(message)


class LLMProviderError(ProcessingError):
    """
    Raised when LLM provider API call fails.

    This exception is raised when the configured LLM provider
    returns an error or the API call fails for any reason.
    No fallback to other providers is attempted.

    Examples:
        - API authentication failure
        - Rate limit exceeded
        - Network timeout
        - Provider service unavailable
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """
        Initialize the LLM provider exception.

        Args:
            message: Human-readable error message.
            provider: Name of the LLM provider (azure/openrouter/gemini).
            status_code: HTTP status code if applicable.
            original_error: The original exception that caused this error.
        """
        self.provider = provider
        self.status_code = status_code
        self.original_error = original_error
        super().__init__(message)


class LLMResponseParseError(ProcessingError):
    """
    Raised when LLM response cannot be parsed.

    This exception is raised when the LLM returns a response
    that cannot be parsed as valid JSON or is missing required
    fields (is_psi, reasoning).

    Examples:
        - Response is not valid JSON
        - Missing 'is_psi' field
        - Missing 'reasoning' field
        - 'is_psi' is not a boolean
    """

    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
        missing_fields: list[str] | None = None,
    ) -> None:
        """
        Initialize the LLM response parse exception.

        Args:
            message: Human-readable error message.
            raw_response: The raw response string that failed to parse (truncated).
            missing_fields: List of fields that are missing from the response.
        """
        # Truncate raw response to prevent logging massive strings
        if raw_response and len(raw_response) > 500:
            raw_response = raw_response[:500] + "..."
        self.raw_response = raw_response
        self.missing_fields = missing_fields or []
        super().__init__(message)
