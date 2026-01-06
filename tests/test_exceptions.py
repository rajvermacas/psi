"""
Tests for PSI Classifier exception classes.

Tests cover all custom exception types and their attributes.
"""

import pytest

from psi.exceptions import (
    ConfigError,
    CSVValidationError,
    InvalidConfigError,
    LLMProviderError,
    LLMResponseParseError,
    ProcessingError,
    PSIException,
    StateValidationError,
    UnknownRegionError,
    ValidationError,
)


class TestPSIException:
    """Tests for the base PSIException class."""

    def test_exception_creation(self) -> None:
        """Test basic exception creation."""
        exc = PSIException("Test error message")
        assert str(exc) == "Test error message"
        assert exc.message == "Test error message"

    def test_exception_inheritance(self) -> None:
        """Test that PSIException inherits from Exception."""
        exc = PSIException("Test")
        assert isinstance(exc, Exception)

    def test_exception_can_be_raised(self) -> None:
        """Test that exception can be raised and caught."""
        with pytest.raises(PSIException) as exc_info:
            raise PSIException("Test error")
        assert "Test error" in str(exc_info.value)


class TestConfigError:
    """Tests for ConfigError and its subclasses."""

    def test_config_error_inheritance(self) -> None:
        """Test ConfigError inherits from PSIException."""
        exc = ConfigError("Config error")
        assert isinstance(exc, PSIException)
        assert isinstance(exc, ConfigError)

    def test_invalid_config_error_basic(self) -> None:
        """Test InvalidConfigError with basic message."""
        exc = InvalidConfigError("Missing config")
        assert str(exc) == "Missing config"
        assert exc.missing_key is None

    def test_invalid_config_error_with_key(self) -> None:
        """Test InvalidConfigError with missing key."""
        exc = InvalidConfigError(
            message="LLM_PROVIDER is required",
            missing_key="LLM_PROVIDER",
        )
        assert exc.missing_key == "LLM_PROVIDER"
        assert "LLM_PROVIDER" in str(exc)

    def test_invalid_config_error_inheritance(self) -> None:
        """Test InvalidConfigError inheritance chain."""
        exc = InvalidConfigError("Test")
        assert isinstance(exc, ConfigError)
        assert isinstance(exc, PSIException)


class TestValidationError:
    """Tests for ValidationError and its subclasses."""

    def test_validation_error_inheritance(self) -> None:
        """Test ValidationError inherits from PSIException."""
        exc = ValidationError("Validation error")
        assert isinstance(exc, PSIException)

    def test_csv_validation_error_basic(self) -> None:
        """Test CSVValidationError with basic message."""
        exc = CSVValidationError("Invalid CSV")
        assert str(exc) == "Invalid CSV"
        assert exc.file_path is None
        assert exc.missing_columns == []

    def test_csv_validation_error_with_details(self) -> None:
        """Test CSVValidationError with file path and missing columns."""
        exc = CSVValidationError(
            message="Missing required columns",
            file_path="/path/to/file.csv",
            missing_columns=["date", "ticker"],
        )
        assert exc.file_path == "/path/to/file.csv"
        assert exc.missing_columns == ["date", "ticker"]

    def test_csv_validation_error_inheritance(self) -> None:
        """Test CSVValidationError inheritance chain."""
        exc = CSVValidationError("Test")
        assert isinstance(exc, ValidationError)
        assert isinstance(exc, PSIException)

    def test_state_validation_error_basic(self) -> None:
        """Test StateValidationError with basic message."""
        exc = StateValidationError("Invalid state")
        assert str(exc) == "Invalid state"
        assert exc.current_state is None
        assert exc.expected_fields == []

    def test_state_validation_error_with_details(self) -> None:
        """Test StateValidationError with state and expected fields."""
        exc = StateValidationError(
            message="Invalid state transition",
            current_state={"current_index": 5},
            expected_fields=["current_item", "framework_criteria"],
        )
        assert exc.current_state == {"current_index": 5}
        assert exc.expected_fields == ["current_item", "framework_criteria"]

    def test_state_validation_error_inheritance(self) -> None:
        """Test StateValidationError inheritance chain."""
        exc = StateValidationError("Test")
        assert isinstance(exc, ValidationError)
        assert isinstance(exc, PSIException)


class TestProcessingError:
    """Tests for ProcessingError and its subclasses."""

    def test_processing_error_inheritance(self) -> None:
        """Test ProcessingError inherits from PSIException."""
        exc = ProcessingError("Processing error")
        assert isinstance(exc, PSIException)

    def test_unknown_region_error_basic(self) -> None:
        """Test UnknownRegionError with basic message."""
        exc = UnknownRegionError("Unknown region")
        assert str(exc) == "Unknown region"
        assert exc.region is None
        assert exc.exchange is None
        assert exc.supported_combinations == []

    def test_unknown_region_error_with_details(self) -> None:
        """Test UnknownRegionError with region and exchange details."""
        exc = UnknownRegionError(
            message="No framework found",
            region="Antarctica",
            exchange="ICE",
            supported_combinations=["India/NSE", "US/NYSE"],
        )
        assert exc.region == "Antarctica"
        assert exc.exchange == "ICE"
        assert exc.supported_combinations == ["India/NSE", "US/NYSE"]

    def test_unknown_region_error_inheritance(self) -> None:
        """Test UnknownRegionError inheritance chain."""
        exc = UnknownRegionError("Test")
        assert isinstance(exc, ProcessingError)
        assert isinstance(exc, PSIException)

    def test_llm_provider_error_basic(self) -> None:
        """Test LLMProviderError with basic message."""
        exc = LLMProviderError("LLM failed")
        assert str(exc) == "LLM failed"
        assert exc.provider is None
        assert exc.status_code is None
        assert exc.original_error is None

    def test_llm_provider_error_with_details(self) -> None:
        """Test LLMProviderError with provider and status code."""
        original = ValueError("Original error")
        exc = LLMProviderError(
            message="API call failed",
            provider="azure",
            status_code=401,
            original_error=original,
        )
        assert exc.provider == "azure"
        assert exc.status_code == 401
        assert exc.original_error is original

    def test_llm_provider_error_inheritance(self) -> None:
        """Test LLMProviderError inheritance chain."""
        exc = LLMProviderError("Test")
        assert isinstance(exc, ProcessingError)
        assert isinstance(exc, PSIException)

    def test_llm_response_parse_error_basic(self) -> None:
        """Test LLMResponseParseError with basic message."""
        exc = LLMResponseParseError("Parse failed")
        assert str(exc) == "Parse failed"
        assert exc.raw_response is None
        assert exc.missing_fields == []

    def test_llm_response_parse_error_with_details(self) -> None:
        """Test LLMResponseParseError with raw response and missing fields."""
        exc = LLMResponseParseError(
            message="Invalid JSON response",
            raw_response='{"is_psi": true}',
            missing_fields=["reasoning"],
        )
        assert exc.raw_response == '{"is_psi": true}'
        assert exc.missing_fields == ["reasoning"]

    def test_llm_response_parse_error_truncates_long_response(self) -> None:
        """Test that long raw responses are truncated."""
        long_response = "x" * 1000
        exc = LLMResponseParseError(
            message="Parse failed",
            raw_response=long_response,
        )
        # Should be truncated to 500 chars + "..."
        assert len(exc.raw_response) == 503
        assert exc.raw_response.endswith("...")

    def test_llm_response_parse_error_does_not_truncate_short_response(self) -> None:
        """Test that short raw responses are not truncated."""
        short_response = "x" * 100
        exc = LLMResponseParseError(
            message="Parse failed",
            raw_response=short_response,
        )
        assert exc.raw_response == short_response
        assert not exc.raw_response.endswith("...")

    def test_llm_response_parse_error_inheritance(self) -> None:
        """Test LLMResponseParseError inheritance chain."""
        exc = LLMResponseParseError("Test")
        assert isinstance(exc, ProcessingError)
        assert isinstance(exc, PSIException)


class TestExceptionCatching:
    """Tests for catching exceptions at different levels of hierarchy."""

    def test_catch_by_base_class(self) -> None:
        """Test catching specific exceptions by base class."""
        # InvalidConfigError can be caught as ConfigError or PSIException
        with pytest.raises(PSIException):
            raise InvalidConfigError("Test")

        with pytest.raises(ConfigError):
            raise InvalidConfigError("Test")

    def test_catch_csv_error_by_validation(self) -> None:
        """Test catching CSVValidationError as ValidationError."""
        with pytest.raises(ValidationError):
            raise CSVValidationError("Test")

    def test_catch_llm_error_by_processing(self) -> None:
        """Test catching LLM errors as ProcessingError."""
        with pytest.raises(ProcessingError):
            raise LLMProviderError("Test")

        with pytest.raises(ProcessingError):
            raise LLMResponseParseError("Test")

        with pytest.raises(ProcessingError):
            raise UnknownRegionError("Test")
