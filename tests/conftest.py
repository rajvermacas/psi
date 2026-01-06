"""
Pytest configuration and fixtures for PSI Classifier tests.
"""

import os
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Clean environment variables before each test.

    This fixture automatically runs for all tests and removes
    PSI-related environment variables to ensure test isolation.
    """
    # List of environment variables used by PSI Classifier
    env_vars = [
        "LLM_PROVIDER",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_MODEL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "LOG_LEVEL",
    ]

    # Remove all PSI-related environment variables
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    # Clear config cache before test
    from psi.config import get_config

    get_config.cache_clear()

    yield

    # Clear config cache after test
    get_config.cache_clear()


@pytest.fixture
def azure_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Configure environment for Azure OpenAI provider.
    """
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-deployment")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4")


@pytest.fixture
def openrouter_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Configure environment for OpenRouter provider.
    """
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3-sonnet")


@pytest.fixture
def gemini_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Configure environment for Gemini provider.
    """
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")


@pytest.fixture
def sample_news_item() -> dict:
    """
    Return a sample news item for testing.
    """
    return {
        "date": "2024-01-15",
        "news": "Company X announces Q4 earnings beat expectations",
        "ticker": "COMPX",
        "region": "India",
        "exchange": "NSE",
    }


@pytest.fixture
def sample_csv_content() -> str:
    """
    Return sample CSV content for testing.
    """
    return """date,news,ticker,region,exchange
2024-01-15,"Company X announces Q4 earnings beat",COMP,India,NSE
2024-01-16,"CEO attends industry conference",COMP,US,NYSE
2024-01-17,"Company Y acquires Z for $1B",COMPY,UK,LSE"""


@pytest.fixture
def tmp_csv_file(tmp_path, sample_csv_content: str):
    """
    Create a temporary CSV file for testing.
    """
    csv_file = tmp_path / "test_news.csv"
    csv_file.write_text(sample_csv_content)
    return csv_file
