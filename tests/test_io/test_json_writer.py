"""
Tests for JSON writer module.
"""

import json
from datetime import datetime

import pytest

from psi.graph.state import ClassificationResult, NewsItem
from psi.io import write_results


class TestWriteResults:
    """Tests for write_results function."""

    def test_write_results_creates_file(self, tmp_path) -> None:
        """Test that write_results creates output file."""
        news_item = NewsItem(
            date="2024-01-15",
            news="Test news",
            ticker="COMP",
            region="India",
            exchange="NSE",
        )

        result = ClassificationResult(
            news_item=news_item,
            is_psi=True,
            reasoning="Test reasoning",
            regulatory_framework="SEBI",
            classified_at=datetime.now(),
        )

        output_path = tmp_path / "output.json"
        write_results(
            results=[result],
            output_path=output_path,
            model_name="gpt-4",
            provider_name="azure",
        )

        assert output_path.exists()

    def test_write_results_content(self, tmp_path) -> None:
        """Test output JSON content structure."""
        news_item = NewsItem(
            date="2024-01-15",
            news="Company announces earnings",
            ticker="COMP",
            region="India",
            exchange="NSE",
        )

        result = ClassificationResult(
            news_item=news_item,
            is_psi=True,
            reasoning="This is PSI because of earnings announcement",
            regulatory_framework="SEBI",
            classified_at=datetime(2024, 1, 15, 10, 30, 0),
        )

        output_path = tmp_path / "output.json"
        write_results(
            results=[result],
            output_path=output_path,
            model_name="gpt-4-turbo",
            provider_name="azure",
        )

        with open(output_path) as f:
            data = json.load(f)

        # Check metadata
        assert "metadata" in data
        assert data["metadata"]["total_items"] == 1
        assert data["metadata"]["psi_count"] == 1
        assert data["metadata"]["non_psi_count"] == 0
        assert data["metadata"]["model_used"] == "gpt-4-turbo"
        assert data["metadata"]["provider_used"] == "azure"

        # Check results
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["is_psi"] is True
        assert data["results"][0]["regulatory_framework"] == "SEBI"
        assert "news_item" in data["results"][0]

    def test_write_results_multiple_items(self, tmp_path) -> None:
        """Test writing multiple results."""
        items = [
            ClassificationResult(
                news_item=NewsItem(
                    date="2024-01-15",
                    news="Earnings announcement",
                    ticker="A",
                    region="India",
                    exchange="NSE",
                ),
                is_psi=True,
                reasoning="PSI",
                regulatory_framework="SEBI",
            ),
            ClassificationResult(
                news_item=NewsItem(
                    date="2024-01-16",
                    news="CEO attends conference",
                    ticker="B",
                    region="US",
                    exchange="NYSE",
                ),
                is_psi=False,
                reasoning="Not PSI",
                regulatory_framework="SEC",
            ),
            ClassificationResult(
                news_item=NewsItem(
                    date="2024-01-17",
                    news="Dividend announcement",
                    ticker="C",
                    region="UK",
                    exchange="LSE",
                ),
                is_psi=True,
                reasoning="PSI",
                regulatory_framework="FCA",
            ),
        ]

        output_path = tmp_path / "output.json"
        write_results(
            results=items,
            output_path=output_path,
            model_name="test-model",
            provider_name="test-provider",
        )

        with open(output_path) as f:
            data = json.load(f)

        assert data["metadata"]["total_items"] == 3
        assert data["metadata"]["psi_count"] == 2
        assert data["metadata"]["non_psi_count"] == 1

    def test_write_results_creates_directory(self, tmp_path) -> None:
        """Test that write_results creates parent directories."""
        news_item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="COMP",
            region="India",
            exchange="NSE",
        )

        result = ClassificationResult(
            news_item=news_item,
            is_psi=False,
            reasoning="Not PSI",
            regulatory_framework="SEBI",
        )

        # Nested path that doesn't exist
        output_path = tmp_path / "subdir" / "nested" / "output.json"
        write_results(
            results=[result],
            output_path=output_path,
            model_name="model",
            provider_name="provider",
        )

        assert output_path.exists()

    def test_write_results_empty_list(self, tmp_path) -> None:
        """Test writing empty results list."""
        output_path = tmp_path / "output.json"
        write_results(
            results=[],
            output_path=output_path,
            model_name="model",
            provider_name="provider",
        )

        with open(output_path) as f:
            data = json.load(f)

        assert data["metadata"]["total_items"] == 0
        assert data["metadata"]["psi_count"] == 0
        assert data["metadata"]["non_psi_count"] == 0
        assert data["results"] == []

    def test_write_results_returns_path(self, tmp_path) -> None:
        """Test that write_results returns the output path."""
        output_path = tmp_path / "output.json"
        result_path = write_results(
            results=[],
            output_path=output_path,
            model_name="model",
            provider_name="provider",
        )

        assert result_path == output_path
