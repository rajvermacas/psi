"""
Tests for graph state models.
"""

from datetime import datetime

import pytest

from psi.graph.state import (
    ClassificationResult,
    GraphState,
    NewsItem,
    OutputMetadata,
    OutputReport,
)


class TestNewsItem:
    """Tests for NewsItem model."""

    def test_create_news_item(self) -> None:
        """Test creating a NewsItem."""
        item = NewsItem(
            date="2024-01-15",
            news="Company announces earnings",
            ticker="COMP",
            region="India",
            exchange="NSE",
        )

        assert item.date == "2024-01-15"
        assert item.news == "Company announces earnings"
        assert item.ticker == "COMP"
        assert item.region == "India"
        assert item.exchange == "NSE"

    def test_news_item_from_dict(self) -> None:
        """Test creating NewsItem from dictionary."""
        data = {
            "date": "2024-01-15",
            "news": "Test news",
            "ticker": "TEST",
            "region": "US",
            "exchange": "NYSE",
        }

        item = NewsItem(**data)

        assert item.ticker == "TEST"
        assert item.exchange == "NYSE"

    def test_news_item_to_dict(self) -> None:
        """Test converting NewsItem to dictionary."""
        item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="TEST",
            region="US",
            exchange="NYSE",
        )

        data = item.model_dump()

        assert data["date"] == "2024-01-15"
        assert data["ticker"] == "TEST"


class TestClassificationResult:
    """Tests for ClassificationResult model."""

    def test_create_classification_result(self) -> None:
        """Test creating a ClassificationResult."""
        item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="TEST",
            region="India",
            exchange="NSE",
        )

        result = ClassificationResult(
            news_item=item,
            is_psi=True,
            reasoning="This is PSI because...",
            regulatory_framework="SEBI",
        )

        assert result.is_psi is True
        assert result.reasoning == "This is PSI because..."
        assert result.regulatory_framework == "SEBI"
        assert result.news_item.ticker == "TEST"
        assert result.classified_at is not None

    def test_classification_result_custom_timestamp(self) -> None:
        """Test setting custom timestamp."""
        item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="TEST",
            region="India",
            exchange="NSE",
        )

        custom_time = datetime(2024, 1, 15, 10, 30, 0)

        result = ClassificationResult(
            news_item=item,
            is_psi=False,
            reasoning="Not PSI",
            regulatory_framework="SEBI",
            classified_at=custom_time,
        )

        assert result.classified_at == custom_time


class TestGraphState:
    """Tests for GraphState model."""

    def test_create_empty_state(self) -> None:
        """Test creating empty GraphState."""
        state = GraphState()

        assert state.news_items == []
        assert state.current_index == 0
        assert state.current_item is None
        assert state.framework_name is None
        assert state.framework_context is None
        assert state.results == []
        assert state.error is None

    def test_create_state_with_items(self) -> None:
        """Test creating GraphState with news items."""
        items = [
            NewsItem(
                date="2024-01-15",
                news="Test 1",
                ticker="A",
                region="India",
                exchange="NSE",
            ),
            NewsItem(
                date="2024-01-16",
                news="Test 2",
                ticker="B",
                region="US",
                exchange="NYSE",
            ),
        ]

        state = GraphState(news_items=items)

        assert len(state.news_items) == 2
        assert state.news_items[0].ticker == "A"
        assert state.news_items[1].ticker == "B"

    def test_is_complete_empty(self) -> None:
        """Test is_complete with empty state."""
        state = GraphState()

        assert state.is_complete() is True

    def test_is_complete_with_items(self) -> None:
        """Test is_complete with items."""
        items = [
            NewsItem(
                date="2024-01-15",
                news="Test",
                ticker="A",
                region="India",
                exchange="NSE",
            ),
        ]

        state = GraphState(news_items=items, current_index=0)
        assert state.is_complete() is False

        state = GraphState(news_items=items, current_index=1)
        assert state.is_complete() is True

    def test_has_items(self) -> None:
        """Test has_items method."""
        empty_state = GraphState()
        assert empty_state.has_items() is False

        state_with_items = GraphState(
            news_items=[
                NewsItem(
                    date="2024-01-15",
                    news="Test",
                    ticker="A",
                    region="India",
                    exchange="NSE",
                )
            ]
        )
        assert state_with_items.has_items() is True

    def test_items_remaining(self) -> None:
        """Test items_remaining method."""
        items = [
            NewsItem(
                date="2024-01-15",
                news="Test",
                ticker="A",
                region="India",
                exchange="NSE",
            ),
            NewsItem(
                date="2024-01-16",
                news="Test",
                ticker="B",
                region="US",
                exchange="NYSE",
            ),
            NewsItem(
                date="2024-01-17",
                news="Test",
                ticker="C",
                region="UK",
                exchange="LSE",
            ),
        ]

        state = GraphState(news_items=items, current_index=0)
        assert state.items_remaining() == 3

        state = GraphState(news_items=items, current_index=1)
        assert state.items_remaining() == 2

        state = GraphState(news_items=items, current_index=3)
        assert state.items_remaining() == 0

        state = GraphState(news_items=items, current_index=5)
        assert state.items_remaining() == 0


class TestOutputMetadata:
    """Tests for OutputMetadata model."""

    def test_create_metadata(self) -> None:
        """Test creating OutputMetadata."""
        metadata = OutputMetadata(
            total_items=10,
            psi_count=3,
            non_psi_count=7,
            model_used="gpt-4",
            provider_used="azure",
        )

        assert metadata.total_items == 10
        assert metadata.psi_count == 3
        assert metadata.non_psi_count == 7
        assert metadata.model_used == "gpt-4"
        assert metadata.provider_used == "azure"
        assert metadata.generated_at is not None


class TestOutputReport:
    """Tests for OutputReport model."""

    def test_create_report(self) -> None:
        """Test creating OutputReport."""
        metadata = OutputMetadata(
            total_items=1,
            psi_count=1,
            non_psi_count=0,
            model_used="gpt-4",
            provider_used="azure",
        )

        item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="TEST",
            region="India",
            exchange="NSE",
        )

        result = ClassificationResult(
            news_item=item,
            is_psi=True,
            reasoning="PSI",
            regulatory_framework="SEBI",
        )

        report = OutputReport(
            metadata=metadata,
            results=[result],
        )

        assert report.metadata.total_items == 1
        assert len(report.results) == 1
        assert report.results[0].is_psi is True
