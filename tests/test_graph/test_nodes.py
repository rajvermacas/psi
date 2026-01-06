"""
Tests for graph node implementations.
"""

from unittest.mock import MagicMock

import pytest

from psi.exceptions import ProcessingError, UnknownRegionError
from psi.graph.nodes import (
    create_classify_node,
    load_node,
    output_node,
    resolve_node,
    should_continue,
)
from psi.graph.state import ClassificationResult, GraphState, NewsItem
from psi.llm import PSIClassification
from psi.regulatory import reset_registry


@pytest.fixture(autouse=True)
def reset_registry_fixture():
    """Reset registry before each test."""
    reset_registry()
    yield
    reset_registry()


class TestLoadNode:
    """Tests for load_node."""

    def test_load_first_item(self) -> None:
        """Test loading the first item."""
        items = [
            NewsItem(
                date="2024-01-15",
                news="Test news 1",
                ticker="A",
                region="India",
                exchange="NSE",
            ),
            NewsItem(
                date="2024-01-16",
                news="Test news 2",
                ticker="B",
                region="US",
                exchange="NYSE",
            ),
        ]

        state = GraphState(news_items=items, current_index=0)
        result = load_node(state)

        assert result["current_item"] == items[0]
        assert result["current_item"].ticker == "A"

    def test_load_second_item(self) -> None:
        """Test loading the second item."""
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

        state = GraphState(news_items=items, current_index=1)
        result = load_node(state)

        assert result["current_item"] == items[1]
        assert result["current_item"].ticker == "B"

    def test_load_when_complete(self) -> None:
        """Test loading when all items processed."""
        items = [
            NewsItem(
                date="2024-01-15",
                news="Test",
                ticker="A",
                region="India",
                exchange="NSE",
            ),
        ]

        state = GraphState(news_items=items, current_index=1)
        result = load_node(state)

        assert result["current_item"] is None


class TestResolveNode:
    """Tests for resolve_node."""

    def test_resolve_sebi_framework(self) -> None:
        """Test resolving SEBI framework for India."""
        item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="A",
            region="India",
            exchange="NSE",
        )

        state = GraphState(current_item=item)
        result = resolve_node(state)

        assert result["framework_name"] == "SEBI"
        assert "SEBI" in result["framework_context"]

    def test_resolve_sec_framework(self) -> None:
        """Test resolving SEC framework for US."""
        item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="A",
            region="US",
            exchange="NYSE",
        )

        state = GraphState(current_item=item)
        result = resolve_node(state)

        assert result["framework_name"] == "SEC"

    def test_resolve_unknown_region(self) -> None:
        """Test resolving unknown region raises error."""
        item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="A",
            region="Antarctica",
            exchange="ICE",
        )

        state = GraphState(current_item=item)

        with pytest.raises(UnknownRegionError):
            resolve_node(state)

    def test_resolve_no_current_item(self) -> None:
        """Test resolving without current item raises error."""
        state = GraphState(current_item=None)

        with pytest.raises(ProcessingError):
            resolve_node(state)


class TestClassifyNode:
    """Tests for classify node."""

    def test_classify_success(self) -> None:
        """Test successful classification."""
        # Create mock LLM provider
        mock_provider = MagicMock()
        mock_provider.classify.return_value = PSIClassification(
            is_psi=True,
            reasoning="Test reasoning",
        )

        classify_node = create_classify_node(mock_provider)

        item = NewsItem(
            date="2024-01-15",
            news="Company announces earnings",
            ticker="A",
            region="India",
            exchange="NSE",
        )

        state = GraphState(
            current_item=item,
            framework_name="SEBI",
            framework_context="SEBI PSI criteria...",
        )

        result = classify_node(state)

        assert len(result["results"]) == 1
        assert result["results"][0].is_psi is True
        assert result["results"][0].reasoning == "Test reasoning"
        assert result["results"][0].regulatory_framework == "SEBI"

    def test_classify_appends_to_existing_results(self) -> None:
        """Test that classify appends to existing results."""
        mock_provider = MagicMock()
        mock_provider.classify.return_value = PSIClassification(
            is_psi=False,
            reasoning="Not PSI",
        )

        classify_node = create_classify_node(mock_provider)

        # Existing result
        existing_item = NewsItem(
            date="2024-01-14",
            news="Previous",
            ticker="X",
            region="India",
            exchange="NSE",
        )
        existing_result = ClassificationResult(
            news_item=existing_item,
            is_psi=True,
            reasoning="Was PSI",
            regulatory_framework="SEBI",
        )

        # Current item
        current_item = NewsItem(
            date="2024-01-15",
            news="Current",
            ticker="Y",
            region="India",
            exchange="NSE",
        )

        state = GraphState(
            current_item=current_item,
            framework_name="SEBI",
            framework_context="SEBI PSI criteria...",
            results=[existing_result],
        )

        result = classify_node(state)

        assert len(result["results"]) == 2
        assert result["results"][0].is_psi is True  # Existing
        assert result["results"][1].is_psi is False  # New

    def test_classify_no_current_item(self) -> None:
        """Test classify without current item raises error."""
        mock_provider = MagicMock()
        classify_node = create_classify_node(mock_provider)

        state = GraphState(
            current_item=None,
            framework_context="SEBI PSI criteria...",
        )

        with pytest.raises(ProcessingError):
            classify_node(state)

    def test_classify_no_framework_context(self) -> None:
        """Test classify without framework context raises error."""
        mock_provider = MagicMock()
        classify_node = create_classify_node(mock_provider)

        item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="A",
            region="India",
            exchange="NSE",
        )

        state = GraphState(
            current_item=item,
            framework_context=None,
        )

        with pytest.raises(ProcessingError):
            classify_node(state)


class TestOutputNode:
    """Tests for output_node."""

    def test_output_increments_index(self) -> None:
        """Test that output node increments current_index."""
        items = [
            NewsItem(
                date="2024-01-15",
                news="Test",
                ticker="A",
                region="India",
                exchange="NSE",
            ),
        ]

        result_item = ClassificationResult(
            news_item=items[0],
            is_psi=True,
            reasoning="PSI",
            regulatory_framework="SEBI",
        )

        state = GraphState(
            news_items=items,
            current_index=0,
            current_item=items[0],
            results=[result_item],
        )

        result = output_node(state)

        assert result["current_index"] == 1

    def test_output_clears_transient_state(self) -> None:
        """Test that output node clears transient state."""
        item = NewsItem(
            date="2024-01-15",
            news="Test",
            ticker="A",
            region="India",
            exchange="NSE",
        )

        state = GraphState(
            news_items=[item],
            current_index=0,
            current_item=item,
            framework_name="SEBI",
            framework_context="Context",
            results=[
                ClassificationResult(
                    news_item=item,
                    is_psi=True,
                    reasoning="PSI",
                    regulatory_framework="SEBI",
                )
            ],
        )

        result = output_node(state)

        assert result["current_item"] is None
        assert result["framework_name"] is None
        assert result["framework_context"] is None


class TestShouldContinue:
    """Tests for should_continue routing function."""

    def test_continue_when_items_remaining(self) -> None:
        """Test returns 'continue' when items remain."""
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

        state = GraphState(news_items=items, current_index=1)

        result = should_continue(state)

        assert result == "continue"

    def test_end_when_all_processed(self) -> None:
        """Test returns 'end' when all items processed."""
        items = [
            NewsItem(
                date="2024-01-15",
                news="Test",
                ticker="A",
                region="India",
                exchange="NSE",
            ),
        ]

        state = GraphState(news_items=items, current_index=1)

        result = should_continue(state)

        assert result == "end"

    def test_end_when_empty(self) -> None:
        """Test returns 'end' for empty state."""
        state = GraphState(news_items=[], current_index=0)

        result = should_continue(state)

        assert result == "end"
