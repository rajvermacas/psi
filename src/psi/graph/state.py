"""
LangGraph state models for PSI classification pipeline.

Defines the state schema used across all nodes in the graph.
"""

import logging
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


class NewsItem(BaseModel):
    """
    Represents a single news item from the input CSV.

    Attributes:
        date: Date of the news.
        news: The news content/headline.
        ticker: Stock ticker symbol.
        region: Geographic region (e.g., 'India', 'US').
        exchange: Stock exchange (e.g., 'NSE', 'NYSE').
    """

    date: str = Field(..., description="Date of the news")
    news: str = Field(..., description="News content/headline")
    ticker: str = Field(..., description="Stock ticker symbol")
    region: str = Field(..., description="Geographic region")
    exchange: str = Field(..., description="Stock exchange name")


class ClassificationResult(BaseModel):
    """
    Represents the classification result for a single news item.

    Attributes:
        news_item: The original news item.
        is_psi: Whether the news is classified as PSI.
        reasoning: LLM's explanation for the classification.
        regulatory_framework: Name of the regulatory framework used.
        classified_at: Timestamp when classification was performed.
    """

    news_item: NewsItem = Field(..., description="Original news item")
    is_psi: bool = Field(..., description="Whether news is PSI")
    reasoning: str = Field(..., description="Classification reasoning")
    regulatory_framework: str = Field(..., description="Framework used for classification")
    classified_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of classification",
    )


class GraphState(BaseModel):
    """
    The state that flows through the LangGraph pipeline.

    This state is passed between nodes and accumulates information
    as the pipeline processes each news item.

    Attributes:
        news_items: List of all news items to process.
        current_index: Index of the current news item being processed.
        current_item: The current news item being classified.
        framework_name: Name of the resolved regulatory framework.
        framework_context: Full regulatory context for LLM prompt.
        results: List of completed classification results.
        error: Error message if processing failed.
    """

    # Input data
    news_items: list[NewsItem] = Field(
        default_factory=list,
        description="All news items to process",
    )

    # Processing state
    current_index: int = Field(
        default=0,
        description="Index of current news item",
    )
    current_item: NewsItem | None = Field(
        default=None,
        description="Current news item being processed",
    )

    # Framework context
    framework_name: str | None = Field(
        default=None,
        description="Resolved regulatory framework name",
    )
    framework_context: str | None = Field(
        default=None,
        description="Full regulatory context for LLM",
    )

    # Results
    results: list[ClassificationResult] = Field(
        default_factory=list,
        description="Completed classification results",
    )

    # Error handling
    error: str | None = Field(
        default=None,
        description="Error message if processing failed",
    )

    model_config = {"arbitrary_types_allowed": True}

    def is_complete(self) -> bool:
        """
        Check if all news items have been processed.

        Returns:
            True if current_index >= len(news_items).
        """
        return self.current_index >= len(self.news_items)

    def has_items(self) -> bool:
        """
        Check if there are news items to process.

        Returns:
            True if news_items is not empty.
        """
        return len(self.news_items) > 0

    def items_remaining(self) -> int:
        """
        Get count of remaining items to process.

        Returns:
            Number of items remaining.
        """
        return max(0, len(self.news_items) - self.current_index)


class OutputMetadata(BaseModel):
    """
    Metadata for the output JSON file.

    Attributes:
        generated_at: Timestamp when output was generated.
        total_items: Total number of news items processed.
        psi_count: Number of items classified as PSI.
        non_psi_count: Number of items classified as non-PSI.
        model_used: LLM model used for classification.
        provider_used: LLM provider used.
    """

    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Output generation timestamp",
    )
    total_items: int = Field(..., description="Total items processed")
    psi_count: int = Field(..., description="Count of PSI items")
    non_psi_count: int = Field(..., description="Count of non-PSI items")
    model_used: str = Field(..., description="LLM model identifier")
    provider_used: str = Field(..., description="LLM provider name")


class OutputReport(BaseModel):
    """
    Complete output report structure.

    Attributes:
        metadata: Report metadata.
        results: List of classification results.
    """

    metadata: OutputMetadata = Field(..., description="Report metadata")
    results: list[ClassificationResult] = Field(
        ..., description="Classification results"
    )
