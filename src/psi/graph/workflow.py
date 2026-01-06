"""
LangGraph workflow builder for PSI classification pipeline.

Constructs the StateGraph with all nodes and edges.
"""

import logging
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from psi.graph.nodes import (
    create_classify_node,
    load_node,
    output_node,
    resolve_node,
    should_continue,
)
from psi.graph.state import GraphState
from psi.llm.base import BaseLLMProvider

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


def build_workflow(llm_provider: BaseLLMProvider) -> "CompiledStateGraph":
    """
    Build the PSI classification workflow graph.

    Creates a StateGraph with the following structure:

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    PSI CLASSIFICATION PIPELINE                  │
    │                                                                 │
    │  ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌──────────┐  │
    │  │   LOAD   │──▶│  RESOLVE  │──▶│  CLASSIFY  │──▶│  OUTPUT  │  │
    │  │   NODE   │   │   NODE    │   │   NODE     │   │   NODE   │  │
    │  └──────────┘   └───────────┘   └────────────┘   └────┬─────┘  │
    │       ▲                                              │        │
    │       │                        ┌──────────────────────┘        │
    │       │                        ▼                               │
    │       │                  ┌──────────┐                          │
    │       └──────────────────┤  ROUTER  │                          │
    │          (continue)      │ continue │                          │
    │                          │   or     │                          │
    │                          │   end    │────────────▶ END         │
    │                          └──────────┘     (end)                │
    └─────────────────────────────────────────────────────────────────┘
    ```

    Args:
        llm_provider: The LLM provider to use for classification.

    Returns:
        Compiled StateGraph ready for execution.
    """
    logger.info(
        f"Building workflow with LLM provider: {llm_provider.provider_name} "
        f"({llm_provider.model_name})"
    )

    # Create the state graph
    workflow = StateGraph(GraphState)

    # Add nodes
    logger.debug("Adding nodes to workflow")

    workflow.add_node("load", load_node)
    workflow.add_node("resolve", resolve_node)
    workflow.add_node("classify", create_classify_node(llm_provider))
    workflow.add_node("output", output_node)

    # Set entry point
    workflow.set_entry_point("load")

    # Add edges
    logger.debug("Adding edges to workflow")

    workflow.add_edge("load", "resolve")
    workflow.add_edge("resolve", "classify")
    workflow.add_edge("classify", "output")

    # Add conditional edge for looping
    workflow.add_conditional_edges(
        "output",
        should_continue,
        {
            "continue": "load",
            "end": END,
        },
    )

    # Compile the graph
    logger.info("Compiling workflow graph")
    compiled = workflow.compile()

    logger.info("Workflow compiled successfully")

    return compiled


def run_classification(
    llm_provider: BaseLLMProvider,
    news_items: list[dict],
) -> GraphState:
    """
    Run the classification pipeline on a list of news items.

    Convenience function that builds the workflow and executes it
    on the provided news items.

    Args:
        llm_provider: The LLM provider to use.
        news_items: List of news item dictionaries.

    Returns:
        Final GraphState with all classification results.

    Example:
        >>> from psi.graph.state import NewsItem
        >>> items = [
        ...     {"date": "2024-01-15", "news": "...", "ticker": "X", "region": "India", "exchange": "NSE"},
        ... ]
        >>> result = run_classification(provider, items)
        >>> print(result.results)
    """
    from psi.graph.state import NewsItem

    logger.info(f"Running classification on {len(news_items)} items")

    # Convert dicts to NewsItem models
    items = [NewsItem(**item) for item in news_items]

    # Build workflow
    workflow = build_workflow(llm_provider)

    # Create initial state
    initial_state = GraphState(news_items=items)

    logger.debug(f"Initial state: {len(initial_state.news_items)} items to process")

    # Run the workflow
    logger.info("Starting workflow execution")

    final_state = None
    for state in workflow.stream(initial_state):
        # The stream yields intermediate states
        # We only need the final one
        final_state = state

    # Extract final state from the stream result
    if final_state:
        # The stream returns dict with node name as key
        for node_name, node_state in final_state.items():
            if isinstance(node_state, GraphState):
                final_state = node_state
                break
            elif isinstance(node_state, dict):
                # Reconstruct state from dict update
                pass

    # Create final state from accumulated results
    # The workflow modifies state in place, but we need to return
    # a complete state object

    logger.info("Workflow execution completed")
    logger.info(
        f"Processed {len(items)} items, "
        f"generated {len(initial_state.results)} results"
    )

    return initial_state


class PSIClassifier:
    """
    High-level interface for PSI classification.

    Wraps the workflow and provides a simple API for classifying news.
    """

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        """
        Initialize the PSI classifier.

        Args:
            llm_provider: The LLM provider to use.
        """
        self._llm_provider = llm_provider
        self._workflow = build_workflow(llm_provider)

        logger.info(
            f"PSIClassifier initialized with {llm_provider.provider_name} "
            f"({llm_provider.model_name})"
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._llm_provider.provider_name

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._llm_provider.model_name

    def classify(self, news_items: list[dict]) -> list[dict]:
        """
        Classify a list of news items.

        Args:
            news_items: List of news item dictionaries with keys:
                - date: str
                - news: str
                - ticker: str
                - region: str
                - exchange: str

        Returns:
            List of classification result dictionaries.
        """
        from psi.graph.state import NewsItem

        logger.info(f"Classifying {len(news_items)} news items")

        # Convert to NewsItem objects
        items = [NewsItem(**item) for item in news_items]

        # Create initial state
        initial_state = GraphState(news_items=items)

        # Run workflow
        final_state = None
        for state in self._workflow.stream(initial_state):
            final_state = state

        # Get results - they are accumulated in the initial_state object
        # because we're modifying lists in place

        # Actually, due to how LangGraph works with Pydantic models,
        # we need to track the state through the stream

        # Re-run to get proper final state
        result_state = self._workflow.invoke(initial_state)

        if isinstance(result_state, dict):
            results = result_state.get("results", [])
        elif isinstance(result_state, GraphState):
            results = result_state.results
        else:
            results = []

        logger.info(f"Classification complete: {len(results)} results")

        # Convert to dictionaries
        return [
            {
                "news_item": result.news_item.model_dump(),
                "is_psi": result.is_psi,
                "reasoning": result.reasoning,
                "regulatory_framework": result.regulatory_framework,
                "classified_at": result.classified_at.isoformat(),
            }
            for result in results
        ]
