"""
LangGraph node implementations for PSI classification pipeline.

Implements the 4-node pipeline:
1. load_node: Load next news item from list
2. resolve_node: Resolve regulatory framework
3. classify_node: Classify news using LLM
4. output_node: Store result and advance
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from psi.exceptions import ProcessingError, UnknownRegionError
from psi.graph.state import ClassificationResult, GraphState, NewsItem
from psi.llm.base import BaseLLMProvider
from psi.regulatory.registry import get_registry

if TYPE_CHECKING:
    from psi.regulatory.base import RegulatoryFramework

logger = logging.getLogger(__name__)


def load_node(state: GraphState) -> dict:
    """
    Load the next news item from the list.

    This node retrieves the current news item based on current_index
    and prepares it for processing.

    Args:
        state: Current graph state.

    Returns:
        Dictionary with current_item set.
    """
    logger.info(f"LOAD NODE: Processing item {state.current_index + 1}/{len(state.news_items)}")

    if state.is_complete():
        logger.info("LOAD NODE: All items processed")
        return {"current_item": None}

    current_item = state.news_items[state.current_index]

    logger.info(
        f"LOAD NODE: Loaded item - ticker={current_item.ticker}, "
        f"region={current_item.region}, exchange={current_item.exchange}"
    )
    logger.debug(f"LOAD NODE: News content: {current_item.news[:100]}...")

    return {"current_item": current_item}


def resolve_node(state: GraphState) -> dict:
    """
    Resolve the regulatory framework for the current news item.

    This node looks up the appropriate regulatory framework based on
    the region and exchange of the current news item.

    Args:
        state: Current graph state with current_item set.

    Returns:
        Dictionary with framework_name and framework_context set.

    Raises:
        UnknownRegionError: If no framework is found for region/exchange.
    """
    logger.info("RESOLVE NODE: Resolving regulatory framework")

    if state.current_item is None:
        logger.error("RESOLVE NODE: No current item to process")
        raise ProcessingError(
            message="Cannot resolve framework: no current item in state"
        )

    item = state.current_item
    registry = get_registry()

    logger.debug(
        f"RESOLVE NODE: Looking up framework for region={item.region}, "
        f"exchange={item.exchange}"
    )

    # This will raise UnknownRegionError if not found (fail-fast)
    framework = registry.get_framework(item.region, item.exchange)

    framework_context = framework.get_prompt_context()

    logger.info(f"RESOLVE NODE: Resolved framework {framework.name} ({framework.jurisdiction})")
    logger.debug(f"RESOLVE NODE: Context length: {len(framework_context)} chars")

    return {
        "framework_name": framework.name,
        "framework_context": framework_context,
    }


def create_classify_node(llm_provider: BaseLLMProvider):
    """
    Create a classify node with the given LLM provider.

    This factory function creates a node that uses the provided LLM
    provider to classify news items.

    Args:
        llm_provider: The LLM provider to use for classification.

    Returns:
        Node function that classifies news using the LLM.
    """

    def classify_node(state: GraphState) -> dict:
        """
        Classify the current news item using the LLM.

        This node sends the news content along with the regulatory
        framework context to the LLM for classification.

        Args:
            state: Current graph state with current_item and framework_context.

        Returns:
            Dictionary with classification result added to results.

        Raises:
            LLMProviderError: If LLM API call fails.
            LLMResponseParseError: If LLM response cannot be parsed.
        """
        logger.info(f"CLASSIFY NODE: Classifying with {llm_provider.provider_name}")

        if state.current_item is None:
            logger.error("CLASSIFY NODE: No current item to classify")
            raise ProcessingError(
                message="Cannot classify: no current item in state"
            )

        if state.framework_context is None:
            logger.error("CLASSIFY NODE: No framework context available")
            raise ProcessingError(
                message="Cannot classify: no framework context in state"
            )

        item = state.current_item

        logger.debug(
            f"CLASSIFY NODE: Processing ticker={item.ticker}, "
            f"framework={state.framework_name}"
        )

        # Call LLM for classification
        classification = llm_provider.classify(
            news=item.news,
            framework_context=state.framework_context,
        )

        # Create result
        result = ClassificationResult(
            news_item=item,
            is_psi=classification.is_psi,
            reasoning=classification.reasoning,
            regulatory_framework=state.framework_name or "Unknown",
            classified_at=datetime.now(),
        )

        logger.info(
            f"CLASSIFY NODE: Result - is_psi={result.is_psi}, "
            f"framework={result.regulatory_framework}"
        )
        logger.debug(f"CLASSIFY NODE: Reasoning: {result.reasoning[:100]}...")

        # Return updated results list
        updated_results = state.results.copy()
        updated_results.append(result)

        return {"results": updated_results}

    return classify_node


def output_node(state: GraphState) -> dict:
    """
    Output the current result and advance to the next item.

    This node prints the classification result to console and
    advances the current_index for the next iteration.

    Args:
        state: Current graph state with latest result added.

    Returns:
        Dictionary with incremented current_index.
    """
    logger.info("OUTPUT NODE: Outputting result and advancing")

    # Get the latest result
    if state.results:
        latest_result = state.results[-1]

        # Console output
        psi_indicator = "✓ PSI" if latest_result.is_psi else "✗ NOT PSI"

        print(
            f"\n[{state.current_index + 1}/{len(state.news_items)}] "
            f"{latest_result.news_item.ticker} ({latest_result.regulatory_framework}): "
            f"{psi_indicator}"
        )
        print(f"  News: {latest_result.news_item.news[:80]}...")
        print(f"  Reasoning: {latest_result.reasoning[:100]}...")

        logger.info(
            f"OUTPUT NODE: Completed item {state.current_index + 1}, "
            f"is_psi={latest_result.is_psi}"
        )
    else:
        logger.warning("OUTPUT NODE: No results to output")

    # Advance to next item
    new_index = state.current_index + 1

    logger.debug(f"OUTPUT NODE: Advancing index from {state.current_index} to {new_index}")

    return {
        "current_index": new_index,
        # Clear transient state for next iteration
        "current_item": None,
        "framework_name": None,
        "framework_context": None,
    }


def should_continue(state: GraphState) -> str:
    """
    Routing function to determine if processing should continue.

    This function is used as a conditional edge in the graph to
    determine whether to process another item or end.

    Args:
        state: Current graph state.

    Returns:
        "continue" if more items to process, "end" otherwise.
    """
    # Note: current_index is already incremented by output_node
    remaining = len(state.news_items) - state.current_index

    if remaining > 0:
        logger.debug(f"ROUTER: {remaining} items remaining, continuing")
        return "continue"
    else:
        logger.info("ROUTER: All items processed, ending")
        return "end"
