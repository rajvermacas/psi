"""
JSON writer for PSI classification output.

Writes classification results to JSON files.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from psi.graph.state import (
    ClassificationResult,
    GraphState,
    OutputMetadata,
    OutputReport,
)

logger = logging.getLogger(__name__)


def write_results(
    results: list[ClassificationResult],
    output_path: str | Path,
    model_name: str,
    provider_name: str,
) -> Path:
    """
    Write classification results to a JSON file.

    Args:
        results: List of classification results.
        output_path: Path to the output JSON file.
        model_name: Name of the LLM model used.
        provider_name: Name of the LLM provider used.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)

    logger.info(f"Writing results to: {output_path}")

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    psi_count = sum(1 for r in results if r.is_psi)
    non_psi_count = len(results) - psi_count

    # Create metadata
    metadata = OutputMetadata(
        generated_at=datetime.now(),
        total_items=len(results),
        psi_count=psi_count,
        non_psi_count=non_psi_count,
        model_used=model_name,
        provider_used=provider_name,
    )

    # Create report
    report = OutputReport(
        metadata=metadata,
        results=results,
    )

    # Convert to JSON-serializable format
    output_data = {
        "metadata": {
            "generated_at": metadata.generated_at.isoformat(),
            "total_items": metadata.total_items,
            "psi_count": metadata.psi_count,
            "non_psi_count": metadata.non_psi_count,
            "model_used": metadata.model_used,
            "provider_used": metadata.provider_used,
        },
        "results": [
            {
                "news_item": {
                    "date": r.news_item.date,
                    "news": r.news_item.news,
                    "ticker": r.news_item.ticker,
                    "region": r.news_item.region,
                    "exchange": r.news_item.exchange,
                },
                "is_psi": r.is_psi,
                "reasoning": r.reasoning,
                "regulatory_framework": r.regulatory_framework,
                "classified_at": r.classified_at.isoformat(),
            }
            for r in results
        ],
    }

    # Write JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Successfully wrote {len(results)} results to {output_path} "
        f"(PSI: {psi_count}, Non-PSI: {non_psi_count})"
    )

    return output_path


def write_results_from_state(
    state: GraphState,
    output_path: str | Path,
    model_name: str,
    provider_name: str,
) -> Path:
    """
    Write classification results from GraphState to a JSON file.

    Convenience function that extracts results from GraphState.

    Args:
        state: GraphState with classification results.
        output_path: Path to the output JSON file.
        model_name: Name of the LLM model used.
        provider_name: Name of the LLM provider used.

    Returns:
        Path to the written file.
    """
    return write_results(
        results=state.results,
        output_path=output_path,
        model_name=model_name,
        provider_name=provider_name,
    )


def print_summary(results: list[ClassificationResult]) -> None:
    """
    Print a summary of classification results to console.

    Args:
        results: List of classification results.
    """
    psi_count = sum(1 for r in results if r.is_psi)
    non_psi_count = len(results) - psi_count

    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total items processed: {len(results)}")
    print(f"  ✓ PSI (Price Sensitive): {psi_count}")
    print(f"  ✗ Non-PSI: {non_psi_count}")
    print("-" * 60)

    if psi_count > 0:
        print("\nPSI Items:")
        for i, r in enumerate(results, 1):
            if r.is_psi:
                print(f"  [{i}] {r.news_item.ticker} ({r.regulatory_framework})")
                print(f"      {r.news_item.news[:60]}...")

    print("=" * 60)

    logger.info(f"Summary: {len(results)} total, {psi_count} PSI, {non_psi_count} Non-PSI")
