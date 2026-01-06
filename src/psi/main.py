"""
CLI entry point for PSI Classifier.

Usage:
    python -m psi.main --input data/news.csv --output reports/results.json
"""

import argparse
import logging
import sys
from pathlib import Path

from psi import __version__
from psi.config import get_config, setup_logging
from psi.exceptions import CSVValidationError, PSIException
from psi.graph import build_workflow
from psi.graph.state import GraphState
from psi.io import print_summary, read_csv, write_results
from psi.llm import create_llm_provider

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="psi",
        description="PSI Classifier - Classify news as Price Sensitive Information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Classify news from CSV file
    python -m psi.main --input data/news.csv --output reports/results.json

    # Use with verbose logging
    python -m psi.main --input data/news.csv --output reports/results.json --verbose

Environment Variables:
    LLM_PROVIDER        LLM provider to use (azure/openrouter/gemini)
    AZURE_OPENAI_*      Azure OpenAI configuration (if LLM_PROVIDER=azure)
    OPENROUTER_*        OpenRouter configuration (if LLM_PROVIDER=openrouter)
    GEMINI_*            Gemini configuration (if LLM_PROVIDER=gemini)
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input CSV file with news items",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output JSON file for results",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"PSI Classifier {__version__}",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for PSI Classifier CLI.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    # Setup logging
    try:
        config = get_config()
        setup_logging(config)

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info(f"PSI Classifier v{__version__} starting")
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"LLM Provider: {config.llm_provider.value}")
    except PSIException as e:
        print(f"Configuration error: {e.message}", file=sys.stderr)
        return 1

    # Read input CSV
    try:
        logger.info("Reading input CSV file")
        news_items = read_csv(args.input)
        logger.info(f"Loaded {len(news_items)} news items")

        if len(news_items) == 0:
            print("No news items found in input file", file=sys.stderr)
            return 1

    except CSVValidationError as e:
        print(f"CSV validation error: {e.message}", file=sys.stderr)
        if e.missing_columns:
            print(f"Missing columns: {e.missing_columns}", file=sys.stderr)
        return 1

    # Create LLM provider
    try:
        logger.info("Initializing LLM provider")
        llm_provider = create_llm_provider(config)
        logger.info(
            f"Using {llm_provider.provider_name} provider "
            f"with model {llm_provider.model_name}"
        )
    except PSIException as e:
        print(f"LLM provider error: {e.message}", file=sys.stderr)
        return 1

    # Build and run workflow
    try:
        logger.info("Building classification workflow")
        workflow = build_workflow(llm_provider)

        # Create initial state
        initial_state = GraphState(news_items=news_items)

        logger.info("Starting classification pipeline")
        print(f"\nClassifying {len(news_items)} news items...\n")

        # Run workflow
        final_state = workflow.invoke(initial_state)

        # Handle result - could be dict or GraphState
        if isinstance(final_state, dict):
            results = final_state.get("results", [])
        else:
            results = final_state.results

        logger.info(f"Classification complete: {len(results)} results")

    except PSIException as e:
        print(f"Classification error: {e.message}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during classification")
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        return 1

    # Write output
    try:
        logger.info("Writing results to output file")
        output_path = write_results(
            results=results,
            output_path=args.output,
            model_name=llm_provider.model_name,
            provider_name=llm_provider.provider_name,
        )
        logger.info(f"Results written to: {output_path}")
    except Exception as e:
        logger.exception("Error writing results")
        print(f"Error writing results: {str(e)}", file=sys.stderr)
        return 1

    # Print summary
    print_summary(results)
    print(f"\nResults written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
