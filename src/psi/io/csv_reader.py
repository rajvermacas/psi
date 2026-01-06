"""
CSV reader for PSI classification input files.

Reads and validates CSV files containing news items for classification.
"""

import logging
from pathlib import Path

import pandas as pd

from psi.exceptions import CSVValidationError
from psi.graph.state import NewsItem

logger = logging.getLogger(__name__)

# Required columns in the input CSV
REQUIRED_COLUMNS = ["date", "news", "ticker", "region", "exchange"]


def read_csv(file_path: str | Path) -> list[NewsItem]:
    """
    Read and validate a CSV file containing news items.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of NewsItem objects.

    Raises:
        CSVValidationError: If file is invalid or missing required columns.
    """
    file_path = Path(file_path)

    logger.info(f"Reading CSV file: {file_path}")

    # Check file exists
    if not file_path.exists():
        raise CSVValidationError(
            message=f"CSV file not found: {file_path}",
            file_path=str(file_path),
        )

    # Check file extension
    if file_path.suffix.lower() != ".csv":
        raise CSVValidationError(
            message=f"Expected .csv file, got: {file_path.suffix}",
            file_path=str(file_path),
        )

    # Read CSV
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Read {len(df)} rows from CSV")
    except pd.errors.EmptyDataError:
        raise CSVValidationError(
            message="CSV file is empty",
            file_path=str(file_path),
        )
    except pd.errors.ParserError as e:
        raise CSVValidationError(
            message=f"Failed to parse CSV: {str(e)}",
            file_path=str(file_path),
        )
    except Exception as e:
        raise CSVValidationError(
            message=f"Error reading CSV file: {str(e)}",
            file_path=str(file_path),
        )

    # Validate required columns
    validate_columns(df, file_path)

    # Check for empty DataFrame
    if len(df) == 0:
        raise CSVValidationError(
            message="CSV file contains no data rows",
            file_path=str(file_path),
        )

    # Convert to NewsItem list
    news_items = []
    for idx, row in df.iterrows():
        try:
            item = NewsItem(
                date=str(row["date"]).strip(),
                news=str(row["news"]).strip(),
                ticker=str(row["ticker"]).strip(),
                region=str(row["region"]).strip(),
                exchange=str(row["exchange"]).strip(),
            )
            news_items.append(item)
            logger.debug(f"Parsed row {idx + 1}: ticker={item.ticker}")
        except Exception as e:
            raise CSVValidationError(
                message=f"Error parsing row {idx + 1}: {str(e)}",
                file_path=str(file_path),
            )

    logger.info(f"Successfully parsed {len(news_items)} news items from CSV")

    return news_items


def validate_columns(df: pd.DataFrame, file_path: Path) -> None:
    """
    Validate that all required columns are present.

    Args:
        df: Pandas DataFrame to validate.
        file_path: Path to the CSV file (for error reporting).

    Raises:
        CSVValidationError: If required columns are missing.
    """
    # Normalize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.lower().str.strip()

    # Check for missing columns
    missing_columns = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        raise CSVValidationError(
            message=(
                f"CSV is missing required columns: {missing_columns}. "
                f"Required columns are: {REQUIRED_COLUMNS}"
            ),
            file_path=str(file_path),
            missing_columns=missing_columns,
        )

    logger.debug(f"Validated columns: {list(df.columns)}")


def read_csv_to_dicts(file_path: str | Path) -> list[dict]:
    """
    Read CSV file and return as list of dictionaries.

    Convenience function that returns dictionaries instead of NewsItem objects.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of dictionaries with news item data.
    """
    news_items = read_csv(file_path)
    return [item.model_dump() for item in news_items]
