"""
Tests for CSV reader module.
"""

import pytest

from psi.exceptions import CSVValidationError
from psi.graph.state import NewsItem
from psi.io import read_csv, read_csv_to_dicts


class TestReadCSV:
    """Tests for read_csv function."""

    def test_read_valid_csv(self, tmp_path) -> None:
        """Test reading a valid CSV file."""
        csv_content = """date,news,ticker,region,exchange
2024-01-15,"Company announces earnings",COMP,India,NSE
2024-01-16,"CEO attends conference",COMP,US,NYSE"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        result = read_csv(csv_file)

        assert len(result) == 2
        assert all(isinstance(item, NewsItem) for item in result)
        assert result[0].ticker == "COMP"
        assert result[0].region == "India"
        assert result[1].exchange == "NYSE"

    def test_read_csv_file_not_found(self) -> None:
        """Test reading non-existent file raises error."""
        with pytest.raises(CSVValidationError) as exc_info:
            read_csv("/nonexistent/path/file.csv")

        assert "not found" in str(exc_info.value).lower()

    def test_read_csv_wrong_extension(self, tmp_path) -> None:
        """Test reading non-CSV file raises error."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a csv")

        with pytest.raises(CSVValidationError) as exc_info:
            read_csv(txt_file)

        assert ".csv" in str(exc_info.value)

    def test_read_csv_missing_columns(self, tmp_path) -> None:
        """Test reading CSV with missing columns raises error."""
        csv_content = """date,news,ticker
2024-01-15,"Company announces earnings",COMP"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(CSVValidationError) as exc_info:
            read_csv(csv_file)

        assert "region" in exc_info.value.missing_columns
        assert "exchange" in exc_info.value.missing_columns

    def test_read_csv_empty_file(self, tmp_path) -> None:
        """Test reading empty CSV raises error."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("")

        with pytest.raises(CSVValidationError):
            read_csv(csv_file)

    def test_read_csv_headers_only(self, tmp_path) -> None:
        """Test reading CSV with only headers raises error."""
        csv_content = """date,news,ticker,region,exchange"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(CSVValidationError) as exc_info:
            read_csv(csv_file)

        assert "no data" in str(exc_info.value).lower()

    def test_read_csv_case_insensitive_columns(self, tmp_path) -> None:
        """Test that column names are case insensitive."""
        csv_content = """DATE,NEWS,TICKER,Region,Exchange
2024-01-15,"Test news",COMP,India,NSE"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        result = read_csv(csv_file)

        assert len(result) == 1
        assert result[0].ticker == "COMP"

    def test_read_csv_strips_whitespace(self, tmp_path) -> None:
        """Test that values are stripped of whitespace."""
        csv_content = """date,news,ticker,region,exchange
2024-01-15,  Test news  ,  COMP  ,  India  ,  NSE  """

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        result = read_csv(csv_file)

        assert result[0].news == "Test news"
        assert result[0].ticker == "COMP"
        assert result[0].region == "India"

    def test_read_csv_from_test_data(self) -> None:
        """Test reading the sample test data file."""
        result = read_csv("test_data/sample_news.csv")

        assert len(result) == 10
        assert result[0].ticker == "COMP"
        assert result[0].region == "India"


class TestReadCSVToDicts:
    """Tests for read_csv_to_dicts function."""

    def test_returns_list_of_dicts(self, tmp_path) -> None:
        """Test that read_csv_to_dicts returns dictionaries."""
        csv_content = """date,news,ticker,region,exchange
2024-01-15,"Test news",COMP,India,NSE"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        result = read_csv_to_dicts(csv_file)

        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["ticker"] == "COMP"
        assert "date" in result[0]
        assert "news" in result[0]
        assert "region" in result[0]
        assert "exchange" in result[0]
