"""
Tests for regulatory framework implementations.
"""

import pytest

from psi.regulatory import (
    ESMAFramework,
    FCAFramework,
    FSAFramework,
    MASFramework,
    RegulatoryFramework,
    SEBIFramework,
    SECFramework,
    SFCFramework,
)


class TestRegulatoryFrameworkInterface:
    """Tests to verify all frameworks implement the interface correctly."""

    @pytest.fixture(
        params=[
            SEBIFramework,
            SECFramework,
            FCAFramework,
            ESMAFramework,
            MASFramework,
            SFCFramework,
            FSAFramework,
        ]
    )
    def framework(self, request) -> RegulatoryFramework:
        """Parameterized fixture for all framework types."""
        return request.param()

    def test_has_name(self, framework: RegulatoryFramework) -> None:
        """Test framework has a non-empty name."""
        assert framework.name
        assert isinstance(framework.name, str)
        assert len(framework.name) > 0

    def test_has_full_name(self, framework: RegulatoryFramework) -> None:
        """Test framework has a non-empty full name."""
        assert framework.full_name
        assert isinstance(framework.full_name, str)
        assert len(framework.full_name) > 0

    def test_has_jurisdiction(self, framework: RegulatoryFramework) -> None:
        """Test framework has a non-empty jurisdiction."""
        assert framework.jurisdiction
        assert isinstance(framework.jurisdiction, str)
        assert len(framework.jurisdiction) > 0

    def test_has_psi_definition(self, framework: RegulatoryFramework) -> None:
        """Test framework has a non-empty PSI definition."""
        definition = framework.get_psi_definition()
        assert definition
        assert isinstance(definition, str)
        assert len(definition) > 100  # Should be substantial

    def test_has_psi_criteria(self, framework: RegulatoryFramework) -> None:
        """Test framework has PSI criteria list."""
        criteria = framework.get_psi_criteria()
        assert criteria
        assert isinstance(criteria, list)
        assert len(criteria) >= 10  # Should have multiple criteria
        assert all(isinstance(c, str) for c in criteria)

    def test_has_psi_examples(self, framework: RegulatoryFramework) -> None:
        """Test framework has PSI and non-PSI examples."""
        examples = framework.get_psi_examples()
        assert examples
        assert isinstance(examples, dict)
        assert "psi" in examples
        assert "non_psi" in examples
        assert len(examples["psi"]) >= 5
        assert len(examples["non_psi"]) >= 5

    def test_prompt_context_contains_all_info(
        self, framework: RegulatoryFramework
    ) -> None:
        """Test prompt context contains all required information."""
        context = framework.get_prompt_context()

        assert framework.name in context
        assert framework.full_name in context
        assert framework.jurisdiction in context

        # Should contain definition and criteria
        assert "Definition" in context or "definition" in context
        assert "Criteria" in context or "criteria" in context

        # Should contain examples section
        assert "PSI" in context
        assert "NOT PSI" in context or "not PSI" in context.lower()

    def test_repr(self, framework: RegulatoryFramework) -> None:
        """Test framework has meaningful repr."""
        repr_str = repr(framework)
        assert framework.name in repr_str
        assert framework.jurisdiction in repr_str


class TestSEBIFramework:
    """Specific tests for SEBI framework."""

    def test_properties(self) -> None:
        """Test SEBI framework properties."""
        framework = SEBIFramework()

        assert framework.name == "SEBI"
        assert "Securities and Exchange Board of India" in framework.full_name
        assert framework.jurisdiction == "India"

    def test_criteria_includes_indian_requirements(self) -> None:
        """Test SEBI criteria includes India-specific items."""
        framework = SEBIFramework()
        criteria = framework.get_psi_criteria()
        criteria_text = " ".join(criteria).lower()

        # Check for India-specific terminology
        assert "dividend" in criteria_text
        assert "financial results" in criteria_text or "quarterly" in criteria_text


class TestSECFramework:
    """Specific tests for SEC framework."""

    def test_properties(self) -> None:
        """Test SEC framework properties."""
        framework = SECFramework()

        assert framework.name == "SEC"
        assert "Securities and Exchange Commission" in framework.full_name
        assert framework.jurisdiction == "United States"

    def test_definition_mentions_regulation_fd(self) -> None:
        """Test SEC definition mentions Regulation FD."""
        framework = SECFramework()
        definition = framework.get_psi_definition()

        assert "Regulation FD" in definition or "Form 8-K" in definition


class TestFCAFramework:
    """Specific tests for FCA framework."""

    def test_properties(self) -> None:
        """Test FCA framework properties."""
        framework = FCAFramework()

        assert framework.name == "FCA"
        assert "Financial Conduct Authority" in framework.full_name
        assert framework.jurisdiction == "United Kingdom"

    def test_definition_mentions_uk_mar(self) -> None:
        """Test FCA definition mentions UK MAR."""
        framework = FCAFramework()
        definition = framework.get_psi_definition()

        assert "UK MAR" in definition or "Market Abuse Regulation" in definition


class TestESMAFramework:
    """Specific tests for ESMA framework."""

    def test_properties(self) -> None:
        """Test ESMA framework properties."""
        framework = ESMAFramework()

        assert framework.name == "ESMA"
        assert "European Securities and Markets Authority" in framework.full_name
        assert framework.jurisdiction == "European Union"

    def test_definition_mentions_eu_mar(self) -> None:
        """Test ESMA definition mentions EU MAR."""
        framework = ESMAFramework()
        definition = framework.get_psi_definition()

        assert "EU MAR" in definition or "EU Market Abuse Regulation" in definition


class TestMASFramework:
    """Specific tests for MAS framework."""

    def test_properties(self) -> None:
        """Test MAS framework properties."""
        framework = MASFramework()

        assert framework.name == "MAS"
        assert "Monetary Authority of Singapore" in framework.full_name
        assert framework.jurisdiction == "Singapore"

    def test_definition_mentions_sfa(self) -> None:
        """Test MAS definition mentions SFA."""
        framework = MASFramework()
        definition = framework.get_psi_definition()

        assert "SFA" in definition or "Securities and Futures Act" in definition


class TestSFCFramework:
    """Specific tests for SFC framework."""

    def test_properties(self) -> None:
        """Test SFC framework properties."""
        framework = SFCFramework()

        assert framework.name == "SFC"
        assert "Securities and Futures Commission" in framework.full_name
        assert framework.jurisdiction == "Hong Kong"

    def test_definition_mentions_sfo(self) -> None:
        """Test SFC definition mentions SFO."""
        framework = SFCFramework()
        definition = framework.get_psi_definition()

        assert "SFO" in definition or "Securities and Futures Ordinance" in definition


class TestFSAFramework:
    """Specific tests for FSA framework."""

    def test_properties(self) -> None:
        """Test FSA framework properties."""
        framework = FSAFramework()

        assert framework.name == "FSA"
        assert "Financial Services Agency" in framework.full_name
        assert framework.jurisdiction == "Japan"

    def test_definition_mentions_fiea(self) -> None:
        """Test FSA definition mentions FIEA."""
        framework = FSAFramework()
        definition = framework.get_psi_definition()

        assert (
            "FIEA" in definition
            or "Financial Instruments and Exchange Act" in definition
        )
