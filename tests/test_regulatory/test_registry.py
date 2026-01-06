"""
Tests for regulatory framework registry.
"""

import pytest

from psi.exceptions import UnknownRegionError
from psi.regulatory import (
    ESMAFramework,
    FCAFramework,
    FrameworkRegistry,
    FSAFramework,
    MASFramework,
    RegulatoryFramework,
    SEBIFramework,
    SECFramework,
    SFCFramework,
    get_registry,
    reset_registry,
)


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset global registry before and after each test."""
    reset_registry()
    yield
    reset_registry()


class TestFrameworkRegistry:
    """Tests for FrameworkRegistry class."""

    def test_register_framework(self) -> None:
        """Test registering a framework."""
        registry = FrameworkRegistry()
        framework = SEBIFramework()

        registry.register_framework(
            framework=framework,
            regions=["INDIA"],
            exchanges=["NSE", "BSE"],
        )

        assert "SEBI" in registry.list_frameworks()

    def test_get_framework_valid(self) -> None:
        """Test getting a valid framework."""
        registry = FrameworkRegistry()
        framework = SEBIFramework()

        registry.register_framework(
            framework=framework,
            regions=["INDIA"],
            exchanges=["NSE", "BSE"],
        )

        result = registry.get_framework("India", "NSE")
        assert result is framework
        assert result.name == "SEBI"

    def test_get_framework_case_insensitive(self) -> None:
        """Test that lookups are case insensitive."""
        registry = FrameworkRegistry()
        framework = SECFramework()

        registry.register_framework(
            framework=framework,
            regions=["US"],
            exchanges=["NYSE"],
        )

        # All these should work
        assert registry.get_framework("us", "nyse").name == "SEC"
        assert registry.get_framework("US", "NYSE").name == "SEC"
        assert registry.get_framework("Us", "Nyse").name == "SEC"

    def test_get_framework_unknown_region(self) -> None:
        """Test that unknown region raises error."""
        registry = FrameworkRegistry()
        registry.register_framework(
            framework=SEBIFramework(),
            regions=["INDIA"],
            exchanges=["NSE"],
        )

        with pytest.raises(UnknownRegionError) as exc_info:
            registry.get_framework("Antarctica", "ICE")

        assert exc_info.value.region == "Antarctica"
        assert exc_info.value.exchange == "ICE"
        assert len(exc_info.value.supported_combinations) > 0

    def test_get_framework_unknown_exchange(self) -> None:
        """Test that unknown exchange for known region raises error."""
        registry = FrameworkRegistry()
        registry.register_framework(
            framework=SEBIFramework(),
            regions=["INDIA"],
            exchanges=["NSE", "BSE"],
        )

        with pytest.raises(UnknownRegionError) as exc_info:
            registry.get_framework("India", "UNKNOWN_EXCHANGE")

        assert "India" in str(exc_info.value)
        assert "UNKNOWN_EXCHANGE" in str(exc_info.value)

    def test_list_supported_combinations(self) -> None:
        """Test listing supported combinations."""
        registry = FrameworkRegistry()
        registry.register_framework(
            framework=SEBIFramework(),
            regions=["INDIA"],
            exchanges=["NSE", "BSE"],
        )
        registry.register_framework(
            framework=SECFramework(),
            regions=["US"],
            exchanges=["NYSE"],
        )

        combinations = registry.list_supported_combinations()

        assert len(combinations) == 3
        assert any("INDIA" in c and "NSE" in c for c in combinations)
        assert any("INDIA" in c and "BSE" in c for c in combinations)
        assert any("US" in c and "NYSE" in c for c in combinations)

    def test_list_frameworks(self) -> None:
        """Test listing registered frameworks."""
        registry = FrameworkRegistry()
        registry.register_framework(
            framework=SEBIFramework(),
            regions=["INDIA"],
            exchanges=["NSE"],
        )
        registry.register_framework(
            framework=SECFramework(),
            regions=["US"],
            exchanges=["NYSE"],
        )

        frameworks = registry.list_frameworks()

        assert "SEBI" in frameworks
        assert "SEC" in frameworks

    def test_get_framework_by_name(self) -> None:
        """Test getting framework by name."""
        registry = FrameworkRegistry()
        framework = SEBIFramework()

        registry.register_framework(
            framework=framework,
            regions=["INDIA"],
            exchanges=["NSE"],
        )

        result = registry.get_framework_by_name("SEBI")
        assert result is framework

    def test_get_framework_by_name_case_insensitive(self) -> None:
        """Test getting framework by name is case insensitive."""
        registry = FrameworkRegistry()
        registry.register_framework(
            framework=SEBIFramework(),
            regions=["INDIA"],
            exchanges=["NSE"],
        )

        assert registry.get_framework_by_name("sebi").name == "SEBI"
        assert registry.get_framework_by_name("SEBI").name == "SEBI"
        assert registry.get_framework_by_name("Sebi").name == "SEBI"

    def test_get_framework_by_name_unknown(self) -> None:
        """Test getting unknown framework by name raises error."""
        registry = FrameworkRegistry()
        registry.register_framework(
            framework=SEBIFramework(),
            regions=["INDIA"],
            exchanges=["NSE"],
        )

        with pytest.raises(UnknownRegionError) as exc_info:
            registry.get_framework_by_name("UNKNOWN")

        assert "UNKNOWN" in str(exc_info.value)


class TestGetRegistry:
    """Tests for get_registry function."""

    def test_get_registry_returns_singleton(self) -> None:
        """Test that get_registry returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_get_registry_initialized_with_all_frameworks(self) -> None:
        """Test that registry is initialized with all frameworks."""
        registry = get_registry()
        frameworks = registry.list_frameworks()

        assert "SEBI" in frameworks
        assert "SEC" in frameworks
        assert "FCA" in frameworks
        assert "ESMA" in frameworks
        assert "MAS" in frameworks
        assert "SFC" in frameworks
        assert "FSA" in frameworks

    def test_sebi_region_exchange_mapping(self) -> None:
        """Test SEBI region/exchange mapping."""
        registry = get_registry()

        assert registry.get_framework("India", "NSE").name == "SEBI"
        assert registry.get_framework("India", "BSE").name == "SEBI"

    def test_sec_region_exchange_mapping(self) -> None:
        """Test SEC region/exchange mapping."""
        registry = get_registry()

        assert registry.get_framework("US", "NYSE").name == "SEC"
        assert registry.get_framework("US", "NASDAQ").name == "SEC"
        assert registry.get_framework("US", "AMEX").name == "SEC"
        assert registry.get_framework("USA", "NYSE").name == "SEC"
        assert registry.get_framework("United States", "NYSE").name == "SEC"

    def test_fca_region_exchange_mapping(self) -> None:
        """Test FCA region/exchange mapping."""
        registry = get_registry()

        assert registry.get_framework("UK", "LSE").name == "FCA"
        assert registry.get_framework("UK", "AIM").name == "FCA"
        assert registry.get_framework("United Kingdom", "LSE").name == "FCA"

    def test_esma_region_exchange_mapping(self) -> None:
        """Test ESMA region/exchange mapping."""
        registry = get_registry()

        assert registry.get_framework("EU", "EURONEXT").name == "ESMA"
        assert registry.get_framework("EU", "XETRA").name == "ESMA"
        assert registry.get_framework("Germany", "XETRA").name == "ESMA"
        assert registry.get_framework("France", "EURONEXT").name == "ESMA"
        assert registry.get_framework("Netherlands", "EURONEXT").name == "ESMA"

    def test_mas_region_exchange_mapping(self) -> None:
        """Test MAS region/exchange mapping."""
        registry = get_registry()

        assert registry.get_framework("Singapore", "SGX").name == "MAS"
        assert registry.get_framework("SG", "SGX").name == "MAS"

    def test_sfc_region_exchange_mapping(self) -> None:
        """Test SFC region/exchange mapping."""
        registry = get_registry()

        assert registry.get_framework("Hong Kong", "HKEX").name == "SFC"
        assert registry.get_framework("Hong Kong", "HKG").name == "SFC"
        assert registry.get_framework("HK", "HKEX").name == "SFC"

    def test_fsa_region_exchange_mapping(self) -> None:
        """Test FSA region/exchange mapping."""
        registry = get_registry()

        assert registry.get_framework("Japan", "TSE").name == "FSA"
        assert registry.get_framework("Japan", "JPX").name == "FSA"
        assert registry.get_framework("Japan", "JASDAQ").name == "FSA"
        assert registry.get_framework("JP", "TSE").name == "FSA"


class TestResetRegistry:
    """Tests for reset_registry function."""

    def test_reset_clears_global_registry(self) -> None:
        """Test that reset creates a new registry on next get."""
        registry1 = get_registry()
        reset_registry()
        registry2 = get_registry()

        # Should be different instances
        assert registry1 is not registry2
