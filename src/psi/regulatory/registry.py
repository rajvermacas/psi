"""
Regulatory framework registry.

Provides mapping from region/exchange combinations to regulatory frameworks.
Follows fail-fast principle - raises UnknownRegionError for unknown combinations.
"""

import logging
from typing import TYPE_CHECKING

from psi.exceptions import UnknownRegionError

if TYPE_CHECKING:
    from psi.regulatory.base import RegulatoryFramework

logger = logging.getLogger(__name__)


class FrameworkRegistry:
    """
    Registry for mapping region/exchange combinations to regulatory frameworks.

    The registry maintains a mapping of (region, exchange) tuples to framework
    instances. It supports case-insensitive lookups and provides detailed
    error messages for unknown combinations.

    Supported mappings (as per architecture):
    - India: NSE, BSE -> SEBI
    - US: NYSE, NASDAQ, AMEX -> SEC
    - UK: LSE, AIM -> FCA
    - EU/Germany/France/Netherlands: EURONEXT, XETRA -> ESMA
    - Singapore: SGX -> MAS
    - Hong Kong: HKEX, HKG -> SFC
    - Japan: TSE, JPX, JASDAQ -> FSA
    """

    def __init__(self) -> None:
        """Initialize the registry with empty mappings."""
        # Mapping: (normalized_region, normalized_exchange) -> framework_name
        self._region_exchange_mapping: dict[tuple[str, str], str] = {}

        # Framework instances by name
        self._frameworks: dict[str, "RegulatoryFramework"] = {}

        logger.debug("FrameworkRegistry initialized")

    def register_framework(
        self,
        framework: "RegulatoryFramework",
        regions: list[str],
        exchanges: list[str],
    ) -> None:
        """
        Register a framework with its supported regions and exchanges.

        Args:
            framework: RegulatoryFramework instance to register.
            regions: List of region names this framework covers.
            exchanges: List of exchange names this framework covers.
        """
        framework_name = framework.name.upper()
        self._frameworks[framework_name] = framework

        # Create mappings for all region/exchange combinations
        for region in regions:
            for exchange in exchanges:
                key = (region.upper(), exchange.upper())
                self._region_exchange_mapping[key] = framework_name
                logger.debug(
                    f"Registered mapping: {region}/{exchange} -> {framework_name}"
                )

        logger.info(
            f"Registered framework {framework_name} for regions={regions}, "
            f"exchanges={exchanges}"
        )

    def get_framework(self, region: str, exchange: str) -> "RegulatoryFramework":
        """
        Get the regulatory framework for a region/exchange combination.

        Args:
            region: Geographic region (e.g., 'India', 'US').
            exchange: Stock exchange (e.g., 'NSE', 'NYSE').

        Returns:
            RegulatoryFramework instance for the given combination.

        Raises:
            UnknownRegionError: If no framework is registered for this combination.
        """
        key = (region.upper(), exchange.upper())

        logger.debug(f"Looking up framework for region={region}, exchange={exchange}")

        if key not in self._region_exchange_mapping:
            supported = self.list_supported_combinations()
            raise UnknownRegionError(
                message=(
                    f"No regulatory framework found for region='{region}', "
                    f"exchange='{exchange}'. "
                    f"Supported combinations: {supported}"
                ),
                region=region,
                exchange=exchange,
                supported_combinations=supported,
            )

        framework_name = self._region_exchange_mapping[key]
        framework = self._frameworks[framework_name]

        logger.info(
            f"Resolved framework {framework_name} for region={region}, "
            f"exchange={exchange}"
        )

        return framework

    def list_supported_combinations(self) -> list[str]:
        """
        List all supported region/exchange combinations.

        Returns:
            List of strings in format 'Region/Exchange -> Framework'.
        """
        combinations = []
        for (region, exchange), framework_name in sorted(
            self._region_exchange_mapping.items()
        ):
            combinations.append(f"{region}/{exchange} -> {framework_name}")
        return combinations

    def list_frameworks(self) -> list[str]:
        """
        List all registered framework names.

        Returns:
            List of registered framework names.
        """
        return sorted(self._frameworks.keys())

    def get_framework_by_name(self, name: str) -> "RegulatoryFramework":
        """
        Get a framework by its name.

        Args:
            name: Framework name (e.g., 'SEBI', 'SEC').

        Returns:
            RegulatoryFramework instance.

        Raises:
            UnknownRegionError: If framework name is not registered.
        """
        normalized_name = name.upper()

        if normalized_name not in self._frameworks:
            available = self.list_frameworks()
            raise UnknownRegionError(
                message=(
                    f"Unknown framework: '{name}'. "
                    f"Available frameworks: {available}"
                ),
                region=None,
                exchange=None,
                supported_combinations=available,
            )

        return self._frameworks[normalized_name]


# Global registry instance
_registry: FrameworkRegistry | None = None


def get_registry() -> FrameworkRegistry:
    """
    Get the global framework registry.

    Returns a singleton registry instance, initializing it with all
    supported frameworks on first call.

    Returns:
        Configured FrameworkRegistry instance.
    """
    global _registry

    if _registry is None:
        logger.info("Initializing global framework registry")
        _registry = FrameworkRegistry()
        _initialize_registry(_registry)

    return _registry


def _initialize_registry(registry: FrameworkRegistry) -> None:
    """
    Initialize the registry with all supported frameworks.

    Args:
        registry: FrameworkRegistry instance to populate.
    """
    # Import framework implementations here to avoid circular imports
    from psi.regulatory.esma import ESMAFramework
    from psi.regulatory.fca import FCAFramework
    from psi.regulatory.fsa import FSAFramework
    from psi.regulatory.mas import MASFramework
    from psi.regulatory.sec import SECFramework
    from psi.regulatory.sebi import SEBIFramework
    from psi.regulatory.sfc import SFCFramework

    # Register SEBI (India)
    registry.register_framework(
        framework=SEBIFramework(),
        regions=["INDIA"],
        exchanges=["NSE", "BSE"],
    )

    # Register SEC (United States)
    registry.register_framework(
        framework=SECFramework(),
        regions=["US", "USA", "UNITED STATES"],
        exchanges=["NYSE", "NASDAQ", "AMEX"],
    )

    # Register FCA (United Kingdom)
    registry.register_framework(
        framework=FCAFramework(),
        regions=["UK", "UNITED KINGDOM", "GB", "GREAT BRITAIN"],
        exchanges=["LSE", "AIM"],
    )

    # Register ESMA (European Union)
    registry.register_framework(
        framework=ESMAFramework(),
        regions=["EU", "GERMANY", "FRANCE", "NETHERLANDS", "EUROPEAN UNION"],
        exchanges=["EURONEXT", "XETRA"],
    )

    # Register MAS (Singapore)
    registry.register_framework(
        framework=MASFramework(),
        regions=["SINGAPORE", "SG"],
        exchanges=["SGX"],
    )

    # Register SFC (Hong Kong)
    registry.register_framework(
        framework=SFCFramework(),
        regions=["HONG KONG", "HK"],
        exchanges=["HKEX", "HKG"],
    )

    # Register FSA (Japan)
    registry.register_framework(
        framework=FSAFramework(),
        regions=["JAPAN", "JP"],
        exchanges=["TSE", "JPX", "JASDAQ"],
    )

    logger.info(
        f"Registry initialized with {len(registry.list_frameworks())} frameworks"
    )


def reset_registry() -> None:
    """
    Reset the global registry.

    Primarily used for testing to ensure a clean state.
    """
    global _registry
    _registry = None
    logger.debug("Global registry reset")
