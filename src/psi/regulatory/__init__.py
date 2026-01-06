"""
Regulatory frameworks module.

Provides PSI classification criteria for various regulatory jurisdictions.
"""

from psi.regulatory.base import RegulatoryFramework
from psi.regulatory.esma import ESMAFramework
from psi.regulatory.fca import FCAFramework
from psi.regulatory.fsa import FSAFramework
from psi.regulatory.mas import MASFramework
from psi.regulatory.registry import (
    FrameworkRegistry,
    get_registry,
    reset_registry,
)
from psi.regulatory.sec import SECFramework
from psi.regulatory.sebi import SEBIFramework
from psi.regulatory.sfc import SFCFramework

__all__ = [
    "RegulatoryFramework",
    "FrameworkRegistry",
    "get_registry",
    "reset_registry",
    "SEBIFramework",
    "SECFramework",
    "FCAFramework",
    "ESMAFramework",
    "MASFramework",
    "SFCFramework",
    "FSAFramework",
]
