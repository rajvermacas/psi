"""
SEC (Securities and Exchange Commission) regulatory framework.

Implements PSI criteria based on US securities laws including Regulation FD,
Form 8-K filing requirements, and materiality standards.
"""

import logging

from psi.regulatory.base import RegulatoryFramework

logger = logging.getLogger(__name__)


class SECFramework(RegulatoryFramework):
    """
    SEC regulatory framework for the United States.

    Based on Securities Exchange Act of 1934, Regulation FD (Fair Disclosure),
    and Form 8-K current report filing requirements.
    """

    @property
    def name(self) -> str:
        """Return framework identifier."""
        return "SEC"

    @property
    def full_name(self) -> str:
        """Return full name of regulatory body."""
        return "Securities and Exchange Commission"

    @property
    def jurisdiction(self) -> str:
        """Return jurisdiction."""
        return "United States"

    def get_psi_definition(self) -> str:
        """Return official PSI definition."""
        return """
Under US securities law, material non-public information (MNPI) is information that:
1. A reasonable investor would consider important in making an investment decision
2. Would substantially alter the total mix of information available to investors
3. Has not been disseminated in a manner making it available to investors generally

Regulation FD requires that when an issuer discloses material non-public information
to certain individuals (analysts, institutional investors), they must make public
disclosure of that information simultaneously (intentional disclosure) or promptly
(non-intentional disclosure).

Form 8-K requires disclosure of specified material events within four business days.
""".strip()

    def get_psi_criteria(self) -> list[str]:
        """Return PSI classification criteria."""
        return [
            "Earnings announcements and financial results (quarterly/annual)",
            "Changes in earnings estimates or guidance",
            "Material mergers, acquisitions, or divestitures",
            "Entry into or termination of material agreements",
            "Bankruptcy or receivership proceedings",
            "Changes in control of the company",
            "Departure or appointment of principal officers (CEO, CFO, etc.)",
            "Changes in directors or principal officers",
            "Changes in certifying accountant",
            "Delisting or transfer between exchanges",
            "Material impairments or asset write-downs",
            "Unregistered sales of equity securities",
            "Material modifications to rights of security holders",
            "Amendments to articles of incorporation or bylaws",
            "Results of shareholder votes",
            "Material cybersecurity incidents",
            "Pending or threatened material litigation",
        ]

    def get_psi_examples(self) -> dict[str, list[str]]:
        """Return examples of PSI and non-PSI news."""
        return {
            "psi": [
                "Company reports Q4 EPS of $2.50, beating estimates by 15%",
                "Company raises full-year guidance from $10B to $11B revenue",
                "XYZ Corp to acquire ABC Inc for $5 billion in all-stock deal",
                "CEO John Smith announces retirement effective March 31",
                "Company files for Chapter 11 bankruptcy protection",
                "Board approves $10 billion share repurchase program",
                "Company discovers material accounting irregularities",
                "FDA approves company's new drug application",
                "Company receives Wells Notice from SEC",
                "Major customer representing 30% of revenue terminates contract",
                "Company announces workforce reduction of 10,000 employees",
                "Activist investor acquires 9.9% stake, files 13D",
            ],
            "non_psi": [
                "CEO speaks at industry conference (no new information)",
                "Company sponsors local charity event",
                "New marketing campaign launched",
                "Company opens new regional office",
                "Employee awarded industry recognition",
                "Routine board meeting held",
                "Company updates investor presentation (no material changes)",
                "Executive purchases shares through 10b5-1 plan",
                "Company responds to routine SEC comment letter",
                "Annual employee satisfaction survey results released internally",
            ],
        }
