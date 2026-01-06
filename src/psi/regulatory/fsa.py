"""
FSA (Financial Services Agency) regulatory framework for Japan.

Implements PSI criteria based on the Financial Instruments and Exchange Act (FIEA)
and Tokyo Stock Exchange disclosure requirements.
"""

import logging

from psi.regulatory.base import RegulatoryFramework

logger = logging.getLogger(__name__)


class FSAFramework(RegulatoryFramework):
    """
    FSA regulatory framework for Japan.

    Based on the Financial Instruments and Exchange Act (FIEA),
    particularly the provisions on material facts (Juyo Jijitsu)
    and timely disclosure requirements of TSE/JPX.
    """

    @property
    def name(self) -> str:
        """Return framework identifier."""
        return "FSA"

    @property
    def full_name(self) -> str:
        """Return full name of regulatory body."""
        return "Financial Services Agency"

    @property
    def jurisdiction(self) -> str:
        """Return jurisdiction."""
        return "Japan"

    def get_psi_definition(self) -> str:
        """Return official PSI definition."""
        return """
Under the Financial Instruments and Exchange Act (FIEA), material facts
(Juyo Jijitsu/重要事実) are facts concerning a listed company that:
1. Are not publicly known
2. Would have a significant impact on investors' investment decisions if disclosed

The FIEA distinguishes between:
- Decided Facts (Kettei Jijitsu): Decisions by the company (e.g., M&A, capital changes)
- Occurred Facts (Hassei Jijitsu): Events that have occurred (e.g., disasters, litigation)
- Financial Information (Kessan Joho): Earnings and financial projections

TSE Timely Disclosure Rules require immediate disclosure of material corporate
information to ensure fair and transparent securities markets.
""".strip()

    def get_psi_criteria(self) -> list[str]:
        """Return PSI classification criteria."""
        return [
            "Issuance of new shares, stock options, or convertible bonds",
            "Capital reduction or share buybacks",
            "Stock splits or reverse stock splits",
            "Dividend decisions or dividend forecast revisions",
            "Mergers, acquisitions, business transfers, or divestitures",
            "Dissolution or bankruptcy filing",
            "Establishment or dissolution of subsidiaries",
            "Material business alliances or joint ventures",
            "Changes in representative directors or key executives",
            "Material litigation or regulatory actions",
            "Revision to earnings forecasts (sales, operating income, net income)",
            "Discovery of material defects in products or services",
            "Occurrence of disasters affecting business operations",
            "Delisting from stock exchange",
            "Changes in major shareholders or shareholding structure",
        ]

    def get_psi_examples(self) -> dict[str, list[str]]:
        """Return examples of PSI and non-PSI news."""
        return {
            "psi": [
                "Company announces FY net profit of ¥100 billion, up 25% YoY",
                "Board resolves interim dividend of ¥50 per share",
                "Company to acquire subsidiary through TOB at ¥3,000 per share",
                "President Yamamoto to resign; Mr. Tanaka appointed successor",
                "Company revises full-year operating income forecast up by ¥20 billion",
                "FSA commences administrative investigation",
                "Major earthquake damages primary manufacturing facility",
                "Company announces 2-for-1 stock split effective April 1",
                "Discovery of accounting irregularities in subsidiary",
                "Company wins ¥500 billion government infrastructure project",
            ],
            "non_psi": [
                "Company participates in Tokyo Motor Show",
                "New Osaka sales office opened",
                "President delivers speech at Keidanren conference",
                "Company sponsors local summer festival",
                "Employee health and wellness programme launched",
                "Annual general meeting of shareholders held",
                "Company receives environmental certification",
                "Management meets investors at IR event (no new information)",
                "Company updates medium-term management plan website",
                "New employee recruitment campaign announced",
            ],
        }
