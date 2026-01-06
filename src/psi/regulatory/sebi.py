"""
SEBI (Securities and Exchange Board of India) regulatory framework.

Implements PSI criteria based on SEBI (Listing Obligations and Disclosure
Requirements) Regulations, particularly Regulation 30.
"""

import logging

from psi.regulatory.base import RegulatoryFramework

logger = logging.getLogger(__name__)


class SEBIFramework(RegulatoryFramework):
    """
    SEBI regulatory framework for India.

    Based on SEBI (Listing Obligations and Disclosure Requirements)
    Regulations, 2015 - particularly Regulation 30 which defines
    events that must be disclosed.
    """

    @property
    def name(self) -> str:
        """Return framework identifier."""
        return "SEBI"

    @property
    def full_name(self) -> str:
        """Return full name of regulatory body."""
        return "Securities and Exchange Board of India"

    @property
    def jurisdiction(self) -> str:
        """Return jurisdiction."""
        return "India"

    def get_psi_definition(self) -> str:
        """Return official PSI definition."""
        return """
Under SEBI (Listing Obligations and Disclosure Requirements) Regulations, 2015,
Price Sensitive Information (PSI) or Unpublished Price Sensitive Information (UPSI)
means any information relating to a company or its securities, directly or indirectly,
that is not generally available which upon becoming generally available, is likely to
materially affect the price of the securities.

As per Regulation 30, listed entities must disclose any information that may have a
bearing on the operation/performance of the company, or information which is price
sensitive in nature.
""".strip()

    def get_psi_criteria(self) -> list[str]:
        """Return PSI classification criteria."""
        return [
            "Financial results (quarterly/half-yearly/annual) - any changes in financial performance",
            "Dividends declared or recommended, including interim dividends",
            "Change in capital structure - bonus issue, rights issue, stock splits, buybacks",
            "Mergers, acquisitions, demergers, amalgamations, or restructuring",
            "Changes in key managerial personnel (MD, CEO, CFO, Company Secretary)",
            "Material agreements, contracts or joint ventures impacting the company",
            "Litigation or disputes that may have material impact",
            "Any frauds or defaults by promoters, directors, or key personnel",
            "Change in rating assigned to any debt instruments or securities",
            "Acquisition or loss of significant orders or contracts",
            "Change in the nature of business or diversification",
            "Any regulatory action against the company or its promoters",
            "Disruption or cessation of material business operations",
            "Change in significant subsidiaries or associate companies",
            "Issue of securities through preferential allotment or QIP",
        ]

    def get_psi_examples(self) -> dict[str, list[str]]:
        """Return examples of PSI and non-PSI news."""
        return {
            "psi": [
                "Company announces Q3 net profit of Rs 500 crore, up 25% YoY",
                "Board recommends final dividend of Rs 10 per share",
                "Company announces 1:1 bonus share issue",
                "XYZ Ltd to acquire ABC Corp for Rs 2000 crore",
                "CEO Mr. Sharma resigns; Mr. Verma appointed as new CEO",
                "Company wins Rs 5000 crore order from Government of India",
                "SEBI imposes penalty on company for disclosure violations",
                "Credit rating downgraded from AAA to AA by CRISIL",
                "Company to demerge its IT services division",
                "Promoter stake reduced from 60% to 45% through OFS",
            ],
            "non_psi": [
                "Company participates in industry conference",
                "New office opened in Bengaluru for employee convenience",
                "Company celebrates annual day function",
                "CSR activity: Tree plantation drive conducted",
                "Company featured in industry magazine article",
                "Routine AGM scheduled for September",
                "Company website redesigned",
                "Employee training program completed",
                "Company signs MOU for potential future collaboration (non-binding)",
                "Management attends investor roadshow",
            ],
        }
