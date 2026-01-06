"""
MAS (Monetary Authority of Singapore) regulatory framework.

Implements PSI criteria based on the Securities and Futures Act (SFA)
and Singapore Exchange (SGX) Listing Rules.
"""

import logging

from psi.regulatory.base import RegulatoryFramework

logger = logging.getLogger(__name__)


class MASFramework(RegulatoryFramework):
    """
    MAS regulatory framework for Singapore.

    Based on the Securities and Futures Act (SFA) which prohibits
    insider trading, and SGX Listing Rules which mandate continuous
    disclosure of material information.
    """

    @property
    def name(self) -> str:
        """Return framework identifier."""
        return "MAS"

    @property
    def full_name(self) -> str:
        """Return full name of regulatory body."""
        return "Monetary Authority of Singapore"

    @property
    def jurisdiction(self) -> str:
        """Return jurisdiction."""
        return "Singapore"

    def get_psi_definition(self) -> str:
        """Return official PSI definition."""
        return """
Under the Securities and Futures Act (SFA), material information is information that:
1. Is not generally available
2. If it were generally available, a reasonable person would expect it to have a
   material effect on the price or value of securities

Under SGX Listing Rules, an issuer must disclose any information known to the issuer
concerning it or any of its subsidiaries or associated companies which:
- Is necessary to avoid the establishment of a false market in the issuer's securities
- Would be likely to materially affect the price or value of its securities

The "reasonable person" test and "trading halt" provisions apply where information
may have a material effect on price.
""".strip()

    def get_psi_criteria(self) -> list[str]:
        """Return PSI classification criteria."""
        return [
            "Material changes in financial condition or results of operations",
            "Declaration of dividends or distribution to shareholders",
            "Major transactions (acquisitions, disposals) exceeding thresholds",
            "Interested person transactions above materiality thresholds",
            "Takeover or merger situations",
            "Changes in directors, CEO, or CFO",
            "Material borrowings or loan defaults",
            "Qualified audit opinion or change in auditors",
            "Litigation or claims with material impact",
            "Changes in share capital or issued securities",
            "Gain or loss of major customers or suppliers",
            "Entry into or termination of material contracts",
            "Regulatory investigations or sanctions",
            "Material joint ventures or strategic alliances",
            "Significant changes in business direction or strategy",
        ]

    def get_psi_examples(self) -> dict[str, list[str]]:
        """Return examples of PSI and non-PSI news."""
        return {
            "psi": [
                "Company reports H1 net profit of S$50 million, up 30% YoY",
                "Board declares interim dividend of S$0.05 per share",
                "Company to acquire Malaysian competitor for S$200 million",
                "CEO Mr. Tan resigns; Board appoints Mr. Lee as successor",
                "Company defaults on S$100 million loan facility",
                "MAS commences investigation into trading irregularities",
                "Company wins S$500 million government infrastructure contract",
                "Major shareholder reduces stake from 20% to 10%",
                "Company announces rights issue of S$80 million",
                "Fire destroys main manufacturing facility",
            ],
            "non_psi": [
                "Company participates in Singapore FinTech Festival",
                "New office opened at Marina Bay Financial Centre",
                "Company sponsors Singapore Grand Prix",
                "Employee town hall meeting held",
                "Company launches corporate social responsibility programme",
                "Management meets analysts (no new material information)",
                "Annual report published (no new information)",
                "Company updates investor presentation on website",
                "New marketing campaign in Southeast Asia launched",
                "Routine annual general meeting conducted",
            ],
        }
