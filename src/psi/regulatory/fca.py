"""
FCA (Financial Conduct Authority) regulatory framework.

Implements PSI criteria based on UK Market Abuse Regulation (UK MAR)
and Disclosure Guidance and Transparency Rules (DTR).
"""

import logging

from psi.regulatory.base import RegulatoryFramework

logger = logging.getLogger(__name__)


class FCAFramework(RegulatoryFramework):
    """
    FCA regulatory framework for the United Kingdom.

    Based on UK Market Abuse Regulation (UK MAR), particularly Article 7
    which defines inside information, and the FCA's Disclosure Guidance
    and Transparency Rules (DTR).
    """

    @property
    def name(self) -> str:
        """Return framework identifier."""
        return "FCA"

    @property
    def full_name(self) -> str:
        """Return full name of regulatory body."""
        return "Financial Conduct Authority"

    @property
    def jurisdiction(self) -> str:
        """Return jurisdiction."""
        return "United Kingdom"

    def get_psi_definition(self) -> str:
        """Return official PSI definition."""
        return """
Under the UK Market Abuse Regulation (UK MAR), inside information means information
of a precise nature, which:
1. Has not been made public
2. Relates, directly or indirectly, to one or more issuers or financial instruments
3. Would, if it were made public, be likely to have a significant effect on the prices
   of those financial instruments or related derivative financial instruments

Information shall be deemed to be of a precise nature if it indicates circumstances
that exist or may reasonably be expected to come into existence, and is specific
enough to enable a conclusion to be drawn as to the possible effect on prices.

A significant effect on price means information that a reasonable investor would be
likely to use as part of the basis of their investment decisions.
""".strip()

    def get_psi_criteria(self) -> list[str]:
        """Return PSI classification criteria."""
        return [
            "Changes to financial condition or performance (profits, losses, dividends)",
            "Changes in expectations of performance previously announced",
            "Changes to board of directors or key executives",
            "Material acquisitions, disposals, or joint ventures",
            "Significant new products, services, or markets",
            "Changes to capital structure or significant financing arrangements",
            "Major litigation or regulatory investigations",
            "Revocation or cancellation of credit facilities",
            "Changes to auditors or qualified audit opinion",
            "Notifiable transactions under the Listing Rules",
            "Profit warnings or trading updates indicating material deviation",
            "Significant changes in customer relationships or contracts",
            "Breach of loan covenants",
            "Regulatory approval or rejection of key applications",
            "Material cybersecurity incidents or data breaches",
        ]

    def get_psi_examples(self) -> dict[str, list[str]]:
        """Return examples of PSI and non-PSI news."""
        return {
            "psi": [
                "Company issues profit warning, expects H1 profits 20% below guidance",
                "Board recommends final dividend of 15p per share",
                "Company to acquire competitor for £500 million",
                "CEO announces immediate departure for personal reasons",
                "Company wins £200 million government contract",
                "FCA launches formal investigation into company practices",
                "Credit facility of £100 million terminated by lenders",
                "Company discovers material misstatement in prior financials",
                "Major customer (15% of revenue) terminates supply agreement",
                "NHS rejects company's application for drug approval",
            ],
            "non_psi": [
                "Chairman gives keynote speech at industry summit",
                "Company launches new sustainability initiative",
                "Annual general meeting scheduled for May",
                "Company participates in industry trade fair",
                "New apprenticeship programme announced",
                "Office refurbishment completed",
                "Company sponsors local football club",
                "Management meets with institutional investors (no new info)",
                "Company updates corporate website",
                "Employee engagement survey results published internally",
            ],
        }
