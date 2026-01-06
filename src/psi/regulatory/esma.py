"""
ESMA (European Securities and Markets Authority) regulatory framework.

Implements PSI criteria based on the EU Market Abuse Regulation (EU MAR)
which applies across European Union member states.
"""

import logging

from psi.regulatory.base import RegulatoryFramework

logger = logging.getLogger(__name__)


class ESMAFramework(RegulatoryFramework):
    """
    ESMA regulatory framework for the European Union.

    Based on Regulation (EU) No 596/2014 on market abuse (EU MAR),
    particularly Article 7 which defines inside information and
    Article 17 which mandates disclosure obligations.
    """

    @property
    def name(self) -> str:
        """Return framework identifier."""
        return "ESMA"

    @property
    def full_name(self) -> str:
        """Return full name of regulatory body."""
        return "European Securities and Markets Authority"

    @property
    def jurisdiction(self) -> str:
        """Return jurisdiction."""
        return "European Union"

    def get_psi_definition(self) -> str:
        """Return official PSI definition."""
        return """
Under the EU Market Abuse Regulation (MAR) Article 7, inside information comprises:
1. Information of a precise nature
2. Which has not been made public
3. Relating, directly or indirectly, to one or more issuers or financial instruments
4. Which, if it were made public, would be likely to have a significant effect on
   the prices of those financial instruments or related derivative financial instruments

Article 17 requires issuers to inform the public as soon as possible of inside
information which directly concerns that issuer. Delay is only permitted in limited
circumstances where immediate disclosure would prejudice legitimate interests,
delay would not mislead the public, and confidentiality can be ensured.

The "reasonable investor test" applies: information is material if a reasonable
investor would be likely to use it as part of the basis of investment decisions.
""".strip()

    def get_psi_criteria(self) -> list[str]:
        """Return PSI classification criteria."""
        return [
            "Changes to financial condition or results (profits, turnover, dividends)",
            "Material changes in business operations or strategy",
            "Significant transactions (acquisitions, disposals, joint ventures)",
            "Changes in executive management or supervisory board",
            "Capital increases or decreases, share buybacks",
            "Profit warnings or upgrades to previously announced forecasts",
            "Material contracts won or lost",
            "Litigation or regulatory proceedings with material impact",
            "Rating agency actions affecting the issuer",
            "Changes to rights attaching to securities",
            "Insolvency proceedings or material debt restructuring",
            "Regulatory approvals or rejections for key products/activities",
            "Significant changes in key relationships (customers, suppliers)",
            "Material cybersecurity incidents",
            "Environmental incidents with material financial impact",
        ]

    def get_psi_examples(self) -> dict[str, list[str]]:
        """Return examples of PSI and non-PSI news."""
        return {
            "psi": [
                "Company revises full-year EBITDA guidance down by €50 million",
                "Supervisory Board proposes dividend of €2.00 per share",
                "Company agrees to acquire competitor for €1.2 billion",
                "CEO to step down; CFO appointed as interim CEO",
                "Company loses major customer representing 25% of revenue",
                "European Commission opens antitrust investigation",
                "S&P downgrades credit rating from A to BBB+",
                "Company announces €500 million share buyback programme",
                "Material accounting error discovered in prior year financials",
                "FDA equivalent rejects drug approval application",
            ],
            "non_psi": [
                "Company attends European investor conference",
                "New sustainability report published",
                "Company opens innovation center in Berlin",
                "Management participates in industry panel discussion",
                "Employee diversity initiative launched",
                "Company website available in additional languages",
                "Routine annual general meeting held",
                "Company sponsors European football tournament",
                "Internal reorganization of administrative functions",
                "New corporate branding guidelines released",
            ],
        }
