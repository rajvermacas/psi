"""
SFC (Securities and Futures Commission) regulatory framework for Hong Kong.

Implements PSI criteria based on the Securities and Futures Ordinance (SFO)
and HKEX Main Board Listing Rules.
"""

import logging

from psi.regulatory.base import RegulatoryFramework

logger = logging.getLogger(__name__)


class SFCFramework(RegulatoryFramework):
    """
    SFC regulatory framework for Hong Kong.

    Based on the Securities and Futures Ordinance (SFO) insider dealing
    provisions and HKEX Main Board Listing Rules regarding disclosure
    of inside information.
    """

    @property
    def name(self) -> str:
        """Return framework identifier."""
        return "SFC"

    @property
    def full_name(self) -> str:
        """Return full name of regulatory body."""
        return "Securities and Futures Commission"

    @property
    def jurisdiction(self) -> str:
        """Return jurisdiction."""
        return "Hong Kong"

    def get_psi_definition(self) -> str:
        """Return official PSI definition."""
        return """
Under the Securities and Futures Ordinance (SFO) Part XIVA, inside information means
specific information that:
1. Is about the corporation, a shareholder/officer of the corporation, or the listed
   securities of the corporation or their derivatives
2. Is not generally known to the persons who are accustomed or would be likely to
   deal in the listed securities of the corporation
3. Would, if generally known, be likely to materially affect the price of the listed
   securities

Listed corporations must disclose inside information as soon as reasonably practicable.
"Safe harbours" exist for delayed disclosure in limited circumstances (incomplete
negotiations, trade secrets, etc.) provided confidentiality is maintained.

The materiality test is whether the information would likely influence persons
accustomed to dealing in the securities.
""".strip()

    def get_psi_criteria(self) -> list[str]:
        """Return PSI classification criteria."""
        return [
            "Changes in financial performance or condition (results, profits, assets)",
            "Changes in expected financial performance previously disclosed",
            "Major transactions (acquisitions, disposals, joint ventures)",
            "Notifiable transactions under Chapter 14 of Listing Rules",
            "Connected transactions under Chapter 14A of Listing Rules",
            "Changes in directors or key executives",
            "Pledging of shares by controlling shareholders",
            "Changes in shareholdings of substantial shareholders",
            "Qualified audit opinion or change in auditors",
            "Litigation or arbitration with material impact",
            "Changes in capital structure or share issues",
            "Material impairments or provisions",
            "Regulatory investigations or sanctions",
            "Material contracts (won or lost)",
            "Changes to business operations or major investments",
        ]

    def get_psi_examples(self) -> dict[str, list[str]]:
        """Return examples of PSI and non-PSI news."""
        return {
            "psi": [
                "Company announces FY profit of HK$2 billion, up 40% YoY",
                "Board recommends final dividend of HK$0.50 per share",
                "Company to acquire mainland property developer for HK$10 billion",
                "Chairman Mr. Wong resigns amid governance concerns",
                "Major shareholder pledges 30% of holdings as loan collateral",
                "SFC commences inquiry into potential market misconduct",
                "Company wins HK$5 billion infrastructure contract",
                "Connected transaction: Company acquires asset from director",
                "Rights issue of HK$1 billion approved by shareholders",
                "Profit warning: Expects H1 profit to decline by more than 50%",
            ],
            "non_psi": [
                "Company participates in Hong Kong Investment Summit",
                "New regional office opened in Guangzhou",
                "Company sponsors Hong Kong Sevens rugby tournament",
                "Management attends investor luncheon (no new information)",
                "Company launches new mobile banking app feature",
                "Annual general meeting held at Hong Kong Convention Centre",
                "Company updates website with new corporate video",
                "Employee volunteer day for local charity",
                "Company receives industry award for best employer",
                "Routine compliance filing submitted to HKEX",
            ],
        }
