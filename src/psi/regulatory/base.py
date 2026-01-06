"""
Abstract base class for regulatory frameworks.

Defines the interface that all regulatory framework implementations must follow.
Each framework provides PSI criteria specific to their jurisdiction.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RegulatoryFramework(ABC):
    """
    Abstract base class for all regulatory frameworks.

    Each regulatory framework (SEBI, SEC, FCA, etc.) must inherit from this
    class and implement all abstract methods to provide PSI classification
    criteria specific to their jurisdiction.

    The framework interface provides:
    - Basic identification (name, full_name, jurisdiction)
    - PSI definition and criteria
    - Examples of PSI and non-PSI news
    - A method to generate full context for LLM prompts
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the framework identifier (e.g., 'SEBI', 'SEC').

        Returns:
            Short name/acronym of the regulatory framework.
        """
        pass

    @property
    @abstractmethod
    def full_name(self) -> str:
        """
        Return the full name of the regulatory body.

        Returns:
            Full official name (e.g., 'Securities and Exchange Board of India').
        """
        pass

    @property
    @abstractmethod
    def jurisdiction(self) -> str:
        """
        Return the jurisdiction of this framework.

        Returns:
            Geographic/political jurisdiction (e.g., 'India', 'United States').
        """
        pass

    @abstractmethod
    def get_psi_definition(self) -> str:
        """
        Return the official PSI definition text.

        This should include the regulatory definition of what constitutes
        Price Sensitive Information under this framework's rules.

        Returns:
            Official PSI definition text from the regulatory body.
        """
        pass

    @abstractmethod
    def get_psi_criteria(self) -> list[str]:
        """
        Return list of PSI classification criteria.

        Each criterion should be a specific condition or category that
        qualifies information as price-sensitive under this framework.

        Returns:
            List of PSI classification criteria strings.
        """
        pass

    @abstractmethod
    def get_psi_examples(self) -> dict[str, list[str]]:
        """
        Return examples of PSI and non-PSI news.

        Returns:
            Dictionary with two keys:
            - 'psi': List of examples that typically constitute PSI
            - 'non_psi': List of examples that typically do NOT constitute PSI
        """
        pass

    def get_prompt_context(self) -> str:
        """
        Generate full context for LLM prompt.

        Combines all framework information into a formatted string suitable
        for inclusion in an LLM prompt for PSI classification.

        Returns:
            Formatted context string containing definition, criteria, and examples.
        """
        logger.debug(f"Generating prompt context for framework: {self.name}")

        # Build criteria list
        criteria = self.get_psi_criteria()
        criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))

        # Build examples
        examples = self.get_psi_examples()
        psi_examples = "\n".join(f"  - {ex}" for ex in examples.get("psi", []))
        non_psi_examples = "\n".join(f"  - {ex}" for ex in examples.get("non_psi", []))

        context = f"""
## Regulatory Framework: {self.name} ({self.full_name})
**Jurisdiction:** {self.jurisdiction}

### PSI Definition under {self.name}:
{self.get_psi_definition()}

### Classification Criteria:
The following categories of information are typically considered Price Sensitive Information under {self.name}:
{criteria_text}

### Examples:

**Typically PSI (Price Sensitive Information):**
{psi_examples}

**Typically NOT PSI:**
{non_psi_examples}
"""

        logger.debug(f"Generated prompt context ({len(context)} chars) for {self.name}")
        return context.strip()

    def __repr__(self) -> str:
        """Return string representation of the framework."""
        return f"{self.__class__.__name__}(name={self.name!r}, jurisdiction={self.jurisdiction!r})"
