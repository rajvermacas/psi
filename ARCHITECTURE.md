# PSI Classifier - Architecture Document

## 1. Overview

The **Price Sensitive Information (PSI) Classifier** is a LangGraph-based system that analyzes stock news and determines whether it constitutes price-sensitive information under various regulatory frameworks.

### 1.1 Problem Statement

Financial markets worldwide are governed by regulations that require timely disclosure of price-sensitive information. Companies and compliance teams need to quickly identify whether a piece of news qualifies as PSI under their applicable regulatory framework.

### 1.2 Solution

An automated classification system that:
- Ingests stock news from CSV files
- Automatically detects the applicable regulatory framework based on region/exchange
- Uses LLM-powered analysis to classify news as PSI or non-PSI
- Provides reasoning for each classification decision
- Outputs results to JSON and console

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PSI CLASSIFIER SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │   INPUT     │                                                            │
│  │   (CSV)     │                                                            │
│  │             │                                                            │
│  │ • date      │                                                            │
│  │ • news      │                                                            │
│  │ • ticker    │                                                            │
│  │ • region    │                                                            │
│  │ • exchange  │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LANGGRAPH STATE GRAPH                            │   │
│  │                                                                     │   │
│  │  ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌──────────────┐  │   │
│  │  │   LOAD   │──▶│  RESOLVE  │──▶│  CLASSIFY  │──▶│   OUTPUT     │  │   │
│  │  │   NODE   │   │   NODE    │   │   NODE     │   │   NODE       │  │   │
│  │  └──────────┘   └───────────┘   └────────────┘   └──────────────┘  │   │
│  │       │               │               │                 │          │   │
│  │       │               │               │                 │          │   │
│  │  Parse CSV      Map region/      Build prompt      Write JSON      │   │
│  │  Validate       exchange to      Call LLM          Print console   │   │
│  │  schema         framework        Parse result      Persist state   │   │
│  │                 Load PSI         Log reasoning                     │   │
│  │                 criteria                                           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐    ┌─────────────┐                                        │
│  │   OUTPUT    │    │   OUTPUT    │                                        │
│  │   (JSON)    │    │  (Console)  │                                        │
│  │             │    │             │                                        │
│  │ • is_psi    │    │ Summary +   │                                        │
│  │ • reasoning │    │ Details     │                                        │
│  │ • metadata  │    │             │                                        │
│  └─────────────┘    └─────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPONENT LAYERS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      PRESENTATION LAYER                              │   │
│  │                                                                      │   │
│  │   main.py (CLI)          io/csv_reader.py       io/json_writer.py   │   │
│  │   • Argument parsing     • CSV ingestion        • JSON output       │   │
│  │   • Orchestration        • Validation           • Console output    │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                      APPLICATION LAYER                               │   │
│  │                                                                      │   │
│  │   graph/workflow.py      graph/nodes.py         graph/state.py      │   │
│  │   • StateGraph builder   • Node implementations • State schema      │   │
│  │   • Edge routing         • Business logic       • Data contracts    │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                       DOMAIN LAYER                                   │   │
│  │                                                                      │   │
│  │   regulatory/base.py     regulatory/registry.py                     │   │
│  │   • Framework interface  • Region/exchange mapping                  │   │
│  │                                                                      │   │
│  │   regulatory/sebi.py     regulatory/sec.py      regulatory/fca.py   │   │
│  │   regulatory/esma.py     regulatory/mas.py      regulatory/sfc.py   │   │
│  │   regulatory/fsa.py                                                  │   │
│  │   • PSI criteria definitions per regulatory body                    │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                    INFRASTRUCTURE LAYER                              │   │
│  │                                                                      │   │
│  │   llm/base.py            llm/factory.py                             │   │
│  │   • Provider interface   • Provider creation                        │   │
│  │                                                                      │   │
│  │   llm/azure.py           llm/openrouter.py      llm/gemini.py       │   │
│  │   • Azure OpenAI         • OpenRouter           • Google Gemini     │   │
│  │                                                                      │   │
│  │   config.py              exceptions.py                              │   │
│  │   • Environment config   • Custom exceptions                        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. LangGraph Pipeline Design

### 3.1 State Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PSIState                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT FIELDS                                                               │
│  ├── csv_path: str                    # Path to input CSV file             │
│  ├── news_items: list[NewsItem]       # Parsed news items                  │
│  │                                                                          │
│  PROCESSING FIELDS                                                          │
│  ├── current_index: int               # Current item being processed       │
│  ├── current_item: NewsItem | None    # Current news item                  │
│  ├── current_framework: str | None    # Resolved framework name            │
│  ├── framework_criteria: str | None   # PSI criteria text                  │
│  │                                                                          │
│  OUTPUT FIELDS                                                              │
│  ├── results: list[PSIResult]         # Classification results             │
│  ├── output_path: str                 # Output JSON file path              │
│  │                                                                          │
│  METADATA FIELDS                                                            │
│  ├── llm_provider: str                # Provider used (azure/openrouter/   │
│  ├── llm_model: str                   #   gemini)                          │
│  └── error: str | None                # Error message if failed            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              NewsItem                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ├── date: str                        # News date                          │
│  ├── news: str                        # News content                       │
│  ├── ticker: str                      # Stock ticker symbol                │
│  ├── region: str                      # Geographic region                  │
│  └── exchange: str                    # Stock exchange                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              PSIResult                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ├── news_item: NewsItem              # Original news item                 │
│  ├── is_psi: bool                     # Classification result              │
│  ├── reasoning: str                   # LLM reasoning                      │
│  ├── regulatory_framework: str        # Framework used                     │
│  ├── classified_at: datetime          # Timestamp                          │
│  ├── llm_provider: str                # Provider used                      │
│  └── llm_model: str                   # Model used                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Node Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NODE FLOW DIAGRAM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              START                                          │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                               │
│                    │      LOAD NODE        │                               │
│                    │                       │                               │
│                    │  Input:               │                               │
│                    │  • csv_path           │                               │
│                    │                       │                               │
│                    │  Actions:             │                               │
│                    │  • Read CSV file      │                               │
│                    │  • Validate columns   │                               │
│                    │  • Parse to NewsItem  │                               │
│                    │  • Set current_index  │                               │
│                    │                       │                               │
│                    │  Output:              │                               │
│                    │  • news_items[]       │                               │
│                    │  • current_index = 0  │                               │
│                    │                       │                               │
│                    │  Errors:              │                               │
│                    │  • CSVValidationError │                               │
│                    └───────────┬───────────┘                               │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                               │
│                    │     RESOLVE NODE      │                               │
│                    │                       │                               │
│                    │  Input:               │                               │
│                    │  • current_item       │                               │
│                    │    (region, exchange) │                               │
│                    │                       │                               │
│                    │  Actions:             │                               │
│                    │  • Lookup framework   │                               │
│                    │  • Load PSI criteria  │                               │
│                    │                       │                               │
│                    │  Output:              │                               │
│                    │  • current_framework  │                               │
│                    │  • framework_criteria │                               │
│                    │                       │                               │
│                    │  Errors:              │                               │
│                    │  • UnknownRegionError │                               │
│                    └───────────┬───────────┘                               │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                               │
│                    │    CLASSIFY NODE      │                               │
│                    │                       │                               │
│                    │  Input:               │                               │
│                    │  • current_item.news  │                               │
│                    │  • framework_criteria │                               │
│                    │                       │                               │
│                    │  Actions:             │                               │
│                    │  • Build LLM prompt   │                               │
│                    │  • Call LLM provider  │                               │
│                    │  • Parse JSON result  │                               │
│                    │  • Create PSIResult   │                               │
│                    │  • Append to results  │                               │
│                    │  • Increment index    │                               │
│                    │                       │                               │
│                    │  Output:              │                               │
│                    │  • results[] updated  │                               │
│                    │  • current_index += 1 │                               │
│                    │                       │                               │
│                    │  Errors:              │                               │
│                    │  • LLMProviderError   │                               │
│                    └───────────┬───────────┘                               │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                               │
│                    │   ROUTING DECISION    │                               │
│                    │                       │                               │
│                    │  current_index <      │──── YES ────┐                 │
│                    │  len(news_items)?     │             │                 │
│                    │                       │             │                 │
│                    └───────────┬───────────┘             │                 │
│                                │ NO                      │                 │
│                                ▼                         │                 │
│                    ┌───────────────────────┐             │                 │
│                    │     OUTPUT NODE       │             │                 │
│                    │                       │             │                 │
│                    │  Input:               │             │                 │
│                    │  • results[]          │             │                 │
│                    │                       │             │                 │
│                    │  Actions:             │             │                 │
│                    │  • Write JSON file    │             │                 │
│                    │  • Print to console   │             │                 │
│                    │  • Log summary        │             │                 │
│                    │  • Persist state      │             │                 │
│                    │                       │             │                 │
│                    │  Output:              │             │                 │
│                    │  • output_path        │             │                 │
│                    └───────────┬───────────┘             │                 │
│                                │                         │                 │
│                                ▼                         │                 │
│                              END                         │                 │
│                                                          │                 │
│                                ┌──────────────────────────┘                 │
│                                │                                            │
│                                ▼                                            │
│                         (Loop back to                                       │
│                          RESOLVE NODE)                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Edge Routing Logic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GRAPH EDGE DEFINITIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENTRY POINT                                                                │
│  ────────────                                                               │
│  START ──────────────────────────────────────────────────────▶ load_node   │
│                                                                             │
│  SEQUENTIAL EDGES                                                           │
│  ────────────────                                                           │
│  load_node ──────────────────────────────────────────────────▶ resolve_node│
│                                                                             │
│  CONDITIONAL EDGES (from classify_node)                                     │
│  ─────────────────────────────────────                                      │
│                                                                             │
│  def should_continue(state: PSIState) -> str:                              │
│      if state.current_index < len(state.news_items):                       │
│          return "resolve"    # Process next item                           │
│      else:                                                                  │
│          return "output"     # All items processed                         │
│                                                                             │
│  classify_node ──┬── "resolve" ─────────────────────────────▶ resolve_node │
│                  │                                                          │
│                  └── "output" ──────────────────────────────▶ output_node  │
│                                                                             │
│  TERMINAL EDGE                                                              │
│  ─────────────                                                              │
│  output_node ────────────────────────────────────────────────▶ END         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Regulatory Framework Design

### 4.1 Framework Registry Mapping

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REGION/EXCHANGE → FRAMEWORK MAPPING                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┬─────────────────┬─────────────────────────────────┐   │
│  │     REGION      │    EXCHANGE     │           FRAMEWORK             │   │
│  ├─────────────────┼─────────────────┼─────────────────────────────────┤   │
│  │                 │                 │                                 │   │
│  │     INDIA       │      NSE        │            SEBI                 │   │
│  │     INDIA       │      BSE        │            SEBI                 │   │
│  │                 │                 │                                 │   │
│  │      US         │     NYSE        │            SEC                  │   │
│  │      US         │    NASDAQ       │            SEC                  │   │
│  │      US         │     AMEX        │            SEC                  │   │
│  │                 │                 │                                 │   │
│  │      UK         │      LSE        │            FCA                  │   │
│  │      UK         │      AIM        │            FCA                  │   │
│  │                 │                 │                                 │   │
│  │      EU         │   EURONEXT      │           ESMA                  │   │
│  │      EU         │   XETRA         │           ESMA                  │   │
│  │   GERMANY       │   XETRA         │           ESMA                  │   │
│  │   FRANCE        │   EURONEXT      │           ESMA                  │   │
│  │  NETHERLANDS    │   EURONEXT      │           ESMA                  │   │
│  │                 │                 │                                 │   │
│  │   SINGAPORE     │      SGX        │            MAS                  │   │
│  │                 │                 │                                 │   │
│  │   HONG KONG     │     HKEX        │            SFC                  │   │
│  │   HONG KONG     │      HKG        │            SFC                  │   │
│  │                 │                 │                                 │   │
│  │    JAPAN        │      TSE        │            FSA                  │   │
│  │    JAPAN        │      JPX        │            FSA                  │   │
│  │    JAPAN        │     JASDAQ      │            FSA                  │   │
│  │                 │                 │                                 │   │
│  └─────────────────┴─────────────────┴─────────────────────────────────┘   │
│                                                                             │
│  UNKNOWN REGION/EXCHANGE → Raise UnknownRegionError (Fail-Fast)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Framework Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     REGULATORY FRAMEWORK INTERFACE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  class RegulatoryFramework(ABC):                                           │
│      """Abstract base class for all regulatory frameworks"""               │
│                                                                             │
│      @property                                                              │
│      @abstractmethod                                                        │
│      def name(self) -> str:                                                │
│          """Return framework identifier (e.g., 'SEBI', 'SEC')"""           │
│                                                                             │
│      @property                                                              │
│      @abstractmethod                                                        │
│      def full_name(self) -> str:                                           │
│          """Return full name (e.g., 'Securities and Exchange Board')"""    │
│                                                                             │
│      @property                                                              │
│      @abstractmethod                                                        │
│      def jurisdiction(self) -> str:                                        │
│          """Return jurisdiction (e.g., 'India', 'United States')"""        │
│                                                                             │
│      @abstractmethod                                                        │
│      def get_psi_definition(self) -> str:                                  │
│          """Return the official PSI definition text"""                     │
│                                                                             │
│      @abstractmethod                                                        │
│      def get_psi_criteria(self) -> list[str]:                              │
│          """Return list of PSI classification criteria"""                  │
│                                                                             │
│      @abstractmethod                                                        │
│      def get_psi_examples(self) -> dict[str, list[str]]:                   │
│          """Return examples: {'psi': [...], 'non_psi': [...]}"""           │
│                                                                             │
│      def get_prompt_context(self) -> str:                                  │
│          """Generate full context for LLM prompt (combines all above)"""   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Framework Implementations Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FRAMEWORK IMPLEMENTATIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SEBI (India)                                                               │
│  ─────────────                                                              │
│  • Regulation: SEBI (Listing Obligations and Disclosure Requirements)      │
│  • Key PSI Categories:                                                      │
│    - Financial results (quarterly/annual)                                  │
│    - Dividends declared                                                    │
│    - Change in capital structure                                           │
│    - Mergers, acquisitions, demergers                                      │
│    - Changes in key personnel                                              │
│    - Material contracts                                                    │
│    - Litigation/disputes                                                   │
│                                                                             │
│  SEC (United States)                                                        │
│  ───────────────────                                                        │
│  • Regulation: Regulation FD, Form 8-K, 10-K requirements                  │
│  • Key PSI Categories:                                                      │
│    - Material earnings information                                         │
│    - M&A activities                                                        │
│    - Changes in control                                                    │
│    - Bankruptcy/receivership                                               │
│    - Delisting notices                                                     │
│    - Executive changes                                                     │
│                                                                             │
│  FCA (United Kingdom)                                                       │
│  ─────────────────────                                                      │
│  • Regulation: Market Abuse Regulation (UK MAR), DTR                       │
│  • Key PSI Categories:                                                      │
│    - Inside information (as defined by UK MAR)                             │
│    - Significant transactions                                              │
│    - Changes to board composition                                          │
│    - Profit warnings                                                       │
│                                                                             │
│  ESMA (European Union)                                                      │
│  ──────────────────────                                                     │
│  • Regulation: Market Abuse Regulation (EU MAR)                            │
│  • Key PSI Categories:                                                      │
│    - Inside information under Article 7 MAR                                │
│    - Ad-hoc disclosures                                                    │
│    - Manager transactions                                                  │
│                                                                             │
│  MAS (Singapore)                                                            │
│  ───────────────                                                            │
│  • Regulation: Securities and Futures Act, SGX Listing Rules               │
│  • Key PSI Categories:                                                      │
│    - Material information                                                  │
│    - Interested person transactions                                        │
│    - Take-over/merger situations                                           │
│                                                                             │
│  SFC (Hong Kong)                                                            │
│  ───────────────                                                            │
│  • Regulation: Securities and Futures Ordinance, HKEX Listing Rules        │
│  • Key PSI Categories:                                                      │
│    - Price sensitive information                                           │
│    - Inside information                                                    │
│    - Notifiable transactions                                               │
│                                                                             │
│  FSA (Japan)                                                                │
│  ───────────                                                                │
│  • Regulation: Financial Instruments and Exchange Act                      │
│  • Key PSI Categories:                                                      │
│    - Material facts (Juyo Jijitsu)                                         │
│    - Timely disclosure items                                               │
│    - Corporate action information                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. LLM Provider Design

### 5.1 Provider Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM PROVIDER ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌─────────────────────┐                             │
│                        │    LLMFactory       │                             │
│                        │                     │                             │
│                        │  create(provider)   │                             │
│                        └──────────┬──────────┘                             │
│                                   │                                         │
│               ┌───────────────────┼───────────────────┐                    │
│               │                   │                   │                    │
│               ▼                   ▼                   ▼                    │
│    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐         │
│    │ AzureOpenAI      │ │ OpenRouter       │ │ Gemini           │         │
│    │ Provider         │ │ Provider         │ │ Provider         │         │
│    │                  │ │                  │ │                  │         │
│    │ • endpoint       │ │ • api_key        │ │ • api_key        │         │
│    │ • api_key        │ │ • model          │ │ • model          │         │
│    │ • api_version    │ │                  │ │                  │         │
│    │ • deployment     │ │                  │ │                  │         │
│    └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘         │
│             │                    │                    │                    │
│             └────────────────────┼────────────────────┘                    │
│                                  │                                         │
│                                  ▼                                         │
│                    ┌─────────────────────────┐                             │
│                    │   BaseLLMProvider       │                             │
│                    │   (Abstract Interface)  │                             │
│                    │                         │                             │
│                    │   + classify(           │                             │
│                    │       news: str,        │                             │
│                    │       context: str      │                             │
│                    │     ) -> PSIClassification                            │
│                    │                         │                             │
│                    │   + validate_config()   │                             │
│                    │                         │                             │
│                    └─────────────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 LLM Prompt Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM PROMPT TEMPLATE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SYSTEM PROMPT:                                                             │
│  ──────────────                                                             │
│  You are a financial regulatory compliance expert specializing in           │
│  {framework.name} ({framework.full_name}) regulations for                   │
│  {framework.jurisdiction}.                                                  │
│                                                                             │
│  Your task is to determine whether a given piece of news constitutes        │
│  Price Sensitive Information (PSI) under {framework.name} regulations.      │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  USER PROMPT:                                                               │
│  ────────────                                                               │
│  ## Regulatory Context                                                      │
│                                                                             │
│  ### PSI Definition under {framework.name}:                                 │
│  {framework.get_psi_definition()}                                           │
│                                                                             │
│  ### Classification Criteria:                                               │
│  {framework.get_psi_criteria()}                                             │
│                                                                             │
│  ### Examples:                                                              │
│  **Typically PSI:**                                                         │
│  {framework.get_psi_examples()['psi']}                                      │
│                                                                             │
│  **Typically NOT PSI:**                                                     │
│  {framework.get_psi_examples()['non_psi']}                                  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ## News to Classify                                                        │
│                                                                             │
│  **Date:** {news_item.date}                                                 │
│  **Ticker:** {news_item.ticker}                                             │
│  **Exchange:** {news_item.exchange}                                         │
│  **Region:** {news_item.region}                                             │
│                                                                             │
│  **News Content:**                                                          │
│  {news_item.news}                                                           │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ## Instructions                                                            │
│                                                                             │
│  Analyze the news content against {framework.name} PSI criteria and         │
│  provide your classification.                                               │
│                                                                             │
│  You MUST respond with a valid JSON object in this exact format:            │
│  {                                                                          │
│    "is_psi": <true or false>,                                               │
│    "reasoning": "<detailed explanation referencing specific criteria>"      │
│  }                                                                          │
│                                                                             │
│  Do not include any text outside the JSON object.                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Project Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROJECT DIRECTORY TREE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  /workspaces/psi/                                                           │
│  │                                                                          │
│  ├── .env.example                 # Environment template                    │
│  ├── .gitignore                   # Git ignore rules                        │
│  ├── pyproject.toml               # Project config & dependencies           │
│  ├── README.md                    # User documentation                      │
│  ├── ARCHITECTURE.md              # This document                           │
│  │                                                                          │
│  ├── src/psi/                     # Main package (PyPI-ready)               │
│  │   ├── __init__.py                                                        │
│  │   ├── config.py                # Configuration loader (.env)             │
│  │   ├── exceptions.py            # Custom exception classes                │
│  │   ├── main.py                  # CLI entry point                         │
│  │   │                                                                      │
│  │   ├── graph/                   # LangGraph pipeline                      │
│  │   │   ├── __init__.py                                                    │
│  │   │   ├── state.py             # Pydantic state schema                   │
│  │   │   ├── nodes.py             # 4 node implementations                  │
│  │   │   └── workflow.py          # StateGraph builder                      │
│  │   │                                                                      │
│  │   ├── regulatory/              # Regulatory frameworks                   │
│  │   │   ├── __init__.py                                                    │
│  │   │   ├── base.py              # Abstract framework class                │
│  │   │   ├── registry.py          # Framework registry                      │
│  │   │   ├── sebi.py              # SEBI (India)                            │
│  │   │   ├── sec.py               # SEC (US)                                │
│  │   │   ├── fca.py               # FCA (UK)                                │
│  │   │   ├── esma.py              # ESMA (EU)                               │
│  │   │   ├── mas.py               # MAS (Singapore)                         │
│  │   │   ├── sfc.py               # SFC (Hong Kong)                         │
│  │   │   └── fsa.py               # FSA (Japan)                             │
│  │   │                                                                      │
│  │   ├── llm/                     # LLM providers                           │
│  │   │   ├── __init__.py                                                    │
│  │   │   ├── base.py              # Provider interface                      │
│  │   │   ├── factory.py           # Provider factory                        │
│  │   │   ├── azure.py             # Azure OpenAI                            │
│  │   │   ├── openrouter.py        # OpenRouter                              │
│  │   │   └── gemini.py            # Google Gemini                           │
│  │   │                                                                      │
│  │   └── io/                      # Input/Output handlers                   │
│  │       ├── __init__.py                                                    │
│  │       ├── csv_reader.py        # CSV parser                              │
│  │       └── json_writer.py       # JSON output                             │
│  │                                                                          │
│  ├── tests/                       # Test suite                              │
│  │   ├── __init__.py                                                        │
│  │   ├── conftest.py              # Pytest fixtures                         │
│  │   ├── test_config.py                                                     │
│  │   ├── test_graph/                                                        │
│  │   │   ├── test_state.py                                                  │
│  │   │   ├── test_nodes.py                                                  │
│  │   │   └── test_workflow.py                                               │
│  │   ├── test_regulatory/                                                   │
│  │   │   ├── test_registry.py                                               │
│  │   │   └── test_frameworks.py                                             │
│  │   ├── test_llm/                                                          │
│  │   │   ├── test_factory.py                                                │
│  │   │   └── test_providers.py                                              │
│  │   └── test_io/                                                           │
│  │       ├── test_csv_reader.py                                             │
│  │       └── test_json_writer.py                                            │
│  │                                                                          │
│  ├── test_data/                   # Test fixtures                           │
│  │   ├── sample_news.csv          # Valid sample data                       │
│  │   ├── invalid_region.csv       # Error case: unknown region              │
│  │   └── edge_cases.csv           # Edge case scenarios                     │
│  │                                                                          │
│  ├── scripts/                     # Utility scripts                         │
│  │   ├── generate_test_data.py    # Generate synthetic test data            │
│  │   └── validate_output.py       # Validate JSON output                    │
│  │                                                                          │
│  └── resources/                   # Runtime resources                       │
│      └── reports/                 # Output directory                        │
│          └── .gitkeep                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Data Flow

### 7.1 End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          END-TO-END DATA FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. USER INVOCATION                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  $ python -m psi.main --input data/news.csv --output reports/results.json  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  2. CLI PARSING (main.py)                                                   │
│  ────────────────────────                                                   │
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │ argparse            │                                                    │
│  │ --input: str        │──▶ input_path = "data/news.csv"                   │
│  │ --output: str       │──▶ output_path = "reports/results.json"           │
│  └─────────────────────┘                                                    │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  3. CONFIGURATION LOADING (config.py)                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────────────────┐    │
│  │ .env file           │───▶│ Config dataclass                        │    │
│  │                     │    │                                         │    │
│  │ LLM_PROVIDER=azure  │    │ llm_provider: "azure"                   │    │
│  │ AZURE_ENDPOINT=...  │    │ azure_endpoint: "https://..."           │    │
│  │ AZURE_API_KEY=...   │    │ azure_api_key: "sk-..."                 │    │
│  │ MODEL_NAME=gpt-4    │    │ model_name: "gpt-4"                     │    │
│  └─────────────────────┘    └─────────────────────────────────────────┘    │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  4. CSV INGESTION (io/csv_reader.py)                                        │
│  ───────────────────────────────────                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ news.csv                                                            │   │
│  │                                                                     │   │
│  │ date,news,ticker,region,exchange                                    │   │
│  │ 2024-01-15,"Company X announces Q4 earnings...",COMP,India,NSE      │   │
│  │ 2024-01-16,"CEO attends industry conference...",COMP,US,NYSE        │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ list[NewsItem]                                                      │   │
│  │                                                                     │   │
│  │ [                                                                   │   │
│  │   NewsItem(date="2024-01-15", news="Company X...",                  │   │
│  │            ticker="COMP", region="India", exchange="NSE"),          │   │
│  │   NewsItem(date="2024-01-16", news="CEO attends...",                │   │
│  │            ticker="COMP", region="US", exchange="NYSE"),            │   │
│  │ ]                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  5. GRAPH EXECUTION (graph/workflow.py)                                     │
│  ──────────────────────────────────────                                     │
│                                                                             │
│  For each NewsItem (sequential):                                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  NewsItem[0]                                                        │   │
│  │  region="India", exchange="NSE"                                     │   │
│  │         │                                                           │   │
│  │         ▼                                                           │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │ Registry Lookup │──▶ framework = SEBIFramework()                │   │
│  │  └─────────────────┘                                                │   │
│  │         │                                                           │   │
│  │         ▼                                                           │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │ Get PSI Context │──▶ criteria = "SEBI LODR Regulation 30..."    │   │
│  │  └─────────────────┘                                                │   │
│  │         │                                                           │   │
│  │         ▼                                                           │   │
│  │  ┌─────────────────┐    ┌────────────────────────────────────────┐ │   │
│  │  │ LLM Classification│──▶│ {"is_psi": true,                      │ │   │
│  │  │                 │    │  "reasoning": "Quarterly earnings..."}  │ │   │
│  │  └─────────────────┘    └────────────────────────────────────────┘ │   │
│  │         │                                                           │   │
│  │         ▼                                                           │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ PSIResult(                                                  │   │   │
│  │  │   news_item=NewsItem[0],                                    │   │   │
│  │  │   is_psi=True,                                              │   │   │
│  │  │   reasoning="Quarterly earnings announcement constitutes...",│   │   │
│  │  │   regulatory_framework="SEBI",                              │   │   │
│  │  │   classified_at="2024-01-20T10:30:00",                      │   │   │
│  │  │   llm_provider="azure",                                     │   │   │
│  │  │   llm_model="gpt-4"                                         │   │   │
│  │  │ )                                                           │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  6. OUTPUT GENERATION                                                       │
│  ────────────────────                                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ JSON Output (reports/results.json)                                  │   │
│  │                                                                     │   │
│  │ {                                                                   │   │
│  │   "metadata": {                                                     │   │
│  │     "generated_at": "2024-01-20T10:35:00",                         │   │
│  │     "total_items": 2,                                               │   │
│  │     "psi_count": 1,                                                 │   │
│  │     "non_psi_count": 1,                                             │   │
│  │     "llm_provider": "azure",                                        │   │
│  │     "llm_model": "gpt-4"                                            │   │
│  │   },                                                                │   │
│  │   "results": [                                                      │   │
│  │     {                                                               │   │
│  │       "news_item": {...},                                           │   │
│  │       "is_psi": true,                                               │   │
│  │       "reasoning": "...",                                           │   │
│  │       "regulatory_framework": "SEBI",                               │   │
│  │       "classified_at": "2024-01-20T10:30:00"                        │   │
│  │     },                                                              │   │
│  │     ...                                                             │   │
│  │   ]                                                                 │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Console Output                                                      │   │
│  │                                                                     │   │
│  │ ═══════════════════════════════════════════════════════════════════ │   │
│  │                    PSI CLASSIFICATION RESULTS                       │   │
│  │ ═══════════════════════════════════════════════════════════════════ │   │
│  │                                                                     │   │
│  │ [1/2] COMP (NSE/India) - 2024-01-15                                │   │
│  │       Framework: SEBI                                               │   │
│  │       PSI: YES                                                      │   │
│  │       Reasoning: Quarterly earnings announcement constitutes...     │   │
│  │                                                                     │   │
│  │ [2/2] COMP (NYSE/US) - 2024-01-16                                  │   │
│  │       Framework: SEC                                                │   │
│  │       PSI: NO                                                       │   │
│  │       Reasoning: Conference attendance is routine and does not...   │   │
│  │                                                                     │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ SUMMARY: 2 items processed | 1 PSI | 1 Non-PSI                     │   │
│  │ Output saved to: reports/results.json                              │   │
│  │ ═══════════════════════════════════════════════════════════════════ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Error Handling

### 8.1 Exception Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXCEPTION HIERARCHY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         PSIException (Base)                                 │
│                                │                                            │
│        ┌───────────────────────┼───────────────────────┐                   │
│        │                       │                       │                   │
│        ▼                       ▼                       ▼                   │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────┐          │
│  │ ConfigError   │    │ ValidationError│    │ ProcessingError   │          │
│  │               │    │               │    │                   │          │
│  │ • Missing env │    │ • CSV format  │    │ • LLM failure     │          │
│  │ • Invalid     │    │ • Missing     │    │ • Unknown region  │          │
│  │   provider    │    │   columns     │    │ • Parse error     │          │
│  └───────────────┘    └───────────────┘    └───────────────────┘          │
│                                                                             │
│  Specific Exceptions:                                                       │
│  ────────────────────                                                       │
│                                                                             │
│  • InvalidConfigError        - Missing or invalid .env configuration       │
│  • CSVValidationError        - Malformed CSV or missing required columns   │
│  • UnknownRegionError        - Region/exchange not in registry             │
│  • LLMProviderError          - LLM API call failure                        │
│  • LLMResponseParseError     - Cannot parse LLM JSON response              │
│  • StateValidationError      - Invalid state transition                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Error Handling Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FAIL-FAST ERROR HANDLING                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRINCIPLE: No fallbacks, no defaults. Fail immediately on errors.         │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  CONFIG VALIDATION (startup)                                                │
│  ───────────────────────────                                                │
│                                                                             │
│  if not os.getenv("LLM_PROVIDER"):                                         │
│      raise InvalidConfigError(                                              │
│          "LLM_PROVIDER environment variable is required. "                 │
│          "Set it to one of: azure, openrouter, gemini"                     │
│      )                                                                      │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  REGION/EXCHANGE LOOKUP (resolve node)                                      │
│  ─────────────────────────────────────                                      │
│                                                                             │
│  framework = registry.get_framework(region, exchange)                       │
│  if framework is None:                                                      │
│      raise UnknownRegionError(                                              │
│          f"No regulatory framework found for region='{region}', "          │
│          f"exchange='{exchange}'. Supported combinations: {registry.list()}"│
│      )                                                                      │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  LLM RESPONSE PARSING (classify node)                                       │
│  ────────────────────────────────────                                       │
│                                                                             │
│  try:                                                                       │
│      result = json.loads(llm_response)                                      │
│      if "is_psi" not in result or "reasoning" not in result:               │
│          raise LLMResponseParseError(                                       │
│              f"LLM response missing required fields. "                     │
│              f"Expected: {{'is_psi': bool, 'reasoning': str}}. "           │
│              f"Got: {result}"                                               │
│          )                                                                  │
│  except json.JSONDecodeError as e:                                          │
│      raise LLMResponseParseError(                                           │
│          f"LLM response is not valid JSON: {llm_response[:200]}... "       │
│          f"Error: {e}"                                                      │
│      )                                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Configuration

### 9.1 Environment Variables

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ENVIRONMENT CONFIGURATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  # .env.example                                                             │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  # LLM Provider Selection (REQUIRED)                                        │
│  # Options: azure, openrouter, gemini                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  LLM_PROVIDER=azure                                                         │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  # Azure OpenAI Configuration                                               │
│  # Required if LLM_PROVIDER=azure                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/              │
│  AZURE_OPENAI_API_KEY=your-api-key-here                                     │
│  AZURE_OPENAI_API_VERSION=2024-02-15-preview                                │
│  AZURE_OPENAI_DEPLOYMENT=gpt-4                                              │
│  AZURE_OPENAI_MODEL=gpt-4                                                   │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  # OpenRouter Configuration                                                 │
│  # Required if LLM_PROVIDER=openrouter                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  OPENROUTER_API_KEY=your-api-key-here                                       │
│  OPENROUTER_MODEL=anthropic/claude-3-sonnet                                 │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  # Google Gemini Configuration                                              │
│  # Required if LLM_PROVIDER=gemini                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  GEMINI_API_KEY=your-api-key-here                                           │
│  GEMINI_MODEL=gemini-1.5-pro                                                │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  # Logging Configuration                                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  LOG_LEVEL=INFO                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Testing Strategy

### 10.1 Test Coverage Plan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEST COVERAGE PLAN                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  UNIT TESTS                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Module                    │ Test File              │ Coverage Target │   │
│  ├───────────────────────────┼────────────────────────┼─────────────────┤   │
│  │ config.py                 │ test_config.py         │ 100%            │   │
│  │ exceptions.py             │ test_exceptions.py     │ 100%            │   │
│  │ graph/state.py            │ test_state.py          │ 100%            │   │
│  │ graph/nodes.py            │ test_nodes.py          │ 90%             │   │
│  │ regulatory/registry.py    │ test_registry.py       │ 100%            │   │
│  │ regulatory/*.py           │ test_frameworks.py     │ 90%             │   │
│  │ llm/factory.py            │ test_factory.py        │ 100%            │   │
│  │ llm/*.py                  │ test_providers.py      │ 80% (mocked)    │   │
│  │ io/csv_reader.py          │ test_csv_reader.py     │ 100%            │   │
│  │ io/json_writer.py         │ test_json_writer.py    │ 100%            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  INTEGRATION TESTS                                                          │
│  ─────────────────                                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Test Scenario                                      │ Test File       │   │
│  ├────────────────────────────────────────────────────┼─────────────────┤   │
│  │ Full pipeline with mocked LLM                      │ test_workflow.py│   │
│  │ CSV → Graph → JSON with all frameworks             │ test_e2e.py     │   │
│  │ Error handling (unknown region)                    │ test_errors.py  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TEST DATA                                                                  │
│  ─────────                                                                  │
│                                                                             │
│  test_data/                                                                 │
│  ├── sample_news.csv          # Valid multi-region data (10 items)         │
│  ├── invalid_region.csv       # Contains unknown region                    │
│  ├── invalid_format.csv       # Missing columns                            │
│  ├── edge_cases.csv           # Empty news, special characters             │
│  └── expected_output.json     # Expected output for validation             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Mock Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             MOCK STRATEGY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LLM PROVIDER MOCKING                                                       │
│  ────────────────────                                                       │
│                                                                             │
│  @pytest.fixture                                                            │
│  def mock_llm_provider():                                                   │
│      """Mock LLM provider for deterministic testing"""                     │
│      provider = Mock(spec=BaseLLMProvider)                                  │
│      provider.classify.return_value = {                                     │
│          "is_psi": True,                                                    │
│          "reasoning": "Test reasoning for PSI classification"              │
│      }                                                                      │
│      return provider                                                        │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  CONFIGURATION MOCKING                                                      │
│  ─────────────────────                                                      │
│                                                                             │
│  @pytest.fixture                                                            │
│  def mock_config(monkeypatch):                                              │
│      """Set up test environment variables"""                               │
│      monkeypatch.setenv("LLM_PROVIDER", "azure")                           │
│      monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.com")│
│      monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")                │
│      monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4")                     │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  FILE SYSTEM MOCKING                                                        │
│  ───────────────────                                                        │
│                                                                             │
│  @pytest.fixture                                                            │
│  def temp_csv(tmp_path):                                                    │
│      """Create temporary CSV file for testing"""                           │
│      csv_file = tmp_path / "test_news.csv"                                 │
│      csv_file.write_text(                                                   │
│          "date,news,ticker,region,exchange\n"                              │
│          "2024-01-15,Test news,TEST,India,NSE\n"                           │
│      )                                                                      │
│      return csv_file                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROJECT DEPENDENCIES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  # pyproject.toml                                                           │
│                                                                             │
│  [project]                                                                  │
│  name = "psi-classifier"                                                    │
│  version = "0.1.0"                                                          │
│  description = "LangGraph-based Price Sensitive Information Classifier"    │
│  requires-python = ">=3.12"                                                 │
│                                                                             │
│  dependencies = [                                                           │
│      # LangGraph & LangChain                                                │
│      "langgraph>=0.2.0",                                                    │
│      "langchain-core>=0.3.0",                                               │
│      "langchain-openai>=0.2.0",        # Azure OpenAI support              │
│      "langchain-google-genai>=2.0.0",  # Gemini support                    │
│                                                                             │
│      # Data validation                                                      │
│      "pydantic>=2.9.0",                                                     │
│      "pydantic-settings>=2.0.0",                                            │
│                                                                             │
│      # Data processing                                                      │
│      "pandas>=2.2.0",                                                       │
│                                                                             │
│      # Configuration                                                        │
│      "python-dotenv>=1.0.0",                                                │
│                                                                             │
│      # HTTP client (for OpenRouter)                                         │
│      "httpx>=0.27.0",                                                       │
│  ]                                                                          │
│                                                                             │
│  [project.optional-dependencies]                                            │
│  dev = [                                                                    │
│      "pytest>=8.0.0",                                                       │
│      "pytest-cov>=4.1.0",                                                   │
│      "pytest-mock>=3.14.0",                                                 │
│      "pytest-asyncio>=0.23.0",                                              │
│  ]                                                                          │
│                                                                             │
│  [build-system]                                                             │
│  requires = ["hatchling"]                                                   │
│  build-backend = "hatchling.build"                                          │
│                                                                             │
│  [tool.hatch.build.targets.wheel]                                           │
│  packages = ["src/psi"]                                                     │
│                                                                             │
│  [tool.pytest.ini_options]                                                  │
│  testpaths = ["tests"]                                                      │
│  python_files = "test_*.py"                                                 │
│  python_functions = "test_*"                                                │
│  addopts = "-v --cov=src/psi --cov-report=term-missing"                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Implementation Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION PHASES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: FOUNDATION                                                        │
│  ───────────────────                                                        │
│  □ Create project structure (pyproject.toml, .gitignore)                   │
│  □ Implement exceptions.py                                                  │
│  □ Implement config.py (.env loading)                                       │
│  □ Create .env.example                                                      │
│  □ Write tests for config and exceptions                                   │
│                                                                             │
│  PHASE 2: REGULATORY LAYER                                                  │
│  ─────────────────────────                                                  │
│  □ Implement regulatory/base.py (abstract class)                           │
│  □ Implement regulatory/registry.py (mapping)                              │
│  □ Implement 7 framework files (sebi, sec, fca, esma, mas, sfc, fsa)       │
│  □ Write tests for registry and frameworks                                 │
│                                                                             │
│  PHASE 3: LLM LAYER                                                         │
│  ───────────────────                                                        │
│  □ Implement llm/base.py (provider interface)                              │
│  □ Implement llm/factory.py                                                │
│  □ Implement llm/azure.py                                                  │
│  □ Implement llm/openrouter.py                                             │
│  □ Implement llm/gemini.py                                                 │
│  □ Write tests for LLM layer (mocked)                                      │
│                                                                             │
│  PHASE 4: GRAPH LAYER                                                       │
│  ────────────────────                                                       │
│  □ Implement graph/state.py (Pydantic models)                              │
│  □ Implement graph/nodes.py (4 nodes)                                      │
│  □ Implement graph/workflow.py (StateGraph)                                │
│  □ Write tests for graph layer                                             │
│                                                                             │
│  PHASE 5: I/O LAYER                                                         │
│  ──────────────────                                                         │
│  □ Implement io/csv_reader.py                                              │
│  □ Implement io/json_writer.py                                             │
│  □ Write tests for I/O layer                                               │
│                                                                             │
│  PHASE 6: CLI & INTEGRATION                                                 │
│  ──────────────────────────                                                 │
│  □ Implement main.py (CLI entry point)                                     │
│  □ Create test data (sample_news.csv)                                      │
│  □ Write integration tests                                                  │
│  □ End-to-end testing                                                      │
│                                                                             │
│  PHASE 7: DOCUMENTATION                                                     │
│  ──────────────────────                                                     │
│  □ Update README.md with usage instructions                                │
│  □ Create scripts/generate_test_data.py                                    │
│  □ Final review and cleanup                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Logging Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOGGING STRATEGY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LOG LEVELS                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  DEBUG  : Detailed information for debugging                               │
│           - Full state dumps                                               │
│           - LLM prompts and responses                                      │
│           - Framework criteria loaded                                      │
│                                                                             │
│  INFO   : General operational information                                  │
│           - Processing progress (item X of Y)                              │
│           - Framework resolved                                             │
│           - Classification result                                          │
│           - File operations                                                │
│                                                                             │
│  WARNING: Potential issues (not used in fail-fast mode)                    │
│                                                                             │
│  ERROR  : Errors before raising exceptions                                 │
│           - Config validation failures                                     │
│           - Unknown region/exchange                                        │
│           - LLM API errors                                                 │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  LOG FORMAT                                                                 │
│  ──────────                                                                 │
│                                                                             │
│  %(asctime)s | %(levelname)-8s | %(name)s | %(message)s                    │
│                                                                             │
│  Example:                                                                   │
│  2024-01-20 10:30:15 | INFO     | psi.graph.nodes | Processing item 1/10   │
│  2024-01-20 10:30:15 | INFO     | psi.graph.nodes | Resolved framework: SEBI│
│  2024-01-20 10:30:16 | DEBUG    | psi.llm.azure | Sending prompt (1500 chars)│
│  2024-01-20 10:30:18 | INFO     | psi.graph.nodes | Classification: PSI=True│
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  LOG LOCATIONS                                                              │
│  ─────────────                                                              │
│                                                                             │
│  • main.py           : CLI invocation, overall progress                    │
│  • graph/nodes.py    : Node entry/exit, state transitions                  │
│  • llm/*.py          : API calls, responses, timing                        │
│  • io/csv_reader.py  : File reading, validation results                    │
│  • io/json_writer.py : File writing, output summary                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. State Persistence

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STATE PERSISTENCE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LANGGRAPH CHECKPOINTING                                                    │
│  ───────────────────────                                                    │
│                                                                             │
│  LangGraph provides built-in state persistence via checkpointers:          │
│                                                                             │
│  from langgraph.checkpoint.sqlite import SqliteSaver                        │
│                                                                             │
│  # Create checkpointer                                                      │
│  checkpointer = SqliteSaver.from_conn_string(                              │
│      "resources/state/psi_state.db"                                        │
│  )                                                                          │
│                                                                             │
│  # Compile graph with checkpointer                                          │
│  workflow = graph.compile(checkpointer=checkpointer)                        │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  AUDIT TRAIL                                                                │
│  ───────────                                                                │
│                                                                             │
│  Each graph invocation is assigned a thread_id for tracking:               │
│                                                                             │
│  config = {"configurable": {"thread_id": f"psi-{timestamp}"}}              │
│  result = workflow.invoke(initial_state, config)                           │
│                                                                             │
│  This enables:                                                              │
│  • Replay of past classifications                                          │
│  • Debugging failed runs                                                   │
│  • Audit compliance                                                        │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  STATE DATABASE SCHEMA (SQLite)                                             │
│  ──────────────────────────────                                             │
│                                                                             │
│  checkpoints                                                                │
│  ├── thread_id: TEXT           # Unique run identifier                     │
│  ├── checkpoint_id: TEXT       # State version                             │
│  ├── parent_id: TEXT           # Previous state                            │
│  ├── checkpoint: BLOB          # Serialized state                          │
│  └── metadata: TEXT            # Node info, timestamps                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Security Considerations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY CONSIDERATIONS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  API KEY PROTECTION                                                         │
│  ───────────────────                                                        │
│  • API keys stored in .env (never committed)                               │
│  • .gitignore includes .env                                                │
│  • Keys never logged (even at DEBUG level)                                 │
│  • Keys never included in error messages                                   │
│                                                                             │
│  INPUT VALIDATION                                                           │
│  ────────────────                                                           │
│  • CSV columns strictly validated                                          │
│  • News content sanitized before LLM prompt                                │
│  • File paths validated (no directory traversal)                           │
│                                                                             │
│  OUTPUT SANITIZATION                                                        │
│  ───────────────────                                                        │
│  • LLM responses parsed as JSON (not eval'd)                               │
│  • Special characters escaped in output                                    │
│  • File paths canonicalized                                                │
│                                                                             │
│  DEPENDENCY SECURITY                                                        │
│  ───────────────────                                                        │
│  • Use uv for dependency management                                        │
│  • Pin dependency versions in pyproject.toml                               │
│  • Regular dependency audits                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 16. Future Enhancements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FUTURE ENHANCEMENTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  POST-POC IMPROVEMENTS                                                      │
│  ─────────────────────                                                      │
│                                                                             │
│  1. PARALLEL PROCESSING                                                     │
│     • Add async support for concurrent LLM calls                           │
│     • Implement rate limiting per provider                                 │
│     • Batch processing with configurable concurrency                       │
│                                                                             │
│  2. ADDITIONAL LLM PROVIDERS                                                │
│     • AWS Bedrock (Claude, Titan)                                          │
│     • Local models via Ollama                                              │
│     • Anthropic direct API                                                 │
│                                                                             │
│  3. ADDITIONAL REGULATORY FRAMEWORKS                                        │
│     • Australia (ASX/ASIC)                                                 │
│     • Canada (TSX/CSA)                                                     │
│     • Brazil (B3/CVM)                                                      │
│     • South Korea (KRX/FSC)                                                │
│                                                                             │
│  4. CONFIDENCE SCORING                                                      │
│     • Add confidence threshold configuration                               │
│     • Flag low-confidence classifications for review                       │
│     • Multi-model voting for high-stakes decisions                         │
│                                                                             │
│  5. WEB UI                                                                  │
│     • Streamlit or Gradio interface                                        │
│     • Real-time classification                                             │
│     • Result visualization                                                 │
│                                                                             │
│  6. API SERVICE                                                             │
│     • FastAPI REST endpoint                                                │
│     • Webhook support for async processing                                 │
│     • Authentication and rate limiting                                     │
│                                                                             │
│  7. MONITORING & OBSERVABILITY                                              │
│     • LangSmith integration for tracing                                    │
│     • Prometheus metrics                                                   │
│     • Classification accuracy tracking                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 17. Glossary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GLOSSARY                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PSI     Price Sensitive Information - Information that could materially   │
│          affect a company's stock price if made public                     │
│                                                                             │
│  SEBI    Securities and Exchange Board of India                            │
│  SEC     Securities and Exchange Commission (US)                           │
│  FCA     Financial Conduct Authority (UK)                                  │
│  ESMA    European Securities and Markets Authority                         │
│  MAS     Monetary Authority of Singapore                                   │
│  SFC     Securities and Futures Commission (Hong Kong)                     │
│  FSA     Financial Services Agency (Japan)                                 │
│                                                                             │
│  LLM     Large Language Model                                              │
│  LangGraph  Framework for building stateful LLM applications               │
│  StateGraph LangGraph's core abstraction for defining workflows            │
│                                                                             │
│  NSE     National Stock Exchange (India)                                   │
│  BSE     Bombay Stock Exchange (India)                                     │
│  NYSE    New York Stock Exchange                                           │
│  NASDAQ  National Association of Securities Dealers Automated Quotations   │
│  LSE     London Stock Exchange                                             │
│  HKEX    Hong Kong Exchanges and Clearing                                  │
│  SGX     Singapore Exchange                                                │
│  TSE     Tokyo Stock Exchange                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Created | 2026-01-06 |
| Author | Claude (AI Assistant) |
| Status | Approved for Implementation |
