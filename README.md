# PSI Classifier

LangGraph-based Price Sensitive Information (PSI) Classifier for analyzing stock news against regulatory frameworks.

## Overview

The PSI Classifier is an automated system that:
- Ingests stock news from CSV files
- Automatically detects the applicable regulatory framework based on region/exchange
- Uses LLM-powered analysis to classify news as PSI or non-PSI
- Provides reasoning for each classification decision
- Outputs results to JSON and console

## Supported Regulatory Frameworks

| Framework | Jurisdiction | Exchanges |
|-----------|-------------|-----------|
| SEBI | India | NSE, BSE |
| SEC | United States | NYSE, NASDAQ, AMEX |
| FCA | United Kingdom | LSE, AIM |
| ESMA | European Union | EURONEXT, XETRA |
| MAS | Singapore | SGX |
| SFC | Hong Kong | HKEX, HKG |
| FSA | Japan | TSE, JPX, JASDAQ |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd psi

# Install dependencies using uv
uv sync --all-extras
```

## Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Configure your LLM provider in `.env`:
```bash
# Choose one: azure, openrouter, gemini
LLM_PROVIDER=azure

# Configure the selected provider's credentials
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
# ... etc
```

## Usage

```bash
# Run classification on a CSV file
python -m psi.main --input data/news.csv --output reports/results.json
```

### Input CSV Format

```csv
date,news,ticker,region,exchange
2024-01-15,"Company X announces Q4 earnings beat",COMP,India,NSE
2024-01-16,"CEO attends industry conference",COMP,US,NYSE
```

### Output JSON Format

```json
{
  "metadata": {
    "generated_at": "2024-01-20T10:35:00",
    "total_items": 2,
    "psi_count": 1,
    "non_psi_count": 1
  },
  "results": [
    {
      "news_item": {...},
      "is_psi": true,
      "reasoning": "...",
      "regulatory_framework": "SEBI"
    }
  ]
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH STATE GRAPH                        │
│                                                                 │
│  ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌──────────┐  │
│  │   LOAD   │──▶│  RESOLVE  │──▶│  CLASSIFY  │──▶│  OUTPUT  │  │
│  │   NODE   │   │   NODE    │   │   NODE     │   │   NODE   │  │
│  └──────────┘   └───────────┘   └────────────┘   └──────────┘  │
│       │               │               │               │        │
│  Parse CSV      Map region/      Build prompt      Write JSON  │
│  Validate       exchange to      Call LLM          Print       │
│  schema         framework        Parse result      console     │
└─────────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/psi --cov-report=term-missing
```

## Project Structure

```
src/psi/
├── config.py          # Configuration loader
├── exceptions.py      # Custom exceptions
├── main.py            # CLI entry point
├── graph/             # LangGraph pipeline
│   ├── state.py       # State schema
│   ├── nodes.py       # Node implementations
│   └── workflow.py    # StateGraph builder
├── regulatory/        # Regulatory frameworks
│   ├── base.py        # Abstract framework
│   ├── registry.py    # Framework registry
│   └── *.py           # Framework implementations
├── llm/               # LLM providers
│   ├── base.py        # Provider interface
│   ├── factory.py     # Provider factory
│   └── *.py           # Provider implementations
└── io/                # I/O handlers
    ├── csv_reader.py  # CSV parser
    └── json_writer.py # JSON output
```

## License

MIT
