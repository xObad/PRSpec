# PRSpec - Ethereum Specification Compliance Checker

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PRSpec is an automated tool for checking Ethereum client implementations against official specifications. It uses Large Language Models (LLMs) to intelligently compare protocol specifications with code implementations, identifying compliance issues and providing actionable recommendations.

## Features

- **Automated Specification Fetching**: Downloads specs from `ethereum/execution-specs`
- **Code Repository Integration**: Clones and analyzes `ethereum/go-ethereum`
- **Intelligent Parsing**: Extracts requirements, formulas, and code structures
- **LLM-Powered Analysis**: Uses GPT-4 to compare specs against implementations
- **Multiple Output Formats**: JSON, Markdown, and HTML reports
- **EIP-1559 Support**: First-class support for EIP-1559 fee market analysis
- **Extensible Architecture**: Easy to add support for new EIPs and clients

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/safi-elhassanine/prspec.git
cd prspec

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### One-Click Demo

Run the demo to analyze EIP-1559 compliance:

```bash
python run_demo.py
```

This will:
1. Fetch the EIP-1559 specification from `ethereum/execution-specs`
2. Clone and analyze `ethereum/go-ethereum`
3. Parse and compare specification with code
4. Generate compliance reports in multiple formats

### Command Line Interface

```bash
# Check EIP-1559 compliance
prspec check --eip 1559

# Generate reports in specific formats
prspec check --eip 1559 --format json --format markdown --format html

# Update cached repositories
prspec update --repo go_ethereum

# View configuration
prspec config-info

# Show help
prspec --help
```

## Project Structure

```
prspec/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── spec_fetcher.py        # Download specs from GitHub
│   ├── code_fetcher.py        # Download client code
│   ├── parser.py              # Parse spec and code files
│   ├── analyzer.py            # LLM-powered analysis
│   ├── report_generator.py    # Generate reports
│   └── cli.py                 # Command line interface
├── tests/
│   ├── __init__.py
│   └── test_eip1559.py        # EIP-1559 tests
├── output/                    # Generated reports
├── requirements.txt           # Python dependencies
├── config.yaml               # Configuration file
├── .env.example              # Environment variables template
├── run_demo.py               # One-click demo script
└── README.md                 # This file
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (increases GitHub API rate limits)
GITHUB_TOKEN=your_github_token_here

# Optional (defaults shown)
DEFAULT_MODEL=gpt-4
CACHE_DIR=./cache
```

### Config File (`config.yaml`)

The `config.yaml` file controls:

- **Repository URLs**: GitHub repos for specs and code
- **EIP Paths**: File locations for specific EIPs
- **LLM Settings**: Model selection, tokens, temperature
- **Output Formats**: JSON, Markdown, HTML
- **Analysis Sections**: Which parts to analyze

Example:

```yaml
repositories:
  execution_specs:
    url: https://github.com/ethereum/execution-specs
    branch: master
    local_path: ./cache/execution-specs
  
  go_ethereum:
    url: https://github.com/ethereum/go-ethereum
    branch: master
    local_path: ./cache/go-ethereum

llm:
  default_model: gpt-4
  max_tokens: 4000
  temperature: 0.1
  confidence_threshold: 0.7
```

## Usage Examples

### Basic Compliance Check

```python
import asyncio
from src.config import get_config
from src.spec_fetcher import SpecFetcher
from src.code_fetcher import CodeFetcher
from src.parser import SpecParser, CodeParser
from src.analyzer import LLMAnalyzer
from src.report_generator import ReportGenerator

async def check_compliance():
    # Load configuration
    config = get_config()
    
    # Fetch specification
    spec_fetcher = SpecFetcher(config)
    spec_content = spec_fetcher.extract_eip1559_spec()
    
    # Fetch code
    code_fetcher = CodeFetcher(config)
    code_files = code_fetcher.get_eip1559_files()
    
    # Parse
    spec_parser = SpecParser()
    parsed_spec = spec_parser.parse(spec_content, title="EIP-1559")
    
    code_parser = CodeParser()
    first_file = list(code_files.values())[0]
    parsed_code = code_parser.parse(first_file)
    
    # Analyze
    analyzer = LLMAnalyzer(config)
    result = await analyzer.analyze(parsed_spec, parsed_code)
    
    # Generate report
    report_gen = ReportGenerator(config)
    report_gen.generate(result, formats=['json', 'markdown', 'html'])
    
    print(f"Compliance: {result.overall_compliance.value}")
    print(f"Confidence: {result.overall_confidence:.1%}")

asyncio.run(check_compliance())
```

### Checking a Specific PR

```bash
# Check compliance of a specific PR
prspec check-pr 12345 --eip 1559
```

### Custom Analysis

```python
from src.analyzer import LLMAnalyzer

# Analyze specific sections only
result = await analyzer.analyze(
    parsed_spec, 
    parsed_code,
    section_filter=['base_fee_calculation', 'fee_burning']
)
```

## How It Works

1. **Fetch**: Downloads specification and code from GitHub repositories
2. **Parse**: 
   - Extracts sections, requirements, and formulas from specs
   - Parses code into AST, extracts functions and structures
3. **Analyze**: 
   - Uses GPT-4 to compare spec requirements with code implementation
   - Generates confidence scores for each section
   - Identifies missing or incorrect implementations
4. **Report**: 
   - Generates structured reports in multiple formats
   - Provides actionable recommendations

## Supported EIPs

| EIP | Status | Description |
|-----|--------|-------------|
| EIP-1559 | ✅ Full | Fee market change for ETH 1.0 chain |
| EIP-2930 | 🚧 Planned | Optional access lists |
| EIP-3198 | 🚧 Planned | BASEFEE opcode |
| EIP-3529 | 🚧 Planned | Reduction in refunds |
| EIP-3541 | 🚧 Planned | Reject new contracts starting with 0xEF |
| EIP-3554 | 🚧 Planned | Difficulty bomb delay to December 2021 |

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_eip1559.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Spec Fetcher   │     │  Code Fetcher   │
│  (GitHub API)   │     │  (Git Clone)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Spec Parser    │     │  Code Parser    │
│  (Markdown)     │     │  (Tree-sitter)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  LLM Analyzer   │
            │  (GPT-4)        │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Report Generator│
            │ (JSON/MD/HTML)  │
            └─────────────────┘
```

## Requirements

- Python 3.8+
- OpenAI API key
- Git (for cloning repositories)
- 2GB+ free disk space (for cached repositories)

## Dependencies

Core dependencies:
- `openai` - OpenAI API client
- `tree-sitter` - Code parsing
- `gitpython` - Git operations
- `click` - CLI framework
- `rich` - Terminal formatting
- `pyyaml` - YAML configuration
- `python-dotenv` - Environment variables
- `aiohttp` - Async HTTP client

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/
black src/ --check

# Run type checking
mypy src/

# Run tests
pytest tests/
```

## Roadmap

- [x] EIP-1559 support
- [ ] Support for additional EIPs (London, Shanghai, Cancun upgrades)
- [ ] Support for additional clients (Nethermind, Besu, Erigon)
- [ ] Web UI for interactive analysis
- [ ] CI/CD integration
- [ ] Historical compliance tracking
- [ ] Automated PR compliance checks

## Troubleshooting

### OpenAI API Key Issues

```
Error: OPENAI_API_KEY not found in environment
```

**Solution**: Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Git Clone Failures

```
Error: Failed to clone repository
```

**Solution**: 
- Check internet connection
- Ensure sufficient disk space
- Try with `GITHUB_TOKEN` for better rate limits

### Rate Limiting

```
Error: 429 Too Many Requests
```

**Solution**: 
- Add `GITHUB_TOKEN` to `.env`
- Increase `cache_duration_hours` in config.yaml

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ethereum Foundation for protocol specifications
- Go-ethereum team for reference implementation
- OpenAI for GPT-4 API

## Author

**Safi El-Hassanine**

- GitHub: [@safi-elhassanine](https://github.com/safi-elhassanine)

## Disclaimer

This tool uses LLMs for analysis and may produce incorrect results. Always verify critical compliance decisions with manual review. The confidence scores provided are estimates and should not be the sole basis for protocol decisions.

---

<p align="center">
  Built with ❤️ for the Ethereum ecosystem
</p>
#   P R S p e c  
 #   P R S p e c  
 