"""
Tests for EIP-1559 compliance checking.

This module contains tests for the EIP-1559 specification compliance
checking functionality of PRSpec.
"""

import os
import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config, get_config
from src.spec_fetcher import SpecFetcher, SpecSection
from src.code_fetcher import CodeFetcher, CodeFile, FunctionInfo
from src.parser import SpecParser, CodeParser, ParsedSpec, ParsedCode
from src.analyzer import LLMAnalyzer, AnalysisResult, ComplianceResult, ComplianceStatus
from src.report_generator import ReportGenerator


# ============== Fixtures ==============

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=Config)
    config.openai_api_key = "test-api-key"
    config.github_token = "test-github-token"
    
    # Set up nested mocks properly
    config.llm = Mock()
    config.llm.default_model = "gpt-4"
    config.llm.fallback_model = "gpt-3.5-turbo"
    config.llm.max_tokens = 4000
    config.llm.temperature = 0.1
    config.llm.confidence_threshold = 0.7
    
    config.output = Mock()
    config.output.report_filename = "test_report"
    config.output.directory = "./test_output"
    config.output_formats = ["json", "markdown"]
    
    config.eip1559 = Mock()
    config.eip1559.spec_markdown_path = "network-upgrades/mainnet-upgrades/london.md"
    config.eip1559.code_files = [
        "core/types/transaction.go",
        "core/types/eip1559.go"
    ]
    config.eip1559.key_functions = [
        "CalcBaseFee",
        "EffectiveGasTip"
    ]
    
    config.repos = {
        'execution_specs': Mock(url="https://github.com/ethereum/execution-specs", 
                                branch="master", 
                                local_path="./test_cache/execution-specs"),
        'go_ethereum': Mock(url="https://github.com/ethereum/go-ethereum",
                          branch="master",
                          local_path="./test_cache/go-ethereum")
    }
    config.get_cache_dir.return_value = Path("./test_cache")
    config.get_output_dir.return_value = Path("./test_output")
    return config


@pytest.fixture
def sample_eip1559_spec():
    """Sample EIP-1559 specification content."""
    return """
# EIP-1559: Fee Market Change for ETH 1.0 Chain

## Simple Summary

The current "first price auction" fee market in Ethereum is inefficient and needlessly costly to users. 
This EIP proposes a way to replace this with a mechanism that adjusts a base fee that is burned, 
while allowing miners (and eventually validators) to receive a small priority fee.

## Abstract

There is a base fee per gas in protocol, which can move up or down by a maximum of 1/8 per block. 
Transactions specify the maximum fee per gas they are willing to give to miners, and the maximum 
base fee they are willing to pay.

## Specification

### Base Fee Calculation

The base fee per gas is calculated as follows:

```
base_fee_per_gas = parent_base_fee_per_gas * (1 + (parent_gas_used - parent_gas_target) / parent_gas_target / 8)
```

Where:
- `parent_gas_target` is the gas target of the parent block (15 million gas)
- `parent_gas_used` is the gas used by the parent block
- `parent_base_fee_per_gas` is the base fee per gas of the parent block

### Fee Burning

The base fee is burned. This means it is subtracted from the sender's balance but not added to 
the miner's balance. This creates deflationary pressure on ETH.

### Transaction Validation

Transactions must specify:
- `max_fee_per_gas`: The maximum total fee per gas the sender is willing to pay
- `max_priority_fee_per_gas`: The maximum priority fee per gas the sender is willing to pay

The effective gas price is calculated as:
```
effective_gas_price = min(max_fee_per_gas, base_fee_per_gas + max_priority_fee_per_gas)
```

## Rationale

The base fee mechanism creates more predictable fees and improves user experience.
"""


@pytest.fixture
def sample_go_code():
    """Sample Go code implementing EIP-1559."""
    return '''
package types

import (
	"math/big"
)

// CalcBaseFee calculates the base fee for the next block based on parent block metrics
func CalcBaseFee(config *params.ChainConfig, parent *BlockHeader) *big.Int {
	// If the current block is the first EIP-1559 block, return the initial base fee
	if !config.IsLondon(parent.Number) {
		return new(big.Int).SetUint64(params.InitialBaseFee)
	}

	parentGasTarget := parent.GasLimit / params.ElasticityMultiplier
	
	// If the parent gas used is the same as the target, the base fee remains the same
	if parent.GasUsed == parentGasTarget {
		return new(big.Int).Set(parent.BaseFee)
	}

	var (
		num   = new(big.Int)
		denom = new(big.Int)
	)

	if parent.GasUsed > parentGasTarget {
		// If the parent block used more gas than its target, the base fee should increase
		num.SetUint64(parent.GasUsed - parentGasTarget)
		num.Mul(num, parent.BaseFee)
		num.Div(num, denom.SetUint64(parentGasTarget))
		num.Div(num, denom.SetUint64(params.BaseFeeChangeDenominator))
		baseFeeDelta := math.Max(num, common.Big1)

		return num.Add(parent.BaseFee, baseFeeDelta)
	} else {
		// Otherwise if the parent block used less gas than its target, the base fee should decrease
		num.SetUint64(parentGasTarget - parent.GasUsed)
		num.Mul(num, parent.BaseFee)
		num.Div(num, denom.SetUint64(parentGasTarget))
		num.Div(num, denom.SetUint64(params.BaseFeeChangeDenominator))
		baseFeeDelta := math.Max(num, common.Big1)

		return num.Sub(parent.BaseFee, baseFeeDelta)
	}
}

// EffectiveGasTip returns the effective miner gas tip for the given transaction
func (tx *Transaction) EffectiveGasTip(baseFee *big.Int) (*big.Int, error) {
	if baseFee == nil {
		return tx.GasTipCap(), nil
	}
	var err error
	gasFeeCap := tx.GasFeeCap()
	if gasFeeCap.Cmp(baseFee) < 0 {
		err = ErrGasFeeCapTooLow
	}
	gasFeeCap.Sub(gasFeeCap, baseFee)
	if gasFeeCap.Cmp(tx.GasTipCap()) > 0 {
		return tx.GasTipCap(), err
	}
	return gasFeeCap, err
}
'''


# ============== Config Tests ==============

class TestConfig:
    """Tests for configuration management."""
    
    def test_config_loads_from_yaml(self, tmp_path):
        """Test that config loads from YAML file."""
        config_content = """
llm:
  default_model: gpt-4
  max_tokens: 4000

output:
  directory: ./output
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            config = Config(str(config_file))
            assert config.llm.default_model == "gpt-4"
            assert config.llm.max_tokens == 4000
    
    def test_openai_api_key_from_env(self, mock_config):
        """Test that OpenAI API key is read from environment."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key-12345'}):
            # Create a real config to test env var reading
            config = Config.__new__(Config)
            config.config_path = "test"
            config._config_data = {}
            assert config.openai_api_key == 'test-api-key-12345'


# ============== Spec Fetcher Tests ==============

class TestSpecFetcher:
    """Tests for specification fetching."""
    
    def test_parse_sections(self, mock_config, sample_eip1559_spec):
        """Test parsing specification into sections."""
        fetcher = SpecFetcher.__new__(SpecFetcher)
        fetcher.config = mock_config
        
        sections = fetcher.parse_sections(sample_eip1559_spec)
        
        assert len(sections) > 0
        assert any("Simple Summary" in s.title for s in sections)
        assert any("Specification" in s.title for s in sections)
    
    def test_find_section_by_title(self, mock_config, sample_eip1559_spec):
        """Test finding sections by title pattern."""
        fetcher = SpecFetcher.__new__(SpecFetcher)
        fetcher.config = mock_config
        
        section = fetcher.find_section_by_title(sample_eip1559_spec, r"Base Fee")
        
        assert section is not None
        assert "Base Fee" in section.title


# ============== Code Fetcher Tests ==============

class TestCodeFetcher:
    """Tests for code fetching."""
    
    def test_detect_language(self, mock_config):
        """Test language detection from file extension."""
        fetcher = CodeFetcher.__new__(CodeFetcher)
        fetcher.config = mock_config
        fetcher.repo_path = Path("./test")
        fetcher._repo = None
        
        assert fetcher._detect_language('.go') == 'go'
        assert fetcher._detect_language('.py') == 'python'
        assert fetcher._detect_language('.rs') == 'rust'
    
    def test_extract_functions(self, mock_config, sample_go_code):
        """Test extracting functions from Go code."""
        fetcher = CodeFetcher.__new__(CodeFetcher)
        fetcher.config = mock_config
        
        code_file = CodeFile(
            path=Path("test.go"),
            content=sample_go_code,
            language='go',
            last_modified=None,
            commit_hash='abc123'
        )
        
        functions = fetcher.extract_functions(code_file)
        
        assert len(functions) >= 2
        func_names = [f.name for f in functions]
        assert 'CalcBaseFee' in func_names
        assert 'EffectiveGasTip' in func_names


# ============== Parser Tests ==============

class TestSpecParser:
    """Tests for specification parsing."""
    
    def test_parse_specification(self, sample_eip1559_spec):
        """Test parsing full specification."""
        parser = SpecParser()
        
        parsed = parser.parse(sample_eip1559_spec, title="EIP-1559")
        
        assert parsed.title == "EIP-1559"
        assert len(parsed.sections) > 0
        assert len(parsed.requirements) > 0
    
    def test_extract_requirements(self, sample_eip1559_spec):
        """Test extracting requirements from spec."""
        parser = SpecParser()
        sections = parser._parse_sections(sample_eip1559_spec)
        
        requirements = parser._extract_requirements(sample_eip1559_spec, sections)
        
        assert len(requirements) > 0
        # Check that we found MUST-type requirements
        must_reqs = [r for r in requirements if r.req_type.value == 'must']
        assert len(must_reqs) > 0
    
    def test_extract_formulas(self, sample_eip1559_spec):
        """Test extracting mathematical formulas."""
        parser = SpecParser()
        
        formulas = parser._extract_formulas(sample_eip1559_spec)
        
        assert len(formulas) >= 2
        # Should find base fee calculation formula
        formula_texts = [f['text'] for f in formulas]
        assert any('base_fee' in ft.lower() for ft in formula_texts)


class TestCodeParser:
    """Tests for code parsing."""
    
    def test_parse_go_code(self, sample_go_code):
        """Test parsing Go source code."""
        parser = CodeParser()
        
        code_file = CodeFile(
            path=Path("test.go"),
            content=sample_go_code,
            language='go',
            last_modified=None,
            commit_hash='abc123'
        )
        
        parsed = parser.parse(code_file)
        
        assert parsed.language == 'go'
        assert len(parsed.functions) >= 2
        assert any(f.name == 'CalcBaseFee' for f in parsed.functions)
    
    def test_extract_structs(self, sample_go_code):
        """Test extracting struct definitions."""
        parser = CodeParser()
        
        structs = parser._extract_structs(sample_go_code, 'go')
        
        # The sample code doesn't have structs, but function should work
        assert isinstance(structs, list)
    
    def test_extract_comments(self, sample_go_code):
        """Test extracting comments from code."""
        parser = CodeParser()
        
        comments = parser._extract_comments(sample_go_code, 'go')
        
        assert len(comments) > 0
        # Should find the CalcBaseFee docstring
        assert any('CalcBaseFee' in c['text'] for c in comments)


# ============== Analyzer Tests ==============

class TestAnalyzer:
    """Tests for LLM analyzer."""
    
    @pytest.mark.asyncio
    async def test_analyze_section(self, mock_config, sample_eip1559_spec, sample_go_code):
        """Test analyzing a specification section."""
        # Create parsed spec and code
        spec_parser = SpecParser()
        parsed_spec = spec_parser.parse(sample_eip1559_spec, title="EIP-1559")
        
        code_parser = CodeParser()
        code_file = CodeFile(
            path=Path("test.go"),
            content=sample_go_code,
            language='go',
            last_modified=None,
            commit_hash='abc123'
        )
        parsed_code = code_parser.parse(code_file)
        
        # Mock the LLM call
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer.config = mock_config
        analyzer.model = "gpt-4"
        analyzer.fallback_model = "gpt-3.5-turbo"
        analyzer.max_tokens = 4000
        analyzer.temperature = 0.1
        
        # Mock the _call_llm method
        mock_response = '''
        {
            "status": "compliant",
            "confidence": 0.85,
            "spec_reference": "Base Fee Calculation section",
            "code_reference": "CalcBaseFee function",
            "explanation": "The code correctly implements the base fee calculation formula.",
            "suggestions": [],
            "missing_implementations": []
        }
        '''
        
        analyzer._call_llm = Mock(return_value=asyncio.Future())
        analyzer._call_llm.return_value.set_result(mock_response)
        
        # Test analyzing a section
        section = parsed_spec.sections[0]
        result = await analyzer._analyze_section(section, parsed_code)
        
        assert result.section == section.title
        assert result.confidence == 0.85
    
    def test_calculate_overall_compliance(self, mock_config):
        """Test calculating overall compliance status."""
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer.config = mock_config
        
        # Test mostly compliant -> COMPLIANT
        results = [
            ComplianceResult("Section 1", ComplianceStatus.COMPLIANT, 0.9, "", "", ""),
            ComplianceResult("Section 2", ComplianceStatus.COMPLIANT, 0.8, "", "", ""),
            ComplianceResult("Section 3", ComplianceStatus.COMPLIANT, 0.85, "", "", ""),
        ]
        overall = analyzer._calculate_overall_compliance(results)
        assert overall == ComplianceStatus.COMPLIANT
        
        # Test with partial results -> PARTIAL
        results = [
            ComplianceResult("Section 1", ComplianceStatus.COMPLIANT, 0.9, "", "", ""),
            ComplianceResult("Section 2", ComplianceStatus.COMPLIANT, 0.8, "", "", ""),
            ComplianceResult("Section 3", ComplianceStatus.PARTIAL, 0.6, "", "", ""),
        ]
        overall = analyzer._calculate_overall_compliance(results)
        assert overall == ComplianceStatus.PARTIAL
    
    def test_generate_recommendations(self, mock_config):
        """Test generating recommendations from results."""
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer.config = mock_config
        
        results = [
            ComplianceResult("Section 1", ComplianceStatus.COMPLIANT, 0.9, "", "", "", 
                           suggestions=["Add more comments"]),
            ComplianceResult("Section 2", ComplianceStatus.NON_COMPLIANT, 0.5, "", "", "",
                           suggestions=["Fix formula implementation"]),
        ]
        
        recommendations = analyzer._generate_recommendations(results)
        
        assert len(recommendations) > 0
        assert any("Fix formula" in r for r in recommendations)


# ============== Report Generator Tests ==============

class TestReportGenerator:
    """Tests for report generation."""
    
    def test_generate_json_report(self, mock_config, tmp_path):
        """Test generating JSON report."""
        mock_config.get_output_dir.return_value = tmp_path
        
        generator = ReportGenerator(mock_config)
        
        # Create mock analysis result
        result = AnalysisResult(
            spec_title="EIP-1559",
            overall_compliance=ComplianceStatus.COMPLIANT,
            overall_confidence=0.85,
            sections_analyzed=3,
            results=[
                ComplianceResult("Base Fee", ComplianceStatus.COMPLIANT, 0.9, "", "", "Good"),
            ],
            summary="Test summary",
            recommendations=["Test recommendation"]
        )
        
        path = generator.generate_json(result, "test_report")
        
        assert path.exists()
        assert path.suffix == '.json'
        
        # Verify content
        import json
        with open(path) as f:
            data = json.load(f)
        assert data['spec_title'] == "EIP-1559"
        assert data['overall_compliance'] == "compliant"
    
    def test_generate_markdown_report(self, mock_config, tmp_path):
        """Test generating Markdown report."""
        mock_config.get_output_dir.return_value = tmp_path
        
        generator = ReportGenerator(mock_config)
        
        result = AnalysisResult(
            spec_title="EIP-1559",
            overall_compliance=ComplianceStatus.COMPLIANT,
            overall_confidence=0.85,
            sections_analyzed=3,
            results=[],
            summary="Test summary",
            recommendations=[]
        )
        
        path = generator.generate_markdown(result, "test_report")
        
        assert path.exists()
        assert path.suffix == '.md'
        
        content = path.read_text()
        assert "EIP-1559" in content
        assert "COMPLIANT" in content


# ============== Integration Tests ==============

class TestEIP1559Integration:
    """Integration tests for EIP-1559 compliance checking."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, sample_eip1559_spec, sample_go_code, mock_config, tmp_path):
        """Test the complete analysis pipeline."""
        mock_config.get_output_dir.return_value = tmp_path
        mock_config.get_cache_dir.return_value = tmp_path / "cache"
        
        # Step 1: Parse specification
        spec_parser = SpecParser()
        parsed_spec = spec_parser.parse(sample_eip1559_spec, title="EIP-1559")
        
        assert len(parsed_spec.sections) > 0
        assert len(parsed_spec.requirements) > 0
        
        # Step 2: Parse code
        code_parser = CodeParser()
        code_file = CodeFile(
            path=Path("core/types/eip1559.go"),
            content=sample_go_code,
            language='go',
            last_modified=None,
            commit_hash='abc123'
        )
        parsed_code = code_parser.parse(code_file)
        
        assert len(parsed_code.functions) >= 2
        
        # Step 3: Analyze (mocked)
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer.config = mock_config
        analyzer.model = "gpt-4"
        analyzer.fallback_model = "gpt-3.5-turbo"
        analyzer.max_tokens = 4000
        analyzer.temperature = 0.1
        
        mock_response = '''
        {
            "status": "compliant",
            "confidence": 0.9,
            "spec_reference": "Base Fee Calculation",
            "code_reference": "CalcBaseFee",
            "explanation": "Correct implementation",
            "suggestions": [],
            "missing_implementations": []
        }
        '''
        analyzer._call_llm = Mock(return_value=asyncio.Future())
        analyzer._call_llm.return_value.set_result(mock_response)
        
        # Step 4: Generate report
        report_gen = ReportGenerator(mock_config)
        
        # Create analysis result
        analysis_result = AnalysisResult(
            spec_title="EIP-1559",
            overall_compliance=ComplianceStatus.COMPLIANT,
            overall_confidence=0.9,
            sections_analyzed=len(parsed_spec.sections),
            results=[],
            summary="Full analysis complete",
            recommendations=[]
        )
        
        # Generate all report formats
        paths = report_gen.generate(analysis_result, formats=['json', 'markdown'])
        
        assert 'json' in paths
        assert 'markdown' in paths
        assert paths['json'].exists()
        assert paths['markdown'].exists()


# ============== Main Test Runner ==============

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
