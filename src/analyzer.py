"""
Analyzer module for PRSpec.

Uses LLM (GPT-4) to compare specifications against code implementations.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum

import openai
from openai import AsyncOpenAI

from .config import Config, get_config
from .parser import ParsedSpec, ParsedCode, SpecRequirement, FunctionInfo

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    section: str
    status: ComplianceStatus
    confidence: float  # 0.0 to 1.0
    spec_reference: str
    code_reference: str
    explanation: str
    suggestions: List[str] = field(default_factory=list)
    missing_implementations: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result for a specification vs code comparison."""
    spec_title: str
    overall_compliance: ComplianceStatus
    overall_confidence: float
    sections_analyzed: int
    results: List[ComplianceResult]
    summary: str
    recommendations: List[str] = field(default_factory=list)


class LLMAnalyzer:
    """
    LLM-powered analyzer for comparing specifications against code.
    
    Uses OpenAI's GPT-4 to perform intelligent comparison between
    specification requirements and code implementations.
    
    Attributes:
        config: Configuration instance
        client: OpenAI async client
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the LLM analyzer.
        
        Args:
            config: Configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
        self.client = AsyncOpenAI(api_key=self.config.openai_api_key)
        self.model = self.config.llm.default_model
        self.fallback_model = self.config.llm.fallback_model
        self.max_tokens = self.config.llm.max_tokens
        self.temperature = self.config.llm.temperature
    
    async def analyze(
        self,
        parsed_spec: ParsedSpec,
        parsed_code: ParsedCode,
        section_filter: Optional[List[str]] = None
    ) -> AnalysisResult:
        """
        Analyze specification compliance against code.
        
        Args:
            parsed_spec: Parsed specification
            parsed_code: Parsed code
            section_filter: Optional list of section names to analyze
            
        Returns:
            AnalysisResult with compliance information
        """
        results = []
        
        # Determine which sections to analyze
        sections_to_analyze = parsed_spec.sections
        if section_filter:
            sections_to_analyze = [
                s for s in sections_to_analyze
                if any(f.lower() in s.title.lower() for f in section_filter)
            ]
        
        logger.info(f"Analyzing {len(sections_to_analyze)} sections")
        
        # Analyze each section
        for section in sections_to_analyze:
            result = await self._analyze_section(section, parsed_code)
            results.append(result)
        
        # Calculate overall compliance
        overall_compliance = self._calculate_overall_compliance(results)
        overall_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
        
        # Generate summary
        summary = self._generate_summary(parsed_spec, results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        return AnalysisResult(
            spec_title=parsed_spec.title,
            overall_compliance=overall_compliance,
            overall_confidence=overall_confidence,
            sections_analyzed=len(results),
            results=results,
            summary=summary,
            recommendations=recommendations
        )
    
    async def _analyze_section(
        self,
        section,
        parsed_code: ParsedCode
    ) -> ComplianceResult:
        """
        Analyze a single specification section against code.
        
        Args:
            section: Specification section
            parsed_code: Parsed code
            
        Returns:
            ComplianceResult for the section
        """
        # Build the prompt
        prompt = self._build_section_prompt(section, parsed_code)
        
        try:
            # Call LLM
            response = await self._call_llm(prompt)
            
            # Parse the response
            result = self._parse_llm_response(section.title, response)
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze section {section.title}: {e}")
            return ComplianceResult(
                section=section.title,
                status=ComplianceStatus.UNKNOWN,
                confidence=0.0,
                spec_reference=f"Line {section.line_start}-{section.line_end}",
                code_reference=str(parsed_code.file_path),
                explanation=f"Analysis failed: {str(e)}",
                suggestions=["Retry analysis or check API connectivity"]
            )
    
    def _build_section_prompt(self, section, parsed_code: ParsedCode) -> str:
        """
        Build the LLM prompt for analyzing a section.
        
        Args:
            section: Specification section
            parsed_code: Parsed code
            
        Returns:
            Formatted prompt string
        """
        # Get relevant code functions
        code_functions = "\n\n".join([
            f"Function: {f.name}\n"
            f"Signature: {f.signature}\n"
            f"Lines: {f.start_line}-{f.end_line}\n"
            f"Docstring: {f.docstring or 'None'}\n"
            f"Body:\n{f.body[:1000]}..."  # Truncate for token limit
            for f in parsed_code.functions[:5]  # Limit to top 5 functions
        ])
        
        prompt = f"""You are an expert Ethereum protocol analyzer. Compare the following specification section against the code implementation.

## SPECIFICATION SECTION
Title: {section.title}
Content:
{section.content[:2000]}

## CODE IMPLEMENTATION
File: {parsed_code.file_path}
Language: {parsed_code.language}

Functions:
{code_functions}

## ANALYSIS INSTRUCTIONS
1. Compare the specification requirements against the code implementation
2. Determine if the code correctly implements the specification
3. Identify any missing implementations or deviations
4. Provide a confidence score (0.0 to 1.0)
5. Suggest improvements if needed

## RESPONSE FORMAT
Respond with a JSON object in this exact format:
{{
    "status": "compliant|partial|non_compliant|unknown",
    "confidence": 0.0-1.0,
    "spec_reference": "specific part of spec",
    "code_reference": "specific part of code",
    "explanation": "detailed explanation",
    "suggestions": ["suggestion1", "suggestion2"],
    "missing_implementations": ["missing1", "missing2"]
}}

Provide only the JSON response, no additional text."""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            prompt: Prompt to send
            
        Returns:
            LLM response text
            
        Raises:
            Exception: If API call fails
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Ethereum protocol analyzer. Provide accurate, detailed analysis of specification compliance."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Primary model failed, trying fallback: {e}")
            
            # Try fallback model
            response = await self.client.chat.completions.create(
                model=self.fallback_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Ethereum protocol analyzer."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
    
    def _parse_llm_response(self, section_name: str, response: str) -> ComplianceResult:
        """
        Parse the LLM response into a ComplianceResult.
        
        Args:
            section_name: Name of the section analyzed
            response: LLM response text
            
        Returns:
            ComplianceResult object
        """
        # Extract JSON from response
        try:
            # Find JSON in the response
            json_match = response.strip()
            if json_match.startswith("```json"):
                json_match = json_match[7:]
            if json_match.endswith("```"):
                json_match = json_match[:-3]
            json_match = json_match.strip()
            
            data = json.loads(json_match)
            
            # Map status string to enum
            status_map = {
                'compliant': ComplianceStatus.COMPLIANT,
                'partial': ComplianceStatus.PARTIAL,
                'non_compliant': ComplianceStatus.NON_COMPLIANT,
                'unknown': ComplianceStatus.UNKNOWN
            }
            
            return ComplianceResult(
                section=section_name,
                status=status_map.get(data.get('status', 'unknown'), ComplianceStatus.UNKNOWN),
                confidence=float(data.get('confidence', 0.0)),
                spec_reference=data.get('spec_reference', ''),
                code_reference=data.get('code_reference', ''),
                explanation=data.get('explanation', ''),
                suggestions=data.get('suggestions', []),
                missing_implementations=data.get('missing_implementations', [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return ComplianceResult(
                section=section_name,
                status=ComplianceStatus.UNKNOWN,
                confidence=0.0,
                spec_reference="",
                code_reference="",
                explanation=f"Failed to parse response: {response[:500]}",
                suggestions=["Check LLM response format"]
            )
    
    def _calculate_overall_compliance(self, results: List[ComplianceResult]) -> ComplianceStatus:
        """
        Calculate overall compliance status from individual results.
        
        Args:
            results: List of compliance results
            
        Returns:
            Overall compliance status
        """
        if not results:
            return ComplianceStatus.UNKNOWN
        
        # Count statuses
        status_counts = {
            ComplianceStatus.COMPLIANT: 0,
            ComplianceStatus.PARTIAL: 0,
            ComplianceStatus.NON_COMPLIANT: 0,
            ComplianceStatus.UNKNOWN: 0
        }
        
        for result in results:
            status_counts[result.status] += 1
        
        total = len(results)
        
        # Determine overall status
        if status_counts[ComplianceStatus.NON_COMPLIANT] > total * 0.2:
            return ComplianceStatus.NON_COMPLIANT
        elif status_counts[ComplianceStatus.PARTIAL] > total * 0.2:
            return ComplianceStatus.PARTIAL
        elif status_counts[ComplianceStatus.COMPLIANT] > total * 0.7:
            return ComplianceStatus.COMPLIANT
        else:
            return ComplianceStatus.PARTIAL
    
    def _generate_summary(self, parsed_spec: ParsedSpec, results: List[ComplianceResult]) -> str:
        """
        Generate a summary of the analysis.
        
        Args:
            parsed_spec: Parsed specification
            results: List of compliance results
            
        Returns:
            Summary string
        """
        compliant = sum(1 for r in results if r.status == ComplianceStatus.COMPLIANT)
        partial = sum(1 for r in results if r.status == ComplianceStatus.PARTIAL)
        non_compliant = sum(1 for r in results if r.status == ComplianceStatus.NON_COMPLIANT)
        
        summary = (
            f"Analyzed {len(results)} sections from '{parsed_spec.title}'. "
            f"Results: {compliant} compliant, {partial} partial, {non_compliant} non-compliant. "
        )
        
        if non_compliant > 0:
            summary += "Critical issues found that need attention. "
        elif partial > 0:
            summary += "Some sections need improvement. "
        else:
            summary += "Implementation appears to be fully compliant."
        
        return summary
    
    def _generate_recommendations(self, results: List[ComplianceResult]) -> List[str]:
        """
        Generate overall recommendations from analysis results.
        
        Args:
            results: List of compliance results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Collect all suggestions
        all_suggestions = []
        for result in results:
            all_suggestions.extend(result.suggestions)
        
        # Deduplicate and prioritize
        seen = set()
        for suggestion in all_suggestions:
            if suggestion and suggestion not in seen:
                seen.add(suggestion)
                recommendations.append(suggestion)
        
        # Add priority recommendations based on status
        non_compliant_sections = [r.section for r in results if r.status == ComplianceStatus.NON_COMPLIANT]
        if non_compliant_sections:
            recommendations.insert(0, f"Priority: Fix non-compliant sections: {', '.join(non_compliant_sections)}")
        
        return recommendations[:10]  # Limit to top 10
    
    async def analyze_requirement(
        self,
        requirement: SpecRequirement,
        function: FunctionInfo
    ) -> Dict[str, Any]:
        """
        Analyze a single requirement against a specific function.
        
        Args:
            requirement: Specification requirement
            function: Code function
            
        Returns:
            Analysis dictionary
        """
        prompt = f"""Analyze if this code function implements the specification requirement.

## REQUIREMENT
Type: {requirement.req_type.value}
Section: {requirement.section}
Text: {requirement.text}
Context: {requirement.context}

## CODE FUNCTION
Name: {function.name}
Signature: {function.signature}
Docstring: {function.docstring or 'None'}
Body:
{function.body[:1500]}

Does this function correctly implement the requirement? Provide:
1. Yes/No/Partial
2. Confidence score (0-1)
3. Explanation
4. Specific line references

Respond in JSON format."""
        
        try:
            response = await self._call_llm(prompt)
            return json.loads(response.strip())
        except Exception as e:
            logger.error(f"Requirement analysis failed: {e}")
            return {
                "implemented": "unknown",
                "confidence": 0.0,
                "explanation": f"Analysis failed: {str(e)}"
            }


def analysis_result_to_dict(result: AnalysisResult) -> Dict[str, Any]:
    """
    Convert AnalysisResult to dictionary for serialization.
    
    Args:
        result: AnalysisResult to convert
        
    Returns:
        Dictionary representation
    """
    return {
        "spec_title": result.spec_title,
        "overall_compliance": result.overall_compliance.value,
        "overall_confidence": result.overall_confidence,
        "sections_analyzed": result.sections_analyzed,
        "summary": result.summary,
        "recommendations": result.recommendations,
        "results": [
            {
                "section": r.section,
                "status": r.status.value,
                "confidence": r.confidence,
                "spec_reference": r.spec_reference,
                "code_reference": r.code_reference,
                "explanation": r.explanation,
                "suggestions": r.suggestions,
                "missing_implementations": r.missing_implementations
            }
            for r in result.results
        ]
    }
