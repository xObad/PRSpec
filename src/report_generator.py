"""
Report generator for PRSpec.

Generates reports in multiple formats (JSON, Markdown, HTML).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .config import Config, get_config
from .analyzer import AnalysisResult, ComplianceResult, ComplianceStatus, analysis_result_to_dict

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates compliance reports in various formats.
    
    Supports JSON (machine-readable), Markdown (human-readable),
    and HTML (formatted web view) output formats.
    
    Attributes:
        config: Configuration instance
        output_dir: Directory for output files
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
        self.output_dir = self.config.get_output_dir()
    
    def generate(
        self,
        result: AnalysisResult,
        formats: Optional[List[str]] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Generate reports in specified formats.
        
        Args:
            result: Analysis result to report
            formats: List of formats to generate ('json', 'markdown', 'html')
            filename: Base filename (without extension)
            
        Returns:
            Dictionary mapping format to file path
        """
        formats = formats or self.config.output_formats
        filename = filename or self.config.output.report_filename
        
        generated_files = {}
        
        for fmt in formats:
            if fmt == 'json':
                path = self.generate_json(result, filename)
                generated_files['json'] = path
            elif fmt == 'markdown':
                path = self.generate_markdown(result, filename)
                generated_files['markdown'] = path
            elif fmt == 'html':
                path = self.generate_html(result, filename)
                generated_files['html'] = path
            else:
                logger.warning(f"Unknown format: {fmt}")
        
        return generated_files
    
    def generate_json(self, result: AnalysisResult, filename: str) -> Path:
        """
        Generate JSON report.
        
        Args:
            result: Analysis result
            filename: Base filename
            
        Returns:
            Path to generated file
        """
        output_path = self.output_dir / f"{filename}.json"
        
        # Convert to dictionary
        data = analysis_result_to_dict(result)
        
        # Add metadata
        data['generated_at'] = datetime.now().isoformat()
        data['tool_version'] = '0.1.0'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report generated: {output_path}")
        return output_path
    
    def generate_markdown(self, result: AnalysisResult, filename: str) -> Path:
        """
        Generate Markdown report.
        
        Args:
            result: Analysis result
            filename: Base filename
            
        Returns:
            Path to generated file
        """
        output_path = self.output_dir / f"{filename}.md"
        
        md_content = self._build_markdown_content(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report generated: {output_path}")
        return output_path
    
    def generate_html(self, result: AnalysisResult, filename: str) -> Path:
        """
        Generate HTML report.
        
        Args:
            result: Analysis result
            filename: Base filename
            
        Returns:
            Path to generated file
        """
        output_path = self.output_dir / f"{filename}.html"
        
        html_content = self._build_html_content(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return output_path
    
    def _build_markdown_content(self, result: AnalysisResult) -> str:
        """
        Build Markdown report content.
        
        Args:
            result: Analysis result
            
        Returns:
            Markdown formatted string
        """
        lines = []
        
        # Header
        lines.append(f"# PRSpec Compliance Report")
        lines.append("")
        lines.append(f"**Specification:** {result.spec_title}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        
        status_emoji = {
            ComplianceStatus.COMPLIANT: "✅",
            ComplianceStatus.PARTIAL: "⚠️",
            ComplianceStatus.NON_COMPLIANT: "❌",
            ComplianceStatus.UNKNOWN: "❓"
        }
        
        lines.append(f"**Overall Compliance:** {status_emoji.get(result.overall_compliance, '❓')} {result.overall_compliance.value.upper()}")
        lines.append("")
        lines.append(f"**Overall Confidence:** {result.overall_confidence:.1%}")
        lines.append("")
        lines.append(f"**Sections Analyzed:** {result.sections_analyzed}")
        lines.append("")
        lines.append(f"**Summary:** {result.summary}")
        lines.append("")
        
        # Compliance Breakdown
        lines.append("## Compliance Breakdown")
        lines.append("")
        
        # Count by status
        status_counts = {}
        for r in result.results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
        
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {status_emoji.get(status, '❓')} {status.value.upper()} | {count} |")
        lines.append("")
        
        # Detailed Results
        lines.append("## Detailed Results")
        lines.append("")
        
        for i, r in enumerate(result.results, 1):
            lines.append(f"### {i}. {r.section}")
            lines.append("")
            lines.append(f"**Status:** {status_emoji.get(r.status, '❓')} {r.status.value.upper()}")
            lines.append("")
            lines.append(f"**Confidence:** {r.confidence:.1%}")
            lines.append("")
            
            if r.spec_reference:
                lines.append(f"**Spec Reference:** {r.spec_reference}")
                lines.append("")
            
            if r.code_reference:
                lines.append(f"**Code Reference:** {r.code_reference}")
                lines.append("")
            
            lines.append(f"**Explanation:**")
            lines.append("")
            lines.append(r.explanation)
            lines.append("")
            
            if r.suggestions:
                lines.append("**Suggestions:**")
                lines.append("")
                for suggestion in r.suggestions:
                    lines.append(f"- {suggestion}")
                lines.append("")
            
            if r.missing_implementations:
                lines.append("**Missing Implementations:**")
                lines.append("")
                for missing in r.missing_implementations:
                    lines.append(f"- {missing}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Recommendations
        if result.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(result.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by PRSpec - Ethereum Specification Compliance Checker*")
        
        return "\n".join(lines)
    
    def _build_html_content(self, result: AnalysisResult) -> str:
        """
        Build HTML report content.
        
        Args:
            result: Analysis result
            
        Returns:
            HTML formatted string
        """
        status_emoji = {
            ComplianceStatus.COMPLIANT: "✅",
            ComplianceStatus.PARTIAL: "⚠️",
            ComplianceStatus.NON_COMPLIANT: "❌",
            ComplianceStatus.UNKNOWN: "❓"
        }
        
        status_class = {
            ComplianceStatus.COMPLIANT: "compliant",
            ComplianceStatus.PARTIAL: "partial",
            ComplianceStatus.NON_COMPLIANT: "non-compliant",
            ComplianceStatus.UNKNOWN: "unknown"
        }
        
        # Count by status
        status_counts = {}
        for r in result.results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
        
        # Build detailed results HTML
        results_html = ""
        for i, r in enumerate(result.results, 1):
            suggestions_html = ""
            if r.suggestions:
                suggestions_list = "".join([f"<li>{s}</li>" for s in r.suggestions])
                suggestions_html = f"""
                <div class="suggestions">
                    <h5>Suggestions:</h5>
                    <ul>{suggestions_list}</ul>
                </div>
                """
            
            missing_html = ""
            if r.missing_implementations:
                missing_list = "".join([f"<li>{m}</li>" for m in r.missing_implementations])
                missing_html = f"""
                <div class="missing">
                    <h5>Missing Implementations:</h5>
                    <ul>{missing_list}</ul>
                </div>
                """
            
            results_html += f"""
            <div class="result-card {status_class.get(r.status, 'unknown')}">
                <h3>{i}. {r.section}</h3>
                <div class="status-bar">
                    <span class="status {status_class.get(r.status, 'unknown')}">
                        {status_emoji.get(r.status, '❓')} {r.status.value.upper()}
                    </span>
                    <span class="confidence">Confidence: {r.confidence:.1%}</span>
                </div>
                <div class="references">
                    {f'<p><strong>Spec Reference:</strong> {r.spec_reference}</p>' if r.spec_reference else ''}
                    {f'<p><strong>Code Reference:</strong> {r.code_reference}</p>' if r.code_reference else ''}
                </div>
                <div class="explanation">
                    <h5>Explanation:</h5>
                    <p>{r.explanation}</p>
                </div>
                {suggestions_html}
                {missing_html}
            </div>
            """
        
        # Build recommendations HTML
        recommendations_html = ""
        if result.recommendations:
            rec_list = "".join([f"<li>{r}</li>" for r in result.recommendations])
            recommendations_html = f"""
            <section class="recommendations">
                <h2>Recommendations</h2>
                <ol>{rec_list}</ol>
            </section>
            """
        
        # Build status breakdown HTML
        breakdown_html = "".join([
            f"""
            <div class="stat {status_class.get(status, 'unknown')}">
                <span class="stat-value">{count}</span>
                <span class="stat-label">{status_emoji.get(status, '❓')} {status.value.upper()}</span>
            </div>
            """
            for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRSpec Compliance Report - {result.spec_title}</title>
    <style>
        :root {{
            --color-compliant: #28a745;
            --color-partial: #ffc107;
            --color-non-compliant: #dc3545;
            --color-unknown: #6c757d;
            --bg-primary: #f8f9fa;
            --bg-secondary: #ffffff;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --border-color: #dee2e6;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            background: var(--bg-secondary);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: var(--text-primary);
            margin-bottom: 10px;
        }}
        
        .meta {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        
        .summary {{
            background: var(--bg-secondary);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .overall-status {{
            font-size: 1.5em;
            margin: 15px 0;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        
        .overall-status.compliant {{ background: rgba(40, 167, 69, 0.1); color: var(--color-compliant); }}
        .overall-status.partial {{ background: rgba(255, 193, 7, 0.1); color: var(--color-partial); }}
        .overall-status.non-compliant {{ background: rgba(220, 53, 69, 0.1); color: var(--color-non-compliant); }}
        .overall-status.unknown {{ background: rgba(108, 117, 125, 0.1); color: var(--color-unknown); }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .stat {{
            background: var(--bg-primary);
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        
        .stat.compliant .stat-value {{ color: var(--color-compliant); }}
        .stat.partial .stat-value {{ color: var(--color-partial); }}
        .stat.non-compliant .stat-value {{ color: var(--color-non-compliant); }}
        .stat.unknown .stat-value {{ color: var(--color-unknown); }}
        
        .results {{
            margin-top: 30px;
        }}
        
        .result-card {{
            background: var(--bg-secondary);
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-left: 4px solid var(--color-unknown);
        }}
        
        .result-card.compliant {{ border-left-color: var(--color-compliant); }}
        .result-card.partial {{ border-left-color: var(--color-partial); }}
        .result-card.non-compliant {{ border-left-color: var(--color-non-compliant); }}
        
        .result-card h3 {{
            margin-bottom: 15px;
            color: var(--text-primary);
        }}
        
        .status-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 10px;
            background: var(--bg-primary);
            border-radius: 4px;
        }}
        
        .status {{
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 4px;
        }}
        
        .status.compliant {{ background: rgba(40, 167, 69, 0.2); color: var(--color-compliant); }}
        .status.partial {{ background: rgba(255, 193, 7, 0.2); color: var(--color-partial); }}
        .status.non-compliant {{ background: rgba(220, 53, 69, 0.2); color: var(--color-non-compliant); }}
        
        .confidence {{
            color: var(--text-secondary);
        }}
        
        .references {{
            margin: 15px 0;
            padding: 10px;
            background: var(--bg-primary);
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        .explanation, .suggestions, .missing {{
            margin-top: 15px;
        }}
        
        .explanation h5, .suggestions h5, .missing h5 {{
            margin-bottom: 8px;
            color: var(--text-secondary);
        }}
        
        .suggestions ul, .missing ul {{
            margin-left: 20px;
        }}
        
        .suggestions li {{
            color: var(--color-partial);
        }}
        
        .missing li {{
            color: var(--color-non-compliant);
        }}
        
        .recommendations {{
            background: var(--bg-secondary);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 30px;
        }}
        
        .recommendations ol {{
            margin-left: 20px;
            margin-top: 15px;
        }}
        
        .recommendations li {{
            margin-bottom: 10px;
            padding: 10px;
            background: var(--bg-primary);
            border-radius: 4px;
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>PRSpec Compliance Report</h1>
            <div class="meta">
                <p><strong>Specification:</strong> {result.spec_title}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </header>
        
        <section class="summary">
            <h2>Executive Summary</h2>
            <div class="overall-status {status_class.get(result.overall_compliance, 'unknown')}">
                {status_emoji.get(result.overall_compliance, '❓')} Overall: {result.overall_compliance.value.upper()}
            </div>
            <p><strong>Confidence:</strong> {result.overall_confidence:.1%}</p>
            <p><strong>Sections Analyzed:</strong> {result.sections_analyzed}</p>
            <p>{result.summary}</p>
            
            <div class="stats-grid">
                {breakdown_html}
            </div>
        </section>
        
        <section class="results">
            <h2>Detailed Results</h2>
            {results_html}
        </section>
        
        {recommendations_html}
        
        <footer>
            <p>Generated by PRSpec - Ethereum Specification Compliance Checker</p>
            <p>Version 0.1.0</p>
        </footer>
    </div>
</body>
</html>"""
        
        return html
    
    def generate_console_output(self, result: AnalysisResult) -> str:
        """
        Generate console-friendly output.
        
        Args:
            result: Analysis result
            
        Returns:
            Formatted string for console display
        """
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        
        console = Console()
        
        # Build output
        output = []
        
        # Header
        output.append("=" * 60)
        output.append("PRSpec Compliance Report")
        output.append("=" * 60)
        output.append("")
        output.append(f"Specification: {result.spec_title}")
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        
        # Summary
        status_emoji = {
            ComplianceStatus.COMPLIANT: "✅",
            ComplianceStatus.PARTIAL: "⚠️",
            ComplianceStatus.NON_COMPLIANT: "❌",
            ComplianceStatus.UNKNOWN: "❓"
        }
        
        output.append(f"Overall Compliance: {status_emoji.get(result.overall_compliance)} {result.overall_compliance.value.upper()}")
        output.append(f"Overall Confidence: {result.overall_confidence:.1%}")
        output.append(f"Sections Analyzed: {result.sections_analyzed}")
        output.append("")
        output.append(result.summary)
        output.append("")
        
        # Results table
        output.append("-" * 60)
        output.append("Detailed Results:")
        output.append("-" * 60)
        
        for r in result.results:
            output.append("")
            output.append(f"{r.section}")
            output.append(f"  Status: {status_emoji.get(r.status)} {r.status.value.upper()}")
            output.append(f"  Confidence: {r.confidence:.1%}")
            if r.suggestions:
                output.append(f"  Suggestions: {len(r.suggestions)}")
        
        output.append("")
        output.append("=" * 60)
        
        return "\n".join(output)
