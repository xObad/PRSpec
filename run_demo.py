#!/usr/bin/env python3
"""
PRSpec Demo Script

One-click demo to run a full EIP-1559 compliance analysis.
This script demonstrates the complete PRSpec workflow:
1. Fetch specification from ethereum/execution-specs
2. Fetch code from ethereum/go-ethereum
3. Parse both specification and code
4. Analyze compliance using LLM
5. Generate reports

Usage:
    python run_demo.py

Requirements:
    - OPENAI_API_KEY set in .env file or environment
    - Python 3.8+
    - Dependencies installed: pip install -r requirements.txt

Author: Safi El-Hassanine
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import rich for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None
    print("Note: Install 'rich' for better output: pip install rich")


def print_header():
    """Print demo header."""
    header = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   PRSpec - Ethereum Specification Compliance Checker             ║
║                                                                  ║
║   Demo: EIP-1559 Compliance Analysis                             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    if HAS_RICH and console:
        console.print(Panel(
            "[bold blue]PRSpec[/bold blue] - Ethereum Specification Compliance Checker\n\n"
            "[green]Demo: EIP-1559 Compliance Analysis[/green]",
            box=box.DOUBLE,
            padding=(1, 2)
        ))
    else:
        print(header)


def print_section(title: str):
    """Print a section header."""
    if HAS_RICH and console:
        console.print(f"\n[bold cyan]{'─' * 60}[/bold cyan]")
        console.print(f"[bold]{title}[/bold]")
        console.print(f"[bold cyan]{'─' * 60}[/bold cyan]\n")
    else:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}\n")


def print_success(message: str):
    """Print a success message."""
    if HAS_RICH and console:
        console.print(f"[green]✓[/green] {message}")
    else:
        print(f"[SUCCESS] {message}")


def print_error(message: str):
    """Print an error message."""
    if HAS_RICH and console:
        console.print(f"[red]✗[/red] {message}")
    else:
        print(f"[ERROR] {message}")


def print_info(message: str):
    """Print an info message."""
    if HAS_RICH and console:
        console.print(f"[blue]ℹ[/blue] {message}")
    else:
        print(f"[INFO] {message}")


async def run_demo():
    """Run the complete demo workflow."""
    print_header()
    
    # Check environment
    print_section("Environment Check")
    
    try:
        from src.config import get_config
        config = get_config()
        api_key = config.openai_api_key
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print_success(f"OpenAI API Key configured: {masked_key}")
    except ValueError as e:
        print_error(f"OpenAI API Key not found: {e}")
        print_info("Please set OPENAI_API_KEY in your .env file or environment")
        print_info("Copy .env.example to .env and add your API key")
        return 1
    except Exception as e:
        print_error(f"Configuration error: {e}")
        return 1
    
    # Import required modules
    try:
        from src.spec_fetcher import SpecFetcher
        from src.code_fetcher import CodeFetcher
        from src.parser import SpecParser, CodeParser
        from src.analyzer import LLMAnalyzer
        from src.report_generator import ReportGenerator
        print_success("All modules imported successfully")
    except ImportError as e:
        print_error(f"Failed to import modules: {e}")
        print_info("Make sure you're running from the project root directory")
        return 1
    
    # Step 1: Fetch Specification
    print_section("Step 1: Fetching EIP-1559 Specification")
    
    try:
        spec_fetcher = SpecFetcher(config)
        print_info("Fetching from ethereum/execution-specs...")
        spec_content = spec_fetcher.extract_eip1559_spec()
        print_success(f"Specification fetched: {len(spec_content)} characters")
        
        # Parse sections
        sections = spec_fetcher.parse_sections(spec_content)
        print_info(f"Found {len(sections)} sections in specification")
        
        # Show section titles
        for i, section in enumerate(sections[:5], 1):
            print_info(f"  {i}. {section.title}")
        if len(sections) > 5:
            print_info(f"  ... and {len(sections) - 5} more sections")
            
    except Exception as e:
        print_error(f"Failed to fetch specification: {e}")
        return 1
    
    # Step 2: Fetch Code
    print_section("Step 2: Fetching go-ethereum Code")
    
    try:
        code_fetcher = CodeFetcher(config)
        print_info("Fetching from ethereum/go-ethereum...")
        code_files = code_fetcher.get_eip1559_files()
        print_success(f"Fetched {len(code_files)} code files")
        
        for file_path, code_file in code_files.items():
            print_info(f"  - {file_path}: {len(code_file.content)} chars, {len(code_file.content.split(chr(10)))} lines")
    except Exception as e:
        print_error(f"Failed to fetch code: {e}")
        return 1
    
    # Step 3: Parse Specification
    print_section("Step 3: Parsing Specification")
    
    try:
        spec_parser = SpecParser()
        parsed_spec = spec_parser.parse(spec_content, title="EIP-1559")
        print_success(f"Parsed specification: {len(parsed_spec.sections)} sections")
        print_info(f"Found {len(parsed_spec.requirements)} requirements")
        print_info(f"Found {len(parsed_spec.formulas)} formulas")
        
        # Show requirements
        if parsed_spec.requirements:
            print_info("Sample requirements:")
            for req in parsed_spec.requirements[:3]:
                print_info(f"  [{req.req_type.value.upper()}] {req.text[:80]}...")
    except Exception as e:
        print_error(f"Failed to parse specification: {e}")
        return 1
    
    # Step 4: Parse Code
    print_section("Step 4: Parsing Code")
    
    try:
        code_parser = CodeParser()
        
        # Parse first code file
        first_file = list(code_files.values())[0]
        parsed_code = code_parser.parse(first_file)
        
        print_success(f"Parsed {first_file.path.name}")
        print_info(f"Found {len(parsed_code.functions)} functions")
        print_info(f"Found {len(parsed_code.structs)} structs")
        print_info(f"Found {len(parsed_code.constants)} constants")
        
        # Show functions
        if parsed_code.functions:
            print_info("Functions found:")
            for func in parsed_code.functions[:5]:
                print_info(f"  - {func.name} (lines {func.start_line}-{func.end_line})")
    except Exception as e:
        print_error(f"Failed to parse code: {e}")
        return 1
    
    # Step 5: Analyze Compliance
    print_section("Step 5: Analyzing Compliance with LLM")
    
    try:
        analyzer = LLMAnalyzer(config)
        print_info(f"Using model: {config.llm.default_model}")
        print_info("Analyzing specification against code implementation...")
        print_info("This may take a minute...")
        
        # Run analysis
        analysis_result = await analyzer.analyze(parsed_spec, parsed_code)
        
        print_success("Analysis complete!")
        print_info(f"Overall Compliance: {analysis_result.overall_compliance.value.upper()}")
        print_info(f"Overall Confidence: {analysis_result.overall_confidence:.1%}")
        print_info(f"Sections Analyzed: {analysis_result.sections_analyzed}")
        
        # Show results summary
        if HAS_RICH and console:
            table = Table(title="Compliance Results by Section")
            table.add_column("Section", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Confidence", justify="right")
            
            status_colors = {
                'compliant': 'green',
                'partial': 'yellow',
                'non_compliant': 'red',
                'unknown': 'grey'
            }
            
            for r in analysis_result.results:
                color = status_colors.get(r.status.value, 'white')
                table.add_row(
                    r.section[:40],
                    f"[{color}]{r.status.value.upper()}[/{color}]",
                    f"{r.confidence:.1%}"
                )
            
            console.print(table)
        else:
            print("\nCompliance Results:")
            for r in analysis_result.results:
                print(f"  {r.section}: {r.status.value.upper()} ({r.confidence:.1%})")
        
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 6: Generate Reports
    print_section("Step 6: Generating Reports")
    
    try:
        report_gen = ReportGenerator(config)
        
        print_info("Generating reports in multiple formats...")
        generated = report_gen.generate(
            analysis_result,
            formats=['json', 'markdown', 'html'],
            filename='eip1559_demo_report'
        )
        
        for fmt, path in generated.items():
            print_success(f"Generated {fmt.upper()}: {path}")
        
    except Exception as e:
        print_error(f"Report generation failed: {e}")
        return 1
    
    # Summary
    print_section("Demo Complete!")
    
    summary_text = f"""
Analysis Summary:
  - Specification: EIP-1559
  - Overall Compliance: {analysis_result.overall_compliance.value.upper()}
  - Confidence: {analysis_result.overall_confidence:.1%}
  - Sections Analyzed: {analysis_result.sections_analyzed}
  - Reports Generated: {len(generated)}

Generated Files:
"""
    for fmt, path in generated.items():
        summary_text += f"  - {fmt.upper()}: {path}\n"
    
    if HAS_RICH and console:
        console.print(Panel(
            f"[bold green]Demo Complete![/bold green]\n\n"
            f"[cyan]Analysis Summary:[/cyan]\n"
            f"  Specification: EIP-1559\n"
            f"  Overall Compliance: {analysis_result.overall_compliance.value.upper()}\n"
            f"  Confidence: {analysis_result.overall_confidence:.1%}\n"
            f"  Sections Analyzed: {analysis_result.sections_analyzed}\n\n"
            f"[cyan]Generated Files:[/cyan]\n" +
            "\n".join([f"  - {fmt.upper()}: {path}" for fmt, path in generated.items()]),
            box=box.DOUBLE,
            padding=(1, 2)
        ))
    else:
        print(summary_text)
    
    print_info("Thank you for using PRSpec!")
    print_info("For more options, run: python -m src.cli --help")
    
    return 0


def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(run_demo())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
