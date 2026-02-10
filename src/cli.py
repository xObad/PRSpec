"""
Command Line Interface for PRSpec.

Provides commands for checking compliance, generating reports, and managing configuration.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Config, get_config
from .spec_fetcher import SpecFetcher
from .code_fetcher import CodeFetcher
from .parser import SpecParser, CodeParser
from .analyzer import LLMAnalyzer
from .report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()


@click.group()
@click.option('--config', '-c', type=click.Path(), help='Path to config file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """
    PRSpec - Ethereum Specification Compliance Checker
    
    Check Ethereum client implementations against official specifications.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Store config path
    ctx.obj['config_path'] = config
    
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    try:
        cfg = get_config(config)
        ctx.obj['config'] = cfg
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--eip', '-e', default='1559', help='EIP number to check (default: 1559)')
@click.option('--format', '-f', 'output_formats', multiple=True, 
              default=['markdown'], 
              type=click.Choice(['json', 'markdown', 'html', 'console']),
              help='Output format(s)')
@click.option('--output', '-o', help='Output filename (without extension)')
@click.option('--force-update', is_flag=True, help='Force update repositories')
@click.pass_context
def check(ctx, eip: str, output_formats: List[str], output: Optional[str], force_update: bool):
    """
    Run compliance check for a specific EIP.
    
    Example:
        prspec check --eip 1559 --format json --format markdown
    """
    config = ctx.obj['config']
    
    asyncio.run(_run_check(config, eip, list(output_formats), output, force_update))


async def _run_check(
    config: Config,
    eip: str,
    output_formats: List[str],
    output_filename: Optional[str],
    force_update: bool
):
    """Run the compliance check asynchronously."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Step 1: Fetch specification
        task = progress.add_task("Fetching specification...", total=None)
        
        try:
            spec_fetcher = SpecFetcher(config)
            spec_content = spec_fetcher.extract_eip1559_spec()
            progress.update(task, description="[green]✓[/green] Specification fetched")
        except Exception as e:
            progress.update(task, description=f"[red]✗[/red] Failed to fetch spec: {e}")
            console.print(f"[red]Error: {e}[/red]")
            return
        
        # Step 2: Fetch code
        task = progress.add_task("Fetching client code...", total=None)
        
        try:
            code_fetcher = CodeFetcher(config)
            if force_update:
                code_fetcher.fetch(force_update=True)
            code_files = code_fetcher.get_eip1559_files()
            progress.update(task, description=f"[green]✓[/green] Fetched {len(code_files)} code files")
        except Exception as e:
            progress.update(task, description=f"[red]✗[/red] Failed to fetch code: {e}")
            console.print(f"[red]Error: {e}[/red]")
            return
        
        # Step 3: Parse specification
        task = progress.add_task("Parsing specification...", total=None)
        
        try:
            spec_parser = SpecParser()
            parsed_spec = spec_parser.parse(spec_content, title=f"EIP-{eip}")
            progress.update(task, description=f"[green]✓[/green] Parsed {len(parsed_spec.sections)} sections")
        except Exception as e:
            progress.update(task, description=f"[red]✗[/red] Failed to parse spec: {e}")
            console.print(f"[red]Error: {e}[/red]")
            return
        
        # Step 4: Parse code
        task = progress.add_task("Parsing code...", total=None)
        
        try:
            code_parser = CodeParser()
            # For now, analyze the first file
            first_file = list(code_files.values())[0] if code_files else None
            if first_file:
                parsed_code = code_parser.parse(first_file)
                progress.update(task, description=f"[green]✓[/green] Parsed {len(parsed_code.functions)} functions")
            else:
                progress.update(task, description="[red]✗[/red] No code files found")
                return
        except Exception as e:
            progress.update(task, description=f"[red]✗[/red] Failed to parse code: {e}")
            console.print(f"[red]Error: {e}[/red]")
            return
        
        # Step 5: Analyze with LLM
        task = progress.add_task("Analyzing compliance with LLM...", total=None)
        
        try:
            analyzer = LLMAnalyzer(config)
            analysis_result = await analyzer.analyze(parsed_spec, parsed_code)
            progress.update(task, description=f"[green]✓[/green] Analysis complete")
        except Exception as e:
            progress.update(task, description=f"[red]✗[/red] Analysis failed: {e}")
            console.print(f"[red]Error: {e}[/red]")
            return
        
        # Step 6: Generate reports
        task = progress.add_task("Generating reports...", total=None)
        
        try:
            report_gen = ReportGenerator(config)
            
            # Handle console output separately
            formats = [f for f in output_formats if f != 'console']
            
            if formats:
                generated = report_gen.generate(
                    analysis_result,
                    formats=formats,
                    filename=output_filename
                )
                file_list = "\n".join([f"  - {fmt}: {path}" for fmt, path in generated.items()])
                progress.update(task, description=f"[green]✓[/green] Reports generated:\n{file_list}")
            else:
                progress.update(task, description="[green]✓[/green] Reports ready")
            
            # Print console output if requested
            if 'console' in output_formats:
                console.print("\n")
                console_output = report_gen.generate_console_output(analysis_result)
                console.print(console_output)
            
        except Exception as e:
            progress.update(task, description=f"[red]✗[/red] Report generation failed: {e}")
            console.print(f"[red]Error: {e}[/red]")
            return
    
    # Summary
    console.print("\n")
    console.print(Panel(
        f"[bold]Analysis Complete[/bold]\n\n"
        f"Overall Compliance: {analysis_result.overall_compliance.value.upper()}\n"
        f"Confidence: {analysis_result.overall_confidence:.1%}\n"
        f"Sections Analyzed: {analysis_result.sections_analyzed}",
        title="PRSpec Summary",
        border_style="green" if analysis_result.overall_compliance.value == 'compliant' else "yellow"
    ))


@cli.command()
@click.option('--format', '-f', 'output_formats', multiple=True,
              default=['markdown'],
              type=click.Choice(['json', 'markdown', 'html']),
              help='Report format(s)')
@click.option('--output', '-o', help='Output filename (without extension)')
@click.argument('analysis_file', type=click.Path(exists=True), required=False)
@click.pass_context
def report(ctx, output_formats: List[str], output: Optional[str], analysis_file: Optional[str]):
    """
    Generate reports from a previous analysis.
    
    If ANALYSIS_FILE is provided, generates reports from that JSON file.
    Otherwise, runs a new analysis first.
    
    Example:
        prspec report analysis.json --format html
    """
    config = ctx.obj['config']
    
    if analysis_file:
        # Load existing analysis
        import json
        with open(analysis_file, 'r') as f:
            data = json.load(f)
        
        # Convert back to AnalysisResult (simplified)
        console.print(f"[green]Loaded analysis from {analysis_file}[/green]")
        console.print("Report generation from saved analysis not yet fully implemented.")
    else:
        console.print("[yellow]No analysis file provided. Run 'prspec check' first.[/yellow]")


@cli.command()
@click.pass_context
def config_info(ctx):
    """
    Display current configuration.
    
    Shows loaded configuration values and paths.
    """
    config = ctx.obj['config']
    
    table = Table(title="PRSpec Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # General settings
    table.add_row("Config Path", config.config_path)
    table.add_row("Cache Directory", str(config.get_cache_dir()))
    table.add_row("Output Directory", str(config.get_output_dir()))
    
    # LLM settings
    table.add_row("LLM Model", config.llm.default_model)
    table.add_row("Fallback Model", config.llm.fallback_model)
    table.add_row("Max Tokens", str(config.llm.max_tokens))
    table.add_row("Temperature", str(config.llm.temperature))
    
    # Repositories
    for name, repo in config.repos.items():
        table.add_row(f"Repo: {name}", f"{repo.url} ({repo.branch})")
    
    # API Keys (masked)
    try:
        api_key = config.openai_api_key
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        table.add_row("OpenAI API Key", masked)
    except ValueError:
        table.add_row("OpenAI API Key", "[red]Not set![/red]")
    
    console.print(table)


@cli.command()
@click.option('--repo', type=click.Choice(['execution_specs', 'go_ethereum']), 
              help='Repository to update')
@click.pass_context
def update(ctx, repo: Optional[str]):
    """
    Update cached repositories.
    
    Fetches latest changes from GitHub repositories.
    
    Example:
        prspec update --repo go_ethereum
    """
    config = ctx.obj['config']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        if repo == 'execution_specs' or repo is None:
            task = progress.add_task("Updating execution-specs...", total=None)
            try:
                fetcher = SpecFetcher(config)
                fetcher.fetch(force_update=True)
                progress.update(task, description="[green]✓[/green] execution-specs updated")
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] Failed: {e}")
        
        if repo == 'go_ethereum' or repo is None:
            task = progress.add_task("Updating go-ethereum...", total=None)
            try:
                fetcher = CodeFetcher(config)
                fetcher.fetch(force_update=True)
                progress.update(task, description="[green]✓[/green] go-ethereum updated")
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] Failed: {e}")
    
    console.print("[green]Update complete![/green]")


@cli.command()
@click.argument('pr_number', type=int)
@click.option('--eip', '-e', default='1559', help='EIP number to check')
@click.option('--format', '-f', 'output_formats', multiple=True,
              default=['markdown'],
              type=click.Choice(['json', 'markdown', 'html', 'console']),
              help='Output format(s)')
@click.pass_context
def check_pr(ctx, pr_number: int, eip: str, output_formats: List[str]):
    """
    Check compliance for a specific Pull Request.
    
    Analyzes the code at a specific PR state against the specification.
    
    Example:
        prspec check-pr 12345 --eip 1559
    """
    config = ctx.obj['config']
    
    console.print(f"[yellow]PR analysis not yet fully implemented.[/yellow]")
    console.print(f"Would analyze PR #{pr_number} for EIP-{eip}")


@cli.command()
def version():
    """
    Show PRSpec version information.
    """
    console.print(Panel(
        f"[bold]PRSpec[/bold] - Ethereum Specification Compliance Checker\n"
        f"Version: 0.1.0\n"
        f"Author: Safi El-Hassanine",
        title="About",
        border_style="blue"
    ))


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
