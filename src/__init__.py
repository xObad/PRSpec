"""
PRSpec - Ethereum Specification Compliance Checker

A tool for checking Ethereum client implementations against official specifications.
"""

__version__ = "0.1.0"
__author__ = "Safi El-Hassanine"

from .config import Config
from .spec_fetcher import SpecFetcher
from .code_fetcher import CodeFetcher
from .parser import SpecParser, CodeParser
from .analyzer import LLMAnalyzer
from .report_generator import ReportGenerator

__all__ = [
    "Config",
    "SpecFetcher",
    "CodeFetcher",
    "SpecParser",
    "CodeParser",
    "LLMAnalyzer",
    "ReportGenerator",
]
