"""
Configuration management for PRSpec.

Handles loading configuration from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    default_model: str = "gpt-4"
    fallback_model: str = "gpt-3.5-turbo"
    max_tokens: int = 4000
    temperature: float = 0.1
    confidence_threshold: float = 0.7


@dataclass
class RepositoryConfig:
    """Repository configuration settings."""
    url: str
    branch: str
    local_path: str


@dataclass
class EIP1559Config:
    """EIP-1559 specific configuration."""
    spec_markdown_path: str
    code_files: list
    key_functions: list


@dataclass
class OutputConfig:
    """Output configuration settings."""
    directory: str = "./output"
    report_filename: str = "prspec_report"
    include_code_snippets: bool = True
    include_line_numbers: bool = True
    include_confidence_scores: bool = True


class Config:
    """
    Main configuration class for PRSpec.
    
    Loads configuration from config.yaml and environment variables.
    Provides easy access to all configuration settings.
    
    Attributes:
        config_path: Path to the YAML configuration file
        _config_data: Raw configuration data loaded from YAML
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml. If None, searches in default locations.
        """
        self.config_path = config_path or self._find_config_file()
        self._config_data = self._load_yaml()
        
        # Initialize sub-configurations
        self.llm = self._load_llm_config()
        self.output = self._load_output_config()
        self.eip1559 = self._load_eip1559_config()
        self.repos = self._load_repository_configs()
    
    def _find_config_file(self) -> str:
        """
        Find configuration file in default locations.
        
        Returns:
            Path to config.yaml
            
        Raises:
            FileNotFoundError: If config file cannot be found
        """
        # Check common locations
        possible_paths = [
            "config.yaml",
            "../config.yaml",
            "../../config.yaml",
            os.path.expanduser("~/.prspec/config.yaml"),
            "/etc/prspec/config.yaml",
        ]
        
        # Get the directory of this file
        module_dir = Path(__file__).parent.parent
        possible_paths.insert(0, str(module_dir / "config.yaml"))
        
        for path in possible_paths:
            if os.path.isfile(path):
                return path
        
        raise FileNotFoundError(
            "Could not find config.yaml. Please create one or specify the path."
        )
    
    def _load_yaml(self) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Returns:
            Dictionary containing configuration data
            
        Raises:
            yaml.YAMLError: If YAML parsing fails
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse config file: {e}")
    
    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration from config data."""
        llm_data = self._config_data.get('llm', {})
        return LLMConfig(
            default_model=llm_data.get('default_model', 'gpt-4'),
            fallback_model=llm_data.get('fallback_model', 'gpt-3.5-turbo'),
            max_tokens=llm_data.get('max_tokens', 4000),
            temperature=llm_data.get('temperature', 0.1),
            confidence_threshold=llm_data.get('confidence_threshold', 0.7)
        )
    
    def _load_output_config(self) -> OutputConfig:
        """Load output configuration from config data."""
        output_data = self._config_data.get('output', {})
        return OutputConfig(
            directory=output_data.get('directory', './output'),
            report_filename=output_data.get('report_filename', 'prspec_report'),
            include_code_snippets=output_data.get('include_code_snippets', True),
            include_line_numbers=output_data.get('include_line_numbers', True),
            include_confidence_scores=output_data.get('include_confidence_scores', True)
        )
    
    def _load_eip1559_config(self) -> EIP1559Config:
        """Load EIP-1559 specific configuration."""
        eip_data = self._config_data.get('eip1559', {})
        return EIP1559Config(
            spec_markdown_path=eip_data.get('spec', {}).get('markdown_path', ''),
            code_files=eip_data.get('code', {}).get('files', []),
            key_functions=eip_data.get('code', {}).get('key_functions', [])
        )
    
    def _load_repository_configs(self) -> Dict[str, RepositoryConfig]:
        """Load repository configurations."""
        repos = {}
        repos_data = self._config_data.get('repositories', {})
        
        for name, data in repos_data.items():
            repos[name] = RepositoryConfig(
                url=data.get('url', ''),
                branch=data.get('branch', 'master'),
                local_path=data.get('local_path', f'./cache/{name}')
            )
        
        return repos
    
    @property
    def openai_api_key(self) -> str:
        """
        Get OpenAI API key from environment.
        
        Returns:
            OpenAI API key
            
        Raises:
            ValueError: If API key is not set
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file or environment."
            )
        return api_key
    
    @property
    def github_token(self) -> Optional[str]:
        """Get GitHub token from environment (optional)."""
        return os.getenv('GITHUB_TOKEN')
    
    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._config_data.get('analysis', {}).get('cache_enabled', True)
    
    @property
    def cache_duration_hours(self) -> int:
        """Get cache duration in hours."""
        return self._config_data.get('analysis', {}).get('cache_duration_hours', 24)
    
    @property
    def analysis_sections(self) -> list:
        """Get list of analysis sections."""
        return self._config_data.get('analysis', {}).get('sections', [])
    
    @property
    def output_formats(self) -> list:
        """Get list of output formats."""
        return self._config_data.get('analysis', {}).get('output_formats', ['json', 'markdown'])
    
    def get_cache_dir(self) -> Path:
        """
        Get cache directory path.
        
        Returns:
            Path object for cache directory
        """
        cache_dir = os.getenv('CACHE_DIR', './cache')
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_output_dir(self) -> Path:
        """
        Get output directory path.
        
        Returns:
            Path object for output directory
        """
        path = Path(self.output.directory)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create global configuration instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = Config(config_path)
    return _config_instance
