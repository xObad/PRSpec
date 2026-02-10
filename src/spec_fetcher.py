"""
Specification fetcher for PRSpec.

Handles downloading and caching Ethereum specifications from GitHub repositories.
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

import git
from git.exc import GitCommandError

from .config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class SpecSection:
    """Represents a section of a specification document."""
    title: str
    content: str
    level: int
    line_start: int
    line_end: int


class SpecFetcher:
    """
    Fetches and manages Ethereum specification documents.
    
    This class handles cloning/pulling the ethereum/execution-specs repository
    and extracting specific EIP sections.
    
    Attributes:
        config: Configuration instance
        repo_path: Local path to the cloned repository
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the spec fetcher.
        
        Args:
            config: Configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
        self.repo_config = self.config.repos.get('execution_specs')
        if not self.repo_config:
            raise ValueError("Execution specs repository not configured")
        
        self.repo_path = Path(self.repo_config.local_path)
        self._repo: Optional[git.Repo] = None
    
    def fetch(self, force_update: bool = False) -> Path:
        """
        Fetch or update the specification repository.
        
        Args:
            force_update: If True, force pull even if repo exists
            
        Returns:
            Path to the local repository
            
        Raises:
            GitCommandError: If git operations fail
        """
        if self._repo_exists():
            if force_update:
                self._pull_repository()
            else:
                logger.info(f"Repository already exists at {self.repo_path}")
        else:
            self._clone_repository()
        
        return self.repo_path
    
    def _repo_exists(self) -> bool:
        """Check if repository already exists locally."""
        return (self.repo_path / ".git").exists()
    
    def _clone_repository(self) -> None:
        """Clone the specification repository."""
        logger.info(f"Cloning {self.repo_config.url} to {self.repo_path}")
        
        # Ensure parent directory exists
        self.repo_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._repo = git.Repo.clone_from(
                self.repo_config.url,
                self.repo_path,
                branch=self.repo_config.branch,
                depth=1  # Shallow clone for faster download
            )
            logger.info("Repository cloned successfully")
        except GitCommandError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
    
    def _pull_repository(self) -> None:
        """Pull latest changes to the repository."""
        logger.info(f"Pulling latest changes to {self.repo_path}")
        
        try:
            self._repo = git.Repo(self.repo_path)
            origin = self._repo.remotes.origin
            origin.pull()
            logger.info("Repository updated successfully")
        except GitCommandError as e:
            logger.error(f"Failed to pull repository: {e}")
            raise
    
    def get_spec_file(self, relative_path: str) -> Path:
        """
        Get path to a specific specification file.
        
        Args:
            relative_path: Path relative to repository root
            
        Returns:
            Absolute path to the file
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        self.fetch()  # Ensure repo is available
        
        file_path = self.repo_path / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"Spec file not found: {file_path}")
        
        return file_path
    
    def read_spec_file(self, relative_path: str) -> str:
        """
        Read contents of a specification file.
        
        Args:
            relative_path: Path relative to repository root
            
        Returns:
            File contents as string
        """
        file_path = self.get_spec_file(relative_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_eip1559_spec(self) -> str:
        """
        Extract EIP-1559 specification from the repository.
        
        Returns:
            EIP-1559 specification content
            
        Raises:
            FileNotFoundError: If spec file not found
        """
        # Try multiple possible locations for EIP-1559 spec
        possible_paths = [
            self.config.eip1559.spec_markdown_path,
            "network-upgrades/mainnet-upgrades/london.md",
            "EIPS/eip-1559.md",
            "src/ethereum/london/spec.py",
        ]
        
        for path in possible_paths:
            if not path:
                continue
            try:
                content = self.read_spec_file(path)
                logger.info(f"Found EIP-1559 spec at: {path}")
                return content
            except FileNotFoundError:
                continue
        
        raise FileNotFoundError(
            "Could not find EIP-1559 specification in any known location"
        )
    
    def parse_sections(self, content: str) -> List[SpecSection]:
        """
        Parse markdown content into sections.
        
        Args:
            content: Markdown content to parse
            
        Returns:
            List of SpecSection objects
        """
        sections = []
        lines = content.split('\n')
        
        current_section: Optional[SpecSection] = None
        current_content: List[str] = []
        line_start = 0
        
        for i, line in enumerate(lines):
            # Check for markdown headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save previous section if exists
                if current_section is not None:
                    current_section.content = '\n'.join(current_content).strip()
                    current_section.line_end = i - 1
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = SpecSection(
                    title=title,
                    content="",
                    level=level,
                    line_start=i,
                    line_end=i
                )
                current_content = []
                line_start = i
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_section is not None:
            current_section.content = '\n'.join(current_content).strip()
            current_section.line_end = len(lines) - 1
            sections.append(current_section)
        
        return sections
    
    def find_section_by_title(self, content: str, title_pattern: str) -> Optional[SpecSection]:
        """
        Find a section by title pattern.
        
        Args:
            content: Markdown content to search
            title_pattern: Regex pattern to match section title
            
        Returns:
            Matching SpecSection or None
        """
        sections = self.parse_sections(content)
        
        for section in sections:
            if re.search(title_pattern, section.title, re.IGNORECASE):
                return section
        
        return None
    
    def get_eip1559_sections(self) -> Dict[str, SpecSection]:
        """
        Get relevant EIP-1559 sections from the specification.
        
        Returns:
            Dictionary mapping section names to SpecSection objects
        """
        content = self.extract_eip1559_spec()
        sections = self.parse_sections(content)
        
        # Define patterns for important EIP-1559 sections
        section_patterns = {
            'base_fee': r'base\s*fee',
            'fee_burning': r'fee\s*burn|burning',
            'gas_pricing': r'gas\s*pricing|gas\s*fee',
            'transaction': r'transaction',
            'block': r'block',
            'specification': r'specification',
        }
        
        found_sections = {}
        for name, pattern in section_patterns.items():
            for section in sections:
                if re.search(pattern, section.title, re.IGNORECASE):
                    found_sections[name] = section
                    break
        
        return found_sections
    
    def get_file_list(self, pattern: str = "**/*.md") -> List[Path]:
        """
        Get list of files matching pattern in the repository.
        
        Args:
            pattern: Glob pattern to match
            
        Returns:
            List of file paths
        """
        self.fetch()
        return list(self.repo_path.glob(pattern))


class AsyncSpecFetcher:
    """
    Async version of SpecFetcher using aiohttp for GitHub API.
    
    Useful for fetching individual files without cloning entire repository.
    """
    
    GITHUB_RAW_URL = "https://raw.githubusercontent.com"
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize async spec fetcher.
        
        Args:
            config: Configuration instance
        """
        self.config = config or get_config()
        self.repo_config = self.config.repos.get('execution_specs')
        if not self.repo_config:
            raise ValueError("Execution specs repository not configured")
    
    async def fetch_file(self, session, relative_path: str) -> str:
        """
        Fetch a single file using GitHub raw content API.
        
        Args:
            session: aiohttp ClientSession
            relative_path: Path to file in repository
            
        Returns:
            File content as string
        """
        import aiohttp
        
        # Parse repo URL to get owner/repo
        url_parts = self.repo_config.url.replace('https://github.com/', '').split('/')
        owner, repo = url_parts[0], url_parts[1].replace('.git', '')
        
        raw_url = (
            f"{self.GITHUB_RAW_URL}/{owner}/{repo}/"
            f"{self.repo_config.branch}/{relative_path}"
        )
        
        headers = {}
        if self.config.github_token:
            headers['Authorization'] = f'token {self.config.github_token}'
        
        async with session.get(raw_url, headers=headers) as response:
            if response.status == 200:
                return await response.text()
            else:
                raise FileNotFoundError(
                    f"Failed to fetch {raw_url}: {response.status}"
                )
