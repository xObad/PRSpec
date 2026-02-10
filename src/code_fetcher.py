"""
Code fetcher for PRSpec.

Handles downloading and caching Ethereum client code from GitHub repositories.
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

import git
from git.exc import GitCommandError
import aiohttp

from .config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class CodeFile:
    """Represents a source code file."""
    path: Path
    content: str
    language: str
    last_modified: datetime
    commit_hash: Optional[str] = None


@dataclass
class FunctionInfo:
    """Information about a function in source code."""
    name: str
    signature: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    body: str


class CodeFetcher:
    """
    Fetches and manages Ethereum client source code.
    
    This class handles cloning/pulling the go-ethereum repository
    and extracting relevant source files.
    
    Attributes:
        config: Configuration instance
        repo_path: Local path to the cloned repository
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the code fetcher.
        
        Args:
            config: Configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
        self.repo_config = self.config.repos.get('go_ethereum')
        if not self.repo_config:
            raise ValueError("Go-ethereum repository not configured")
        
        self.repo_path = Path(self.repo_config.local_path)
        self._repo: Optional[git.Repo] = None
    
    def fetch(self, force_update: bool = False) -> Path:
        """
        Fetch or update the code repository.
        
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
        """Clone the code repository."""
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
    
    def checkout_commit(self, commit_hash: str) -> None:
        """
        Checkout a specific commit.
        
        Args:
            commit_hash: Git commit hash to checkout
        """
        self.fetch()
        
        if self._repo is None:
            self._repo = git.Repo(self.repo_path)
        
        logger.info(f"Checking out commit: {commit_hash}")
        try:
            self._repo.git.checkout(commit_hash)
            logger.info(f"Checked out {commit_hash}")
        except GitCommandError as e:
            logger.error(f"Failed to checkout commit: {e}")
            raise
    
    def checkout_pr(self, pr_number: int) -> str:
        """
        Checkout a specific PR.
        
        Args:
            pr_number: Pull request number
            
        Returns:
            Commit hash of the PR HEAD
        """
        self.fetch()
        
        if self._repo is None:
            self._repo = git.Repo(self.repo_path)
        
        logger.info(f"Fetching PR #{pr_number}")
        try:
            # Fetch PR refs
            self._repo.git.fetch(
                'origin',
                f'pull/{pr_number}/head:pr-{pr_number}'
            )
            self._repo.git.checkout(f'pr-{pr_number}')
            
            # Get commit hash
            commit_hash = self._repo.head.commit.hexsha
            logger.info(f"Checked out PR #{pr_number} at {commit_hash}")
            return commit_hash
        except GitCommandError as e:
            logger.error(f"Failed to checkout PR: {e}")
            raise
    
    def get_file(self, relative_path: str) -> CodeFile:
        """
        Get a specific source file.
        
        Args:
            relative_path: Path relative to repository root
            
        Returns:
            CodeFile object with file information
        """
        self.fetch()
        
        file_path = self.repo_path / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"Code file not found: {file_path}")
        
        # Detect language from extension
        language = self._detect_language(file_path.suffix)
        
        # Get file stats
        stat = file_path.stat()
        last_modified = datetime.fromtimestamp(stat.st_mtime)
        
        # Get commit hash if repo available
        commit_hash = None
        if self._repo is None:
            self._repo = git.Repo(self.repo_path)
        try:
            commit_hash = self._repo.head.commit.hexsha[:8]
        except GitCommandError:
            pass
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return CodeFile(
            path=file_path,
            content=content,
            language=language,
            last_modified=last_modified,
            commit_hash=commit_hash
        )
    
    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        language_map = {
            '.go': 'go',
            '.py': 'python',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.java': 'java',
            '.js': 'javascript',
            '.ts': 'typescript',
        }
        return language_map.get(extension.lower(), 'unknown')
    
    def get_eip1559_files(self) -> Dict[str, CodeFile]:
        """
        Get all EIP-1559 related source files.
        
        Returns:
            Dictionary mapping file names to CodeFile objects
        """
        files = {}
        
        for file_path in self.config.eip1559.code_files:
            try:
                code_file = self.get_file(file_path)
                files[file_path] = code_file
                logger.info(f"Loaded: {file_path}")
            except FileNotFoundError as e:
                logger.warning(f"File not found: {e}")
        
        return files
    
    def extract_functions(self, code_file: CodeFile, function_names: Optional[List[str]] = None) -> List[FunctionInfo]:
        """
        Extract function definitions from Go source code.
        
        Args:
            code_file: CodeFile to parse
            function_names: Optional list of function names to extract. If None, extracts all.
            
        Returns:
            List of FunctionInfo objects
        """
        if code_file.language != 'go':
            logger.warning(f"Function extraction for {code_file.language} not fully supported")
        
        functions = []
        content = code_file.content
        lines = content.split('\n')
        
        # Go function pattern: func Name(params) returns { ... }
        # Match function declarations (handles pointer returns like *big.Int)
        func_pattern = re.compile(
            r'^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(([^)]*)\)\s*([*\w]+(?:\[[^\]]*\])?(?:\.\w+)?|\([^)]*\))?\s*\{',
            re.MULTILINE
        )
        
        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            
            # Filter by name if specified
            if function_names and func_name not in function_names:
                continue
            
            # Find function boundaries
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1
            
            # Find matching closing brace
            brace_count = 1
            pos = match.end() - 1  # Position at opening brace
            
            while brace_count > 0 and pos < len(content) - 1:
                pos += 1
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
            
            end_line = content[:pos].count('\n') + 1
            
            # Extract function body
            body = '\n'.join(lines[start_line - 1:end_line])
            
            # Try to extract docstring (Go comments before function)
            docstring = None
            if start_line > 1:
                comment_lines = []
                for i in range(start_line - 2, -1, -1):
                    line = lines[i].strip()
                    if line.startswith('//'):
                        comment_lines.insert(0, line[2:].strip())
                    elif line == '' or line.startswith('package') or line.startswith('import'):
                        break
                    else:
                        break
                if comment_lines:
                    docstring = '\n'.join(comment_lines)
            
            signature = match.group(0).rstrip('{').strip()
            
            functions.append(FunctionInfo(
                name=func_name,
                signature=signature,
                start_line=start_line,
                end_line=end_line,
                docstring=docstring,
                body=body
            ))
        
        return functions
    
    def get_key_functions(self) -> Dict[str, List[FunctionInfo]]:
        """
        Get key EIP-1559 functions from all relevant files.
        
        Returns:
            Dictionary mapping file paths to lists of FunctionInfo
        """
        files = self.get_eip1559_files()
        key_function_names = set(self.config.eip1559.key_functions)
        
        result = {}
        for file_path, code_file in files.items():
            functions = self.extract_functions(code_file, list(key_function_names))
            if functions:
                result[file_path] = functions
                # Remove found functions from the set
                for func in functions:
                    key_function_names.discard(func.name)
        
        if key_function_names:
            logger.warning(f"Could not find functions: {key_function_names}")
        
        return result


class AsyncCodeFetcher:
    """
    Async version of CodeFetcher using aiohttp for GitHub API.
    
    Useful for fetching individual files without cloning entire repository.
    """
    
    GITHUB_API_URL = "https://api.github.com"
    GITHUB_RAW_URL = "https://raw.githubusercontent.com"
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize async code fetcher.
        
        Args:
            config: Configuration instance
        """
        self.config = config or get_config()
        self.repo_config = self.config.repos.get('go_ethereum')
        if not self.repo_config:
            raise ValueError("Go-ethereum repository not configured")
        
        # Parse repo URL
        url_parts = self.repo_config.url.replace('https://github.com/', '').split('/')
        self.owner = url_parts[0]
        self.repo = url_parts[1].replace('.git', '')
    
    async def fetch_file(self, session: aiohttp.ClientSession, relative_path: str, ref: Optional[str] = None) -> str:
        """
        Fetch a single file using GitHub raw content API.
        
        Args:
            session: aiohttp ClientSession
            relative_path: Path to file in repository
            ref: Git reference (branch, tag, or commit). Defaults to configured branch.
            
        Returns:
            File content as string
        """
        ref = ref or self.repo_config.branch
        
        raw_url = f"{self.GITHUB_RAW_URL}/{self.owner}/{self.repo}/{ref}/{relative_path}"
        
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
    
    async def fetch_pr_files(self, session: aiohttp.ClientSession, pr_number: int) -> List[Dict[str, Any]]:
        """
        Fetch list of files changed in a PR.
        
        Args:
            session: aiohttp ClientSession
            pr_number: Pull request number
            
        Returns:
            List of file change information
        """
        url = f"{self.GITHUB_API_URL}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/files"
        
        headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        if self.config.github_token:
            headers['Authorization'] = f'token {self.config.github_token}'
        
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise RuntimeError(f"Failed to fetch PR files: {response.status}")
    
    async def fetch_file_at_commit(self, session: aiohttp.ClientSession, relative_path: str, commit_hash: str) -> str:
        """
        Fetch a file at a specific commit.
        
        Args:
            session: aiohttp ClientSession
            relative_path: Path to file in repository
            commit_hash: Commit hash
            
        Returns:
            File content as string
        """
        return await self.fetch_file(session, relative_path, ref=commit_hash)
