"""
Parser module for PRSpec.

Handles parsing of specification documents and source code files.
"""

import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from tree_sitter import Language, Parser, Tree, Node
import tree_sitter_go as ts_go

from .spec_fetcher import SpecSection
from .code_fetcher import CodeFile, FunctionInfo

logger = logging.getLogger(__name__)


class SpecRequirementType(Enum):
    """Types of requirements in specifications."""
    MUST = "must"
    SHOULD = "should"
    MAY = "may"
    MUST_NOT = "must_not"
    SHOULD_NOT = "should_not"


@dataclass
class SpecRequirement:
    """Represents a single requirement from a specification."""
    text: str
    req_type: SpecRequirementType
    section: str
    line_number: int
    context: str = ""


@dataclass
class ParsedSpec:
    """Parsed specification document."""
    title: str
    sections: List[SpecSection]
    requirements: List[SpecRequirement]
    formulas: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedCode:
    """Parsed source code file."""
    file_path: str
    language: str
    functions: List[FunctionInfo]
    structs: List[Dict[str, Any]]
    constants: List[Dict[str, Any]]
    comments: List[Dict[str, Any]]
    ast: Optional[Tree] = None


class SpecParser:
    """
    Parser for Ethereum specification documents.
    
    Extracts requirements, formulas, and structured content from markdown specs.
    """
    
    # Regex patterns for requirement detection
    REQUIREMENT_PATTERNS = {
        SpecRequirementType.MUST: re.compile(r'\b(must|required to|shall)\b', re.IGNORECASE),
        SpecRequirementType.SHOULD: re.compile(r'\b(should|recommended)\b', re.IGNORECASE),
        SpecRequirementType.MAY: re.compile(r'\b(may|optional|can)\b', re.IGNORECASE),
        SpecRequirementType.MUST_NOT: re.compile(r'\b(must not|shall not|prohibited)\b', re.IGNORECASE),
        SpecRequirementType.SHOULD_NOT: re.compile(r'\b(should not|not recommended)\b', re.IGNORECASE),
    }
    
    # Pattern for mathematical formulas
    FORMULA_PATTERNS = [
        re.compile(r'`([^`]+=[^`]+)`'),  # Inline code with equals
        re.compile(r'```math\s*(.*?)\s*```', re.DOTALL),  # Math blocks
        re.compile(r'\$\$(.*?)\$\$', re.DOTALL),  # LaTeX display math
        re.compile(r'\$(.*?)\$'),  # LaTeX inline math
    ]
    
    def __init__(self):
        """Initialize the specification parser."""
        pass
    
    def parse(self, content: str, title: str = "") -> ParsedSpec:
        """
        Parse specification content.
        
        Args:
            content: Markdown specification content
            title: Document title
            
        Returns:
            ParsedSpec object with extracted information
        """
        sections = self._parse_sections(content)
        requirements = self._extract_requirements(content, sections)
        formulas = self._extract_formulas(content)
        metadata = self._extract_metadata(content)
        
        return ParsedSpec(
            title=title,
            sections=sections,
            requirements=requirements,
            formulas=formulas,
            metadata=metadata
        )
    
    def _parse_sections(self, content: str) -> List[SpecSection]:
        """
        Parse markdown content into sections.
        
        Args:
            content: Markdown content
            
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
    
    def _extract_requirements(self, content: str, sections: List[SpecSection]) -> List[SpecRequirement]:
        """
        Extract requirement statements from content.
        
        Args:
            content: Full markdown content
            sections: Parsed sections
            
        Returns:
            List of SpecRequirement objects
        """
        requirements = []
        lines = content.split('\n')
        
        for section in sections:
            section_lines = section.content.split('\n')
            
            for i, line in enumerate(section_lines):
                line_number = section.line_start + i + 1
                
                # Check each requirement type
                for req_type, pattern in self.REQUIREMENT_PATTERNS.items():
                    if pattern.search(line):
                        # Get context (surrounding lines)
                        context_start = max(0, i - 2)
                        context_end = min(len(section_lines), i + 3)
                        context = '\n'.join(section_lines[context_start:context_end])
                        
                        requirements.append(SpecRequirement(
                            text=line.strip(),
                            req_type=req_type,
                            section=section.title,
                            line_number=line_number,
                            context=context
                        ))
                        break  # Only capture first matching type
        
        return requirements
    
    def _extract_formulas(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical formulas from content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of formula dictionaries
        """
        formulas = []
        
        for pattern in self.FORMULA_PATTERNS:
            for match in pattern.finditer(content):
                formula_text = match.group(1) if match.groups() else match.group(0)
                line_number = content[:match.start()].count('\n') + 1
                
                formulas.append({
                    'text': formula_text.strip(),
                    'line_number': line_number,
                    'raw': match.group(0)
                })
        
        return formulas
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from specification (YAML frontmatter, etc.).
        
        Args:
            content: Markdown content
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Check for YAML frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if frontmatter_match:
            try:
                import yaml
                metadata = yaml.safe_load(frontmatter_match.group(1))
            except Exception as e:
                logger.warning(f"Failed to parse YAML frontmatter: {e}")
        
        # Extract EIP number if present
        eip_match = re.search(r'EIP[-\s]*(\d+)', content, re.IGNORECASE)
        if eip_match:
            metadata['eip_number'] = int(eip_match.group(1))
        
        # Extract status
        status_match = re.search(r'Status:\s*(\w+)', content, re.IGNORECASE)
        if status_match:
            metadata['status'] = status_match.group(1)
        
        return metadata
    
    def find_section_by_title(self, parsed_spec: ParsedSpec, pattern: str) -> Optional[SpecSection]:
        """
        Find a section by title pattern.
        
        Args:
            parsed_spec: Parsed specification
            pattern: Regex pattern to match
            
        Returns:
            Matching section or None
        """
        for section in parsed_spec.sections:
            if re.search(pattern, section.title, re.IGNORECASE):
                return section
        return None


class CodeParser:
    """
    Parser for source code files using tree-sitter.
    
    Extracts functions, structs, constants, and other code elements.
    """
    
    def __init__(self):
        """Initialize the code parser with tree-sitter languages."""
        self.parsers: Dict[str, Parser] = {}
        self._init_parsers()
    
    def _init_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        try:
            # Initialize Go parser
            go_language = Language(ts_go.language())
            go_parser = Parser(go_language)
            self.parsers['go'] = go_parser
        except Exception as e:
            logger.warning(f"Failed to initialize Go parser: {e}")
        
        # Python parser can be added similarly
        try:
            import tree_sitter_python as ts_python
            py_language = Language(ts_python.language())
            py_parser = Parser(py_language)
            self.parsers['python'] = py_parser
        except ImportError:
            logger.debug("Python parser not available")
    
    def parse(self, code_file: CodeFile) -> ParsedCode:
        """
        Parse a source code file.
        
        Args:
            code_file: CodeFile to parse
            
        Returns:
            ParsedCode object with extracted information
        """
        language = code_file.language
        content = code_file.content
        
        # Parse with tree-sitter if available
        ast = None
        if language in self.parsers:
            try:
                ast = self.parsers[language].parse(bytes(content, 'utf8'))
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed: {e}")
        
        # Extract code elements
        functions = self._extract_functions(content, language)
        structs = self._extract_structs(content, language)
        constants = self._extract_constants(content, language)
        comments = self._extract_comments(content, language)
        
        return ParsedCode(
            file_path=str(code_file.path),
            language=language,
            functions=functions,
            structs=structs,
            constants=constants,
            comments=comments,
            ast=ast
        )
    
    def _extract_functions(self, content: str, language: str) -> List[FunctionInfo]:
        """
        Extract function definitions from source code.
        
        Args:
            content: Source code content
            language: Programming language
            
        Returns:
            List of FunctionInfo objects
        """
        functions = []
        lines = content.split('\n')
        
        if language == 'go':
            # Go function pattern - matches functions with optional receiver, parameters, and return types
            # Handles: func Name(params) return { }, func (r *Receiver) Name(params) *Return { }
            func_pattern = re.compile(
                r'^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(([^)]*)\)\s*([*\w]+(?:\[[^\]]*\])?(?:\.\w+)?|\([^)]*\))?\s*\{',
                re.MULTILINE
            )
            
            for match in func_pattern.finditer(content):
                func_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find matching closing brace
                brace_count = 1
                pos = match.end() - 1
                
                while brace_count > 0 and pos < len(content) - 1:
                    pos += 1
                    if content[pos] == '{':
                        brace_count += 1
                    elif content[pos] == '}':
                        brace_count -= 1
                
                end_line = content[:pos].count('\n') + 1
                body = '\n'.join(lines[start_line - 1:end_line])
                
                # Extract docstring
                docstring = self._extract_docstring(lines, start_line - 1)
                
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
    
    def _extract_docstring(self, lines: List[str], func_line: int) -> Optional[str]:
        """
        Extract docstring/comments before a function.
        
        Args:
            lines: All lines of the file
            func_line: Line number where function starts (0-indexed)
            
        Returns:
            Docstring text or None
        """
        comment_lines = []
        
        for i in range(func_line - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('//'):
                comment_lines.insert(0, line[2:].strip())
            elif line == '' or line.startswith('package') or line.startswith('import'):
                break
            elif not line.startswith('//'):
                break
        
        return '\n'.join(comment_lines) if comment_lines else None
    
    def _extract_structs(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Extract struct/type definitions from source code.
        
        Args:
            content: Source code content
            language: Programming language
            
        Returns:
            List of struct dictionaries
        """
        structs = []
        lines = content.split('\n')
        
        if language == 'go':
            # Go struct pattern
            struct_pattern = re.compile(
                r'^type\s+(\w+)\s+struct\s*\{',
                re.MULTILINE
            )
            
            for match in struct_pattern.finditer(content):
                struct_name = match.group(1)
                start_line = content[:match.start()].count('\n') + 1
                
                # Find closing brace
                brace_count = 1
                pos = match.end() - 1
                
                while brace_count > 0 and pos < len(content) - 1:
                    pos += 1
                    if content[pos] == '{':
                        brace_count += 1
                    elif content[pos] == '}':
                        brace_count -= 1
                
                end_line = content[:pos].count('\n') + 1
                body = '\n'.join(lines[start_line:end_line - 1])
                
                # Extract fields
                fields = []
                for line in body.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('//'):
                        field_match = re.match(r'(\w+)\s+(\S+)', line)
                        if field_match:
                            fields.append({
                                'name': field_match.group(1),
                                'type': field_match.group(2)
                            })
                
                structs.append({
                    'name': struct_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'fields': fields,
                    'body': body
                })
        
        return structs
    
    def _extract_constants(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Extract constant definitions from source code.
        
        Args:
            content: Source code content
            language: Programming language
            
        Returns:
            List of constant dictionaries
        """
        constants = []
        lines = content.split('\n')
        
        if language == 'go':
            # Go const pattern
            const_pattern = re.compile(
                r'^const\s+(?:\([^)]*\)|(\w+)\s+(\w+)\s*=\s*(.+))',
                re.MULTILINE
            )
            
            for match in const_pattern.finditer(content):
                line_number = content[:match.start()].count('\n') + 1
                
                if match.group(1):  # Single const
                    constants.append({
                        'name': match.group(1),
                        'type': match.group(2),
                        'value': match.group(3).strip(),
                        'line': line_number
                    })
                else:  # Const block
                    # Parse const block
                    block_start = match.end()
                    block_end = content.find(')', block_start)
                    if block_end > 0:
                        block_content = content[block_start:block_end]
                        for line in block_content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('//'):
                                parts = line.split('=')
                                if len(parts) == 2:
                                    name_type = parts[0].strip().split()
                                    if len(name_type) >= 1:
                                        constants.append({
                                            'name': name_type[0],
                                            'type': name_type[1] if len(name_type) > 1 else 'inferred',
                                            'value': parts[1].strip(),
                                            'line': line_number
                                        })
        
        return constants
    
    def _extract_comments(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Extract comments from source code.
        
        Args:
            content: Source code content
            language: Programming language
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        if language == 'go':
            # Go single-line comments
            comment_pattern = re.compile(r'//(.+)$', re.MULTILINE)
            
            for match in comment_pattern.finditer(content):
                line_number = content[:match.start()].count('\n') + 1
                comments.append({
                    'text': match.group(1).strip(),
                    'line': line_number,
                    'type': 'single_line'
                })
            
            # Go multi-line comments
            multiline_pattern = re.compile(r'/\*(.*?)\*/', re.DOTALL)
            
            for match in multiline_pattern.finditer(content):
                line_number = content[:match.start()].count('\n') + 1
                comments.append({
                    'text': match.group(1).strip(),
                    'line': line_number,
                    'type': 'multi_line'
                })
        
        return comments
    
    def find_function(self, parsed_code: ParsedCode, func_name: str) -> Optional[FunctionInfo]:
        """
        Find a specific function in parsed code.
        
        Args:
            parsed_code: Parsed code object
            func_name: Name of function to find
            
        Returns:
            FunctionInfo or None
        """
        for func in parsed_code.functions:
            if func.name == func_name:
                return func
        return None
