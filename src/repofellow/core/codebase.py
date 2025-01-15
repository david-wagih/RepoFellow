"""Codebase context and analysis module"""
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import os

@dataclass
class FileInfo:
    """Information about a source code file"""
    path: str
    content: str
    language: str
    size: int

class CodebaseContext:
    """Handles codebase loading and context management"""
    
    def __init__(self, root_path: Path):
        self.root_path = Path(root_path)
        self.files: Dict[str, FileInfo] = {}
        self._load_files()
    
    def _load_files(self) -> None:
        """Load all relevant files from the codebase"""
        for file_path in self.root_path.rglob("*"):
            if self._should_include_file(file_path):
                relative_path = str(file_path.relative_to(self.root_path))
                content = self._read_file(file_path)
                language = self._detect_language(file_path)
                size = os.path.getsize(file_path)
                
                self.files[relative_path] = FileInfo(
                    path=relative_path,
                    content=content,
                    language=language,
                    size=size
                )
    
    def _should_include_file(self, path: Path) -> bool:
        """Determine if a file should be included in analysis"""
        if not path.is_file():
            return False
            
        # Skip common non-source directories
        exclude_dirs = {".git", "__pycache__", "node_modules", "venv", ".env"}
        if any(part in exclude_dirs for part in path.parts):
            return False
            
        # Skip binary and large files
        if path.stat().st_size > 1_000_000:  # Skip files > 1MB
            return False
            
        return True
    
    def _read_file(self, path: Path) -> str:
        """Read file content safely"""
        try:
            return path.read_text(encoding='utf-8')
        except Exception:
            return f"Error reading file: {path}"
    
    def _detect_language(self, path: Path) -> str:
        """Detect programming language from file extension"""
        extension_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".go": "Go",
            ".rs": "Rust",
            ".rb": "Ruby",
            ".php": "PHP",
            ".cs": "C#",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".html": "HTML",
            ".css": "CSS",
            ".sql": "SQL",
            ".md": "Markdown",
            ".json": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".toml": "TOML",
            ".xml": "XML",
        }
        return extension_map.get(path.suffix.lower(), "Unknown")
    
    def get_summary(self) -> str:
        """Generate a summary of the codebase"""
        total_files = len(self.files)
        languages = {}
        
        for file_info in self.files.values():
            languages[file_info.language] = languages.get(file_info.language, 0) + 1
        
        summary = [
            "# Codebase Summary\n",
            f"Total files: {total_files}\n",
            "\n## Language Distribution:\n"
        ]
        
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            if lang != "Unknown":
                percentage = (count / total_files) * 100
                summary.append(f"- {lang}: {count} files ({percentage:.1f}%)\n")
        
        return "".join(summary)
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a specific file"""
        file_info = self.files.get(file_path)
        return file_info.content if file_info else None
    
    def get_files_by_language(self, language: str) -> List[FileInfo]:
        """Get all files of a specific language"""
        return [
            file_info for file_info in self.files.values()
            if file_info.language.lower() == language.lower()
        ] 