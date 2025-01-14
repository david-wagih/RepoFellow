from pathlib import Path
import os

class CodebaseContext:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.files = {}
        self.ignore_patterns = self._load_gitignore()
        # Add default ignore patterns
        self.default_ignores = {
            # Dependency directories
            'node_modules/',
            'venv/',
            '.env/',
            '.venv/',
            'env/',
            '__pycache__/',
            'dist/',
            'build/',
            '.next/',
            '.nuxt/',
            
            # Build outputs
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '*.so',
            '*.dll',
            '*.dylib',
            
            # IDE directories
            '.idea/',
            '.vscode/',
            '.vs/',
            
            # System files
            '.DS_Store',
            'Thumbs.db',
            
            # Version control
            '.git/',
            '.svn/',
            '.hg/',
            
            # Package files
            '*.egg-info/',
            '*.egg',
            
            # Large files
            '*.log',
            '*.sqlite',
            '*.db'
        }
        self._load_codebase()
    
    def _load_gitignore(self) -> set:
        """Load patterns from .gitignore file"""
        gitignore_patterns = set()
        gitignore_path = self.base_path / '.gitignore'
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        gitignore_patterns.add(line)
        
        return gitignore_patterns
    
    def _should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns"""
        relative_path = str(Path(file_path).relative_to(self.base_path))
        
        # Check against both .gitignore and default patterns
        all_patterns = self.ignore_patterns.union(self.default_ignores)
        
        for pattern in all_patterns:
            # Handle directory patterns
            if pattern.endswith('/'):
                if pattern[:-1] in relative_path.split(os.sep):
                    return True
            # Handle file patterns with wildcards
            elif '*' in pattern:
                import fnmatch
                if fnmatch.fnmatch(relative_path, pattern):
                    return True
            # Handle exact matches
            elif pattern == relative_path:
                return True
        
        return False
    
    def _is_binary_file(self, file_path: str) -> bool:
        """Check if file is binary"""
        try:
            with open(file_path, 'tr') as check_file:
                check_file.read(1024)
                return False
        except UnicodeDecodeError:
            return True
    
    def _load_codebase(self):
        """Load all relevant code files from the codebase"""
        python_files = []
        
        for file_path in self.base_path.rglob('*.py'):
            if (
                file_path.is_file() 
                and not self._should_ignore(str(file_path))
                and file_path.stat().st_size <= 1024 * 1024  # 1MB limit
                and not self._is_binary_file(str(file_path))
            ):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():  # Only store non-empty files
                            self.files[str(file_path)] = content
                            python_files.append(str(file_path))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        self._python_files = python_files
        return python_files
    
    def get_file_content(self, file_path: str) -> str:
        """Get content of a specific file"""
        return self.files.get(str(Path(file_path)))
    
    def get_all_files(self):
        """Get all loaded files and their content"""
        return self.files
    
    def get_summary(self) -> str:
        """Get a summary of the loaded codebase"""
        total_files = len(self.files)
        extensions = {}
        total_lines = 0
        
        for file_path, content in self.files.items():
            ext = Path(file_path).suffix
            extensions[ext] = extensions.get(ext, 0) + 1
            total_lines += len(content.splitlines())
        
        summary = f"Loaded {total_files} files with {total_lines} total lines\n"
        summary += "File types:\n"
        for ext, count in sorted(extensions.items()):
            summary += f"  {ext}: {count} files\n"
        
        return summary 
    
    def get_python_files(self):
        """Get list of Python files"""
        if not hasattr(self, '_python_files'):
            self._load_codebase()
        return self._python_files 