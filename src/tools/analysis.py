from typing import Any, Dict, Set, List
from models.schema import RepoFile
from collections import Counter
from pathlib import Path

class RepoAnalysisTools:
    @staticmethod
    def analyze_dependencies(files: List[RepoFile]) -> Dict[str, Set[str]]:
        """Analyze project dependencies from package files"""
        dependencies = {}
        for file in files:
            if file.path.endswith(("requirements.txt", "package.json", "Cargo.toml")):
                deps = RepoAnalysisTools._parse_dependencies(file)
                dependencies[file.path] = deps
        return dependencies

    @staticmethod
    def _parse_dependencies(file: RepoFile) -> Set[str]:
        """Parse dependencies from package files"""
        deps = set()
        if file.path.endswith("requirements.txt"):
            for line in file.content.splitlines():
                if line and not line.startswith("#"):
                    deps.add(line.split("==")[0])
        return deps

    @staticmethod
    def analyze_structure(files: List[RepoFile]) -> Dict[str, Any]:
        """Analyze code structure of files"""
        analysis = {
            "file_tree": {},
            "components": [],
            "metrics": {}
        }
        
        # Build file tree and analyze components
        for file in files:
            parts = Path(file.path).parts
            current = analysis["file_tree"]
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = {"type": "file", "language": file.language}

        # Calculate metrics
        analysis["metrics"] = {
            "total_files": len(files),
            "languages": dict(Counter(f.language for f in files if f.language)),
            "avg_file_size": sum(len(f.content) for f in files) / len(files) if files else 0
        }

        return analysis 