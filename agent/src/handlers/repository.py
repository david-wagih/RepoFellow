from typing import Dict, Any
from github import Github
import os
from models.state import RepoAnalysisState
from models.schema import RepoFile, RepoMetadata
from tools.analysis import RepoAnalysisTools

def analyze_repository_node(state: RepoAnalysisState) -> Dict[str, Any]:
    """Analyze repository and update state with results"""
    try:
        # Get GitHub token
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            return {**state, "error": "GitHub token not found"}

        # Initialize GitHub client
        g = Github(github_token)
        
        # Extract owner and repo name
        parts = state["repo_url"].split("github.com/")[-1].split("/")
        owner, repo_name = parts[0], parts[1].replace(".git", "")
        
        # Get repository
        repo = g.get_repo(f"{owner}/{repo_name}")
        
        # Get repository metadata
        metadata = RepoMetadata(
            name=repo.name,
            description=repo.description or "",
            stars=repo.stargazers_count,
            forks=repo.forks_count,
            topics=repo.get_topics()
        )

        # Get repository files
        files = []
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
                try:
                    files.append(RepoFile(
                        path=file_content.path,
                        content=file_content.decoded_content.decode(),
                        language=file_content.path.split(".")[-1] if "." in file_content.path else None
                    ))
                except Exception:
                    # Skip binary files or files that can't be decoded
                    continue

        # Analyze repository
        analysis_results = {
            "metadata": metadata.dict(),
            "files": [f.dict() for f in files],
            "structure": RepoAnalysisTools.analyze_structure(files),
            "dependencies": RepoAnalysisTools.analyze_dependencies(files),
        }

        return {
            **state,
            "files": files,
            "repo_context": analysis_results,
            "analysis_complete": True,
        }

    except Exception as e:
        return {**state, "error": f"Repository analysis failed: {str(e)}"} 