import re
from typing import Optional, Dict, Any
from pathlib import Path
from github import Github
import os

def extract_repo_url(text: str) -> Optional[str]:
    """Extract GitHub repo URL from text"""
    words = text.split()
    return next((word for word in words if "github.com" in word), None)

def needs_repo_url(message: str) -> bool:
    """Check if message needs repo URL"""
    repo_keywords = {"repo", "repository", "github", "code", "project"}
    return (
        any(word.lower() in message.lower() for word in repo_keywords)
        or "github.com" in message
    )

def create_initial_state(repo_url: str, query: str, cached_data: Optional[Dict] = None, 
                        is_cached: bool = False) -> Dict[str, Any]:
    """Create initial state for analysis"""
    if is_cached and cached_data:
        return {
            "repo_url": repo_url,
            "query": query,
            "files": cached_data.get("files", []),
            "repo_context": cached_data.get("repo_context", {}),
            "query_type": "",
            "response": "",
            "messages": [],
            "context": [],
            "analysis_complete": True,
        }

    return {
        "repo_url": repo_url,
        "query": query,
        "files": [],
        "repo_context": {},
        "query_type": "",
        "response": "",
        "messages": [],
        "context": [],
        "analysis_complete": False,
    } 