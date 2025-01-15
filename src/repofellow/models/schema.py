from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Analyst(BaseModel):
    """Schema for repository analysts"""
    affiliation: str = Field(description="Primary area of expertise for the analyst.")
    name: str = Field(description="Name of the analyst")
    role: str = Field(description="Role of the analyst in analyzing the repository.")
    description: str = Field(description="Description of the analyst's focus areas and expertise.")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Team of analysts examining different aspects of the repository."
    )

class RepoFile(BaseModel):
    path: str = Field(description="File path in the repository")
    content: str = Field(description="Content of the file")
    language: Optional[str] = Field(description="Programming language or file type")

class RepoMetadata(BaseModel):
    name: str = Field(description="Repository name")
    description: str = Field(description="Repository description")
    stars: int = Field(description="Number of stars")
    forks: int = Field(description="Number of forks")
    topics: List[str] = Field(description="Repository topics/tags") 