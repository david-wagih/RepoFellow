import datetime
import traceback
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, Dict, Set, Any, Union
from typing_extensions import TypedDict
import re
import os
import base64
from pathlib import Path
from io import BytesIO
import tempfile
from collections import Counter
import certifi

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph imports
from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph

# Other dependencies
from github import Github
from graphviz import Digraph

# 2. Configuration
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000)


# 3. Schema Classes
class Analyst(BaseModel):
    """Schema for repository analysts"""

    affiliation: str = Field(description="Primary area of expertise for the analyst.")
    name: str = Field(description="Name of the analyst")
    role: str = Field(description="Role of the analyst in analyzing the repository.")
    description: str = Field(
        description="Description of the analyst's focus areas and expertise."
    )

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


# 4. State Management Classes
class BaseState(TypedDict):
    """Base state class with common fields"""

    messages: Annotated[List[Any], "append"]
    context: Annotated[List[str], "append"]
    files: Annotated[List[RepoFile], "merge"]


class RepoAnalysisState(BaseState):
    """State for repository analysis"""

    repo_url: str
    repo_context: Dict[str, Any]
    query: str
    query_type: str
    response: str
    analysis_complete: bool
    error: Optional[str] = None


class InterviewState(BaseState):
    """State for analyst interviews"""

    max_num_turns: int
    analyst: Analyst
    interview: str
    sections: Annotated[List[str], "append"]
    analyst_id: int


class ChatFlowState(TypedDict):
    """Enhanced state management for chat flow"""

    conversation_id: str
    messages: Annotated[List[BaseMessage], "append"]
    current_context: Dict[str, Any]
    repo_context: Optional[Dict[str, Any]]
    source_type: str
    source_path: str
    analysis_type: str
    generated_artifacts: Dict[str, Any]
    waiting_for_repo: bool
    last_query: str
    error: Optional[str]
    # Add new fields for better flow control
    analysis_stage: str  # Track current analysis stage
    pending_operations: List[str]  # Track operations in queue
    analysis_depth: int  # Control depth of analysis


class GenerateAnalystsState(TypedDict):
    """State for analyst generation"""

    repo_url: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    files: List[RepoFile]


# 5. Repository Memory Management
class RepositoryMemory:
    """Manages memory of analyzed repositories with TTL and size limits"""

    def __init__(self, max_cache_size: int = 100, ttl_hours: int = 24):
        self.repositories: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = max_cache_size
        self.ttl_hours = ttl_hours
        self._cleanup_timer = 0  # Add cleanup timer

    def _cleanup_old_entries(self) -> None:
        """Clean up expired entries"""
        current_time = datetime.datetime.now()
        expired_keys = [
            key
            for key, data in self.repositories.items()
            if (current_time - data["last_analyzed"]).total_seconds()
            > self.ttl_hours * 3600
        ]
        for key in expired_keys:
            del self.repositories[key]

    def save_analysis(self, repo_url: str, analysis_data: Dict[str, Any]):
        if len(self.repositories) >= self.max_cache_size:
            oldest_key = min(
                self.repositories.keys(),
                key=lambda k: self.repositories[k]["last_analyzed"],
            )
            del self.repositories[oldest_key]

        self.repositories[repo_url] = {
            "last_analyzed": datetime.datetime.now(),
            "repo_context": analysis_data.get("repo_context", {}),
            "files": analysis_data.get("files", []),
            "analysis_complete": analysis_data.get("analysis_complete", False),
        }

    def get_analysis(self, repo_url: str) -> Optional[Dict[str, Any]]:
        if repo_url not in self.repositories:
            return None

        analysis = self.repositories[repo_url]
        age = datetime.datetime.now() - analysis["last_analyzed"]

        if age.total_seconds() > self.ttl_hours * 3600:
            del self.repositories[repo_url]
            return None

        return analysis


# Initialize global repository memory
repo_memory = RepositoryMemory()


# 6. Helper Functions
def extract_repo_url(text: str) -> Optional[str]:
    """Extract GitHub repo URL from text"""
    words = text.split()
    return next((word for word in words if "github.com" in word), None)


def create_initial_state(
    repo_url: str,
    query: str,
    cached_data: Optional[Dict] = None,
    is_cached: bool = False,
) -> Dict[str, Any]:
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


def initialize_chat() -> ChatFlowState:
    """Initialize a new chat session"""
    return {
        "conversation_id": str(datetime.datetime.now().timestamp()),
        "messages": [],
        "current_context": {},
        "source_type": "",
        "source_path": "",
        "analysis_type": "chat",
        "generated_artifacts": {},
        "waiting_for_repo": False,
        "last_query": "",
        "analysis_stage": "initial",
        "pending_operations": [],
        "analysis_depth": 0,
    }


def needs_repo_url(message: str) -> bool:
    """Check if message needs repo URL"""
    repo_keywords = {"repo", "repository", "github", "code", "project"}
    return (
        any(word.lower() in message.lower() for word in repo_keywords)
        or "github.com" in message
    )


# 7. Core Analysis Functions
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
    def generate_architecture_diagram(files: List[RepoFile]) -> Dict[str, Any]:
        """Generate architecture diagram"""
        try:
            if not files:
                raise ValueError("No files provided for diagram generation")

            # Convert files to RepoFile objects if needed
            repo_files = []
            for file in files:
                if isinstance(file, RepoFile):
                    repo_files.append(file)
                elif isinstance(file, dict):
                    repo_files.append(RepoFile(**file))

            # Generate diagram
            dot = Digraph(comment="Repository Structure")
            dot.attr(rankdir="TB")

            # Add nodes and edges
            for file in repo_files:
                parts = Path(file.path).parts
                for i in range(len(parts)):
                    node_id = f"node_{hash(''.join(parts[:i+1]))}"
                    dot.node(node_id, parts[i])
                    if i > 0:
                        parent_id = f"node_{hash(''.join(parts[:i]))}"
                        dot.edge(parent_id, node_id)

            # Save diagram
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                dot.render(tmp.name, format="png", cleanup=True)
                with open(f"{tmp.name}.png", "rb") as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")

            return {
                "image_base64": image_base64,
                "image_format": "png",
            }

        except Exception as e:
            print(f"Error generating architecture diagram: {str(e)}")
            return {"error": str(e)}


# 8. Query Handlers
def handle_general_query(state: RepoAnalysisState) -> Dict:
    """Enhanced general query handler with web search integration"""
    try:
        query = state.get("query", "").lower()

        # Check for greetings first
        greeting_patterns = {
            r"^hi\b": "Hi! I'm your GitHub repository assistant. How can I help you today?",
            r"^hello\b": "Hello! I'm here to help you analyze GitHub repositories. What would you like to know?",
            r"^hey\b": "Hey there! Need help with a GitHub repository?",
            r"^greetings\b": "Greetings! I'm your GitHub analysis assistant. How can I assist you?",
            r"^good (morning|afternoon|evening)\b": "Good day! How can I help you with repository analysis?",
            r"^how are you": "I'm functioning well, thank you! Ready to help you analyze repositories. What can I do for you?",
        }

        # Check for greetings
        for pattern, response in greeting_patterns.items():
            if re.match(pattern, query):
                return {
                    "response": response,
                    "messages": state.get("messages", [])
                    + [AIMessage(content=response)],
                }

        # If the query needs web search, redirect to search_web node
        web_search_indicators = [
            "what is",
            "how to",
            "tell me about",
            "difference between",
            "compare",
            "explain",
            "search",
            "find information",
        ]

        if any(indicator in query.lower() for indicator in web_search_indicators):
            # Perform web search directly
            search_results = perform_web_search(query)
            if search_results["success"] and search_results["results"]:
                web_context = "\n\n".join(
                    [
                        f"Source: {result['url']}\n{result['content']}"
                        for result in search_results["results"]
                    ]
                )

                system_prompt = """You are a technical assistant with access to web search results.
                Provide a comprehensive answer based on the search results.
                Always cite your sources using [Web: URL].
                Keep the response focused and relevant to the query.
                
                Web Search Results:
                {web_context}
                
                Query: {query}"""

                response = llm.invoke(
                    [
                        SystemMessage(
                            content=system_prompt.format(
                                web_context=web_context, query=query
                            )
                        ),
                        HumanMessage(content=query),
                    ]
                )

                return {
                    "response": response.content,
                    "messages": state.get("messages", [])
                    + [AIMessage(content=response.content)],
                }

        # If we have repository context, try to answer from it
        if state.get("repo_context"):
            system_prompt = """You are a technical assistant analyzing a repository.
            Use the provided context to answer questions accurately and concisely.
            If you cannot find the information in the context, indicate that you'll search the web for more details.
            
            Repository Context:
            {context}
            
            Question: {query}"""

            response = llm.invoke(
                [
                    SystemMessage(
                        content=system_prompt.format(
                            context=str(state.get("repo_context", {})), query=query
                        )
                    ),
                    HumanMessage(content=query),
                ]
            )

            return {
                "response": response.content,
                "messages": state.get("messages", [])
                + [AIMessage(content=response.content)],
            }

        # If no repository context, provide a helpful response
        if "github" in query or "repository" in query:
            response = "I can help you analyze GitHub repositories. Please provide a repository URL to get started."
        else:
            response = """I'm your GitHub repository analysis assistant. I can help you with:
1. Analyzing GitHub repositories
2. Understanding code structure
3. Generating documentation
4. Creating visualizations
5. Answering questions about code

Please provide a GitHub repository URL when you're ready to analyze one."""

        return {
            "response": response,
            "messages": state.get("messages", []) + [AIMessage(content=response)],
        }

    except Exception as e:
        error_msg = f"Error in general query handler: {str(e)}"
        print(error_msg)
        return {
            "response": "I encountered an error processing your request. Please try again.",
            "messages": state.get("messages", [])
            + [
                AIMessage(
                    content="I encountered an error processing your request. Please try again."
                )
            ],
            "error": str(e),
        }


# 9. Additional Query Handlers
def handle_docs_query(state: RepoAnalysisState) -> Dict:
    """Generate documentation based on request"""
    try:
        query = state.get("query", "")
        repo_context = state.get("repo_context", {})
        files = state.get("files", [])
        repo_url = state.get("repo_url", "")

        if not repo_context:
            return {
                "response": "Repository hasn't been analyzed yet. Please provide a repository URL first.",
                "error": "No repository context available",
            }

        # Extract existing markdown content with error handling
        try:
            markdown_files = _extract_markdown_content(files)
            existing_readme = markdown_files.get("README.md", "")
        except Exception as e:
            print(f"Error extracting markdown content: {e}")
            existing_readme = ""

        system_prompt = """Generate comprehensive documentation based on the repository analysis.
        Focus on clarity, structure, and technical accuracy.
        
        Existing README Content:
        {existing_readme}
        
        Repository Context:
        {context}
        
        Files Available:
        {files}
        
        Documentation Request:
        {query}
        
        Format the response in markdown. Include:
        1. Project overview
        2. Key features
        3. Installation instructions
        4. Usage examples
        5. Architecture overview
        6. Contributing guidelines
        
        Keep the response focused and technical. Do not include any meta-commentary about the documentation itself."""

        response = llm.invoke(
            [
                SystemMessage(
                    content=system_prompt.format(
                        existing_readme=existing_readme,
                        context=str(repo_context),
                        files="\n".join(f"- {f.path}" for f in files),
                        query=query,
                    )
                ),
                HumanMessage(content=query),
            ]
        )

        new_content = response.content

        # Create a PR with the new documentation
        branch_name = f"docs/update-readme-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        pr_result = _create_pull_request(
            repo_url=repo_url,
            branch_name=branch_name,
            file_path="README.md",
            content=new_content,  # Use just the generated content
            commit_message="Update documentation with AI-generated content",
        )

        if pr_result["success"]:
            # Keep meta-commentary only in the response, not in the PR content
            return {
                "response": f"I've generated new documentation and created a pull request. You can review and merge the changes here: {pr_result['pr_url']}",
                "messages": state.get("messages", []) + [AIMessage(content=new_content)],
                "pr_url": pr_result["pr_url"],
                "generated_content": new_content
            }
        else:
            return {
                "response": f"I've generated new documentation but couldn't create a pull request (Error: {pr_result['error']}).",
                "messages": state.get("messages", []) + [AIMessage(content=new_content)],
                "generated_content": new_content
            }

    except Exception as e:
        return {
            "response": f"Error generating documentation: {str(e)}",
            "error": str(e),
        }


def handle_graph_query(state: RepoAnalysisState) -> Dict:
    """Handle requests for technical graphs and visualizations with fallback"""
    try:
        query = state.get("query", "")
        files = state.get("files", [])

        # Check if we have repository data
        if not files:
            return {
                "response": "I need a repository to analyze before I can create any visualizations. Please provide a GitHub repository URL."
            }

        try:
            # Try generating visualization
            if "dependency" in query.lower():
                dependencies = RepoAnalysisTools.analyze_dependencies(files)
                graph_data = RepoVisualizer.create_dependency_graph(dependencies)
            else:
                graph_data = RepoAnalysisTools.generate_architecture_diagram(files)

            if "error" not in graph_data:
                response_text = (
                    "Here's the requested visualization:\n\n"
                    f"![Repository Visualization](data:image/{graph_data['image_format']};"
                    f"base64,{graph_data['image_base64']})\n\n"
                    "This diagram shows the "
                    f"{'dependency relationships' if 'dependency' in query.lower() else 'architectural structure'} "
                    "of the repository."
                )
                return {
                    "response": response_text,
                    "has_image": True,
                    "image_data": graph_data,
                }

        except Exception as viz_error:
            print(
                f"Visualization generation failed, falling back to text description: {str(viz_error)}"
            )

        # Fallback to text-based description
        structure_description = _generate_text_structure(files)
        return {
            "response": f"""I apologize, but I cannot generate a visual diagram in this environment. 
However, I can describe the repository structure:

{structure_description}""",
            "has_image": False,
        }

    except Exception as e:
        error_msg = f"Error handling graph query: {str(e)}"
        print(error_msg)
        return {"response": error_msg, "has_image": False}


def _generate_text_structure(files: List[RepoFile]) -> str:
    """Generate a text-based description of the repository structure"""
    try:
        # Build directory tree
        tree = {}
        for file in files:
            parts = Path(file.path).parts
            current = tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = "file"

        # Convert tree to text representation
        def tree_to_text(node, prefix="", is_last=True):
            lines = []
            if isinstance(node, dict):
                items = list(node.items())
                for i, (name, child) in enumerate(items):
                    is_last_item = i == len(items) - 1
                    lines.append(f"{prefix}{'└── ' if is_last_item else '├── '}{name}")
                    if isinstance(child, dict):
                        extension = "    " if is_last_item else "│   "
                        lines.extend(
                            tree_to_text(child, prefix + extension, is_last_item)
                        )
            return lines

        structure_lines = tree_to_text(tree)

        # Add summary statistics
        file_types = Counter(Path(f.path).suffix for f in files if Path(f.path).suffix)
        stats = f"""
Repository Statistics:
- Total files: {len(files)}
- File types: {', '.join(f'{ext} ({count})' for ext, count in file_types.most_common())}
"""

        return "Repository Structure:\n" + "\n".join(structure_lines) + stats

    except Exception as e:
        return f"Error generating text structure: {str(e)}"


def handle_code_query(state: RepoAnalysisState) -> Dict:
    """Find and return relevant code snippets"""
    query = state.get("query", "")
    files = state.get("files", [])

    system_prompt = (
        "You are a code analysis expert. For the given query:\n"
        "1. Find the most relevant code snippets\n"
        "2. Present them in proper code blocks with language tags\n"
        "3. Provide a brief, precise explanation below each snippet\n"
        "4. If suggesting improvements, show them in separate code blocks\n\n"
        "Format your response as:\n"
        "1. Code block with original code\n"
        "2. Brief explanation (1-2 sentences)\n"
        "3. Improvements (if any) in a separate code block\n\n"
        "Keep explanations concise and technical."
    )

    # Filter relevant files based on query
    relevant_files = []
    for file in files:
        if any(keyword in file.content.lower() for keyword in query.lower().split()):
            relevant_files.append(file)

    if not relevant_files:
        return {"response": "No relevant code found for your query."}

    # Prepare context with relevant code snippets
    code_context = "\n\n".join(
        f"File: {file.path}\n```{file.language or ''}\n{file.content[:500]}...\n```"
        for file in relevant_files[:3]  # Limit to 3 most relevant files
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\n\nAvailable Code:\n{code_context}"),
        ]
    )

    return {"response": response.content}


# 10. Repository Analysis Functions
def analyze_repository(state: RepoAnalysisState) -> Dict[str, Any]:
    """Perform initial repository analysis with improved error handling"""
    try:
        if not isinstance(state, dict):
            raise ValueError("Invalid state object")

        files = state.get("files", [])
        if isinstance(files, dict) and "merge" in files:
            files = files["merge"]

        if not files:
            return {
                **state,
                "error": "No files found in repository",
                "response": "No files found in repository. Please check if the repository is accessible and not empty.",
                "analysis_complete": False,
            }

        # Build repository context
        repo_context = {
            "structure": {},
            "dependencies": {},
            "architecture": "",
            "file_summary": {},
            "key_components": [],
        }

        # Analyze code structure
        for file in files:
            if file.language in ["py", "js", "ts", "java", "cpp"]:
                structure = _analyze_file_structure(file)
                repo_context["structure"][file.path] = structure

        # Analyze dependencies
        repo_context["dependencies"] = RepoAnalysisTools.analyze_dependencies(files)

        # Generate file summaries
        for file in files[:10]:  # Limit to first 10 files
            summary = _analyze_single_snippet({"context": []}, file)
            repo_context["file_summary"][file.path] = summary

        # Extract key components
        key_components = []
        for file_path, structure in repo_context["structure"].items():
            if "classes" in structure:
                key_components.extend(
                    [
                        {"type": "class", "name": cls, "file": file_path}
                        for cls in structure["classes"]
                    ]
                )
            if "functions" in structure:
                key_components.extend(
                    [
                        {"type": "function", "name": func, "file": file_path}
                        for func in structure["functions"]
                    ]
                )
        repo_context["key_components"] = key_components

        # Generate analysis summary
        summary = f"""Repository Analysis Complete:
- Files analyzed: {len(files)}
- Key components found: {len(key_components)}
- File types: {', '.join(set(f.language for f in files if f.language))}

Repository is ready for queries. You can ask about:
1. Code structure and organization
2. Dependencies and relationships
3. Specific components or files
4. Request documentation or diagrams"""

        return {
            **state,
            "repo_context": repo_context,
            "messages": [AIMessage(content=summary)],
            "analysis_complete": True,
            "context": [summary],
        }

    except Exception as e:
        return {
            **state,
            "error": str(e),
            "response": f"An error occurred during analysis: {str(e)}",
            "analysis_complete": False,
        }


def _analyze_file_structure(file: RepoFile) -> Dict[str, List[str]]:
    """Analyze code structure of a file"""
    structure: Dict[str, List[str]] = {
        "classes": [],
        "functions": [],
        "imports": [],
        "variables": [],
    }

    try:
        lines = file.content.split("\n")
        for line in lines:
            line = line.strip()

            # Detect classes
            if line.startswith("class "):
                class_name = line.split("class ")[1].split("(")[0].strip(":")
                structure["classes"].append(class_name)

            # Detect functions
            elif line.startswith("def "):
                func_name = line.split("def ")[1].split("(")[0]
                structure["functions"].append(func_name)

            # Detect imports
            elif line.startswith(("import ", "from ")):
                structure["imports"].append(line)

            # Detect global variables
            elif "=" in line and not line.startswith((" ", "\t", "def", "class")):
                var_name = line.split("=")[0].strip()
                structure["variables"].append(var_name)

    except Exception as e:
        print(f"Error analyzing file structure: {str(e)}")

    return structure


def _analyze_single_snippet(state: Dict, file: RepoFile, max_length: int = 500) -> str:
    """Analyze a single code snippet"""
    try:
        truncated_content = file.content[:max_length]
        if len(file.content) > max_length:
            truncated_content += "\n... (content truncated)"

        system_prompt = """Analyze this code snippet and provide a brief, focused summary.
        Focus on:
        - Main purpose
        - Key functionality
        - Important patterns
        - Potential issues
        
        File: {path}
        Language: {language}
        
        Provide a concise summary (max 100 words)."""

        response = llm.invoke(
            [
                SystemMessage(
                    content=system_prompt.format(
                        path=file.path, language=file.language or "unknown"
                    )
                ),
                HumanMessage(content=truncated_content),
            ]
        )

        return response.content

    except Exception as e:
        return f"Error analyzing {file.path}: {str(e)}"


# Add these functions after the _analyze_single_snippet function


def process_chat_message(state: ChatFlowState) -> Dict[str, Any]:
    """Process incoming chat messages and determine next steps"""
    try:
        messages = state["messages"]
        if not messages:
            return state

        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return state

        content = last_message.content

        # Check if we need a repo URL
        if state["waiting_for_repo"]:
            repo_url = extract_repo_url(content)
            if repo_url:
                return {
                    **state,
                    "waiting_for_repo": False,
                    "source_path": repo_url,
                    "source_type": "github",
                }
            else:
                return {
                    **state,
                    "messages": state["messages"]
                    + [
                        AIMessage(
                            content="Please provide a valid GitHub repository URL."
                        )
                    ],
                }

        # Check if message mentions repository
        if needs_repo_url(content) and not state["source_path"]:
            return {
                **state,
                "waiting_for_repo": True,
                "last_query": content,
                "messages": state["messages"]
                + [
                    AIMessage(
                        content="I notice you're asking about a repository. Please provide the GitHub repository URL."
                    )
                ],
            }

        return state

    except Exception as e:
        return {**state, "error": str(e)}


def analyze_repo_node(state: ChatFlowState) -> Dict[str, Any]:
    """Analyze repository if needed"""
    try:
        if not state["source_path"] or state["repo_context"]:
            return state

        # Analyze repository
        repo_url = state["source_path"]
        analysis_result = analyze_repository(
            {"repo_url": repo_url, "files": [], "messages": state["messages"]}
        )

        return {
            **state,
            "repo_context": analysis_result.get("repo_context"),
            "current_context": analysis_result.get("repo_context", {}),
        }

    except Exception as e:
        return {**state, "error": str(e)}


async def handle_repo_request(
    message: str, state: Dict[str, Any], github_token: Optional[str]
) -> Dict[str, Any]:
    """Handle requests involving repository URLs"""
    repo_url = extract_repo_url(message)
    if repo_url:
        state["source_type"] = "github"
        state["source_path"] = repo_url
        analysis_result = await analyze_and_query_repo(
            repo_url, "initial_analysis", github_token
        )
        state["current_context"] = analysis_result.get("repo_context", {})
        return state
    else:
        return {
            **state,
            "waiting_for_repo": True,
            "last_query": message,
            "messages": state["messages"]
            + [
                AIMessage(
                    content="I notice you're asking about a repository. "
                    "Please provide the GitHub repository URL you'd like me to examine."
                )
            ],
        }


async def handle_waiting_for_repo(
    message: str, state: Dict[str, Any], github_token: Optional[str]
) -> Dict[str, Any]:
    """Handle state when waiting for repo URL"""
    repo_url = extract_repo_url(message)
    if not repo_url:
        return {
            **state,
            "messages": state["messages"]
            + [
                AIMessage(
                    content="I still need a valid GitHub repository URL. "
                    "Please provide one in the format: https://github.com/owner/repo"
                )
            ],
        }

    # Process the repo URL
    state["waiting_for_repo"] = False
    state["source_type"] = "github"
    state["source_path"] = repo_url

    # Analyze repository
    analysis_result = await analyze_and_query_repo(
        repo_url, "initial_analysis", github_token
    )
    state["current_context"] = analysis_result.get("repo_context", {})

    # Process original query if exists
    if state["last_query"]:
        return await chat_interaction(state["last_query"], state, github_token)

    return state


# 11. Graph Setup and Routing
def classify_query(state: RepoAnalysisState) -> Dict:
    """Enhanced query classification using LLM for better understanding"""
    try:
        query = state.get("query", "").lower()

        # System prompt for classification
        system_prompt = """You are a query classifier for a GitHub repository analysis assistant.
        Analyze the user's query and classify it into one of these categories:
        
        1. greeting: General greetings, pleasantries, or casual conversation
        2. repo_request: Requests to analyze or load a GitHub repository
        3. code: Questions about specific code, implementation, or code examples
        4. docs: Requests for documentation or explanation of functionality
        5. graph: Requests for visualizations, diagrams, or structural representations
        6. web_search: General questions that need web search for answers
        
        Consider these examples:
        - "Hi there" -> greeting
        - "Can you analyze this repo: github.com/user/repo" -> repo_request
        - "Show me the class structure" -> graph
        - "How does the authentication work?" -> code
        - "Generate documentation for the API" -> docs
        - "What's the difference between React and Vue?" -> web_search
        
        Respond with just the category name in lowercase."""

        # Get classification from LLM
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        )

        # Extract classification
        query_type = response.content.strip().lower()

        # Debug log
        print(f"Query: '{query}' classified as: {query_type}")

        return {"query_type": query_type}

    except Exception as e:
        print(f"Query classification error: {str(e)}")
        return {"query_type": "general"}


def route_query(state: RepoAnalysisState) -> str:
    """Enhanced routing logic based on LLM classification"""
    try:
        query_type = state.get("query_type", "").lower()

        # Define routing map
        routing_map = {
            "greeting": "handle_general",
            "repo_request": "load_repo",
            "code": "handle_code",
            "docs": "handle_docs",
            "graph": "handle_graph",
            "web_search": "search_web",
            "general": "handle_general",
        }

        # Get route from map
        route = routing_map.get(query_type, "handle_general")

        # Debug log
        print(f"Routing query type '{query_type}' to: {route}")

        return route

    except Exception as e:
        print(f"Routing error: {str(e)}")
        return "handle_general"


def build_main_graph() -> StateGraph:
    """Build the main analysis graph with improved routing"""
    builder = StateGraph(RepoAnalysisState)

    # Add nodes
    builder.add_node("classify_query", classify_query)
    builder.add_node("load_repository", load_repository_node)
    builder.add_node("analyze_repository", analyze_repository)
    builder.add_node("search_web", search_web_node)
    builder.add_node("handle_general", handle_general_query)
    builder.add_node("handle_docs", handle_docs_query)
    builder.add_node("handle_graph", handle_graph_query)
    builder.add_node("handle_code", handle_code_query)

    # Start with query classification
    builder.add_edge(START, "classify_query")

    # Add conditional routing from classify_query
    builder.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "handle_general": "handle_general",
            "handle_docs": "handle_docs",
            "handle_graph": "handle_graph",
            "handle_code": "handle_code",
            "load_repo": "load_repository",
            "search_web": "search_web",
        },
    )

    # Repository loading and analysis flow
    builder.add_edge("load_repository", "analyze_repository")
    builder.add_edge("analyze_repository", "handle_general")

    # Web search can lead to general handler
    builder.add_edge("search_web", "handle_general")

    # All handlers lead to END
    builder.add_edge("handle_general", END)
    builder.add_edge("handle_docs", END)
    builder.add_edge("handle_graph", END)
    builder.add_edge("handle_code", END)

    return builder.compile()


def perform_web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Perform web search using Tavily for additional context"""
    try:
        # Initialize Tavily search
        tavily_search = TavilySearchResults(max_results=max_results)

        # Enhance query for better technical results
        enhanced_query = f"technical documentation {query}"

        # Perform search and get results
        raw_results = tavily_search.invoke(enhanced_query)

        # Format results - handle both list and dict responses
        formatted_results = []
        if isinstance(raw_results, list):
            for result in raw_results:
                if isinstance(result, dict):
                    formatted_results.append(
                        {
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0),
                        }
                    )
        elif isinstance(raw_results, dict):
            formatted_results.append(
                {
                    "title": raw_results.get("title", ""),
                    "content": raw_results.get("content", ""),
                    "url": raw_results.get("url", ""),
                    "score": raw_results.get("score", 0),
                }
            )

        if formatted_results:
            return {
                "success": True,
                "results": formatted_results,
                "query": enhanced_query,
            }
        else:
            return {
                "success": False,
                "error": "No results found",
                "query": enhanced_query,
            }

    except Exception as e:
        print(f"Web search error: {str(e)}")
        return {"success": False, "error": str(e), "query": query}


def search_web_node(state: RepoAnalysisState) -> Dict[str, Any]:
    """Node for handling web searches and integrating results"""
    try:
        query = state.get("query", "")

        # Perform web search
        search_results = perform_web_search(query)

        if search_results["success"] and search_results["results"]:
            # Format search results for LLM
            web_context = "\n\n".join(
                [
                    f"Source: {result['url']}\n{result['content']}"
                    for result in search_results["results"]
                ]
            )

            # Generate response using LLM
            system_prompt = """You are a technical assistant with access to web search results.
            Provide a comprehensive answer based on the search results.
            Always cite your sources using [Web: URL].
            Keep the response focused and relevant to the query.
            
            Web Search Results:
            {web_context}
            
            Query: {query}"""

            response = llm.invoke(
                [
                    SystemMessage(
                        content=system_prompt.format(
                            web_context=web_context, query=query
                        )
                    ),
                    HumanMessage(content=query),
                ]
            )

            return {
                **state,
                "web_context": web_context,
                "search_results": search_results["results"],
                "response": response.content,
                "messages": state.get("messages", [])
                + [AIMessage(content=response.content)],
            }

        else:
            error_msg = search_results.get("error", "No relevant information found")
            return {
                **state,
                "response": f"I wasn't able to find relevant information: {error_msg}",
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"I wasn't able to find relevant information: {error_msg}"
                    )
                ],
            }

    except Exception as e:
        error_msg = f"Error in web search: {str(e)}"
        print(error_msg)
        return {
            **state,
            "response": f"An error occurred during web search: {str(e)}",
            "messages": state.get("messages", [])
            + [AIMessage(content=f"An error occurred during web search: {str(e)}")],
        }


# 12. Graph Building
def build_main_graph() -> StateGraph:
    """Build the main analysis graph with improved routing"""
    builder = StateGraph(RepoAnalysisState)

    # Add nodes
    builder.add_node("classify_query", classify_query)
    builder.add_node("load_repository", load_repository_node)
    builder.add_node("analyze_repository", analyze_repository)
    builder.add_node("search_web", search_web_node)
    builder.add_node("handle_general", handle_general_query)
    builder.add_node("handle_docs", handle_docs_query)
    builder.add_node("handle_graph", handle_graph_query)
    builder.add_node("handle_code", handle_code_query)

    # Start with query classification
    builder.add_edge(START, "classify_query")

    # Add conditional routing from classify_query
    builder.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "handle_general": "handle_general",
            "handle_docs": "handle_docs",
            "handle_graph": "handle_graph",
            "handle_code": "handle_code",
            "load_repo": "load_repository",
            "search_web": "search_web",
        },
    )

    # Repository loading and analysis flow
    builder.add_edge("load_repository", "analyze_repository")
    builder.add_edge("analyze_repository", "handle_general")

    # Web search can lead to general handler
    builder.add_edge("search_web", "handle_general")

    # All handlers lead to END
    builder.add_edge("handle_general", END)
    builder.add_edge("handle_docs", END)
    builder.add_edge("handle_graph", END)
    builder.add_edge("handle_code", END)

    return builder.compile()


# 13. Main Execution Functions
async def analyze_and_query_repo(
    repo_url: Optional[str],
    query: str,
    github_token: Optional[str] = None,
) -> Dict:
    """Execute repository analysis and query workflow"""
    try:
        check_environment()
        # Validate inputs
        if not repo_url and not query:
            raise ValueError("Must provide either repo_url or query with repo URL")

        # Extract repo URL from query if needed
        if not repo_url:
            repo_url = extract_repo_url(query)
            if not repo_url:
                return {
                    "messages": [
                        AIMessage(
                            content="Please provide a valid GitHub repository URL."
                        )
                    ],
                    "needs_repo_url": True,
                }

        # Check cache first
        cached = repo_memory.get_analysis(repo_url)
        if cached:
            print(f"Using cached analysis for {repo_url}")
            return create_initial_state(repo_url, query, cached, True)

        # Initialize new analysis
        print(f"Starting new analysis for {repo_url}")
        initial_state = create_initial_state(repo_url, query)

        # Set up GitHub token
        if github_token:
            os.environ["GITHUB_TOKEN"] = github_token
        elif not os.getenv("GITHUB_TOKEN"):
            raise ValueError("GitHub token not provided")

        # Get the compiled graph
        graph = build_main_graph()

        # Execute analysis
        result = await graph.ainvoke(initial_state)

        # Cache successful analysis
        if result.get("analysis_complete"):
            repo_memory.save_analysis(repo_url, result)

        return result

    except Exception as e:
        error_msg = f"Error in analyze_and_query_repo: {str(e)}"
        print(error_msg)
        print("Stack trace:", traceback.format_exc())
        return {"messages": [AIMessage(content=error_msg)], "analysis_complete": False}


async def chat_interaction(
    message: str,
    state: Optional[ChatFlowState] = None,
    github_token: Optional[str] = None,
) -> ChatFlowState:
    """Improved chat interaction handler"""
    try:
        # Initialize or update state
        if state is None:
            state = {
                "conversation_id": str(datetime.datetime.now().timestamp()),
                "messages": [],
                "current_context": {},
                "repo_context": None,
                "source_type": "",
                "source_path": "",
                "analysis_type": "chat",
                "generated_artifacts": {},
                "waiting_for_repo": False,
                "last_query": "",
                "analysis_stage": "initial",
                "pending_operations": [],
                "analysis_depth": 0,
                "error": None,
            }

        # Set GitHub token if provided
        if github_token:
            os.environ["GITHUB_TOKEN"] = github_token

        # Add new message
        state["messages"].append(HumanMessage(content=message))

        # Get chat graph
        chat_graph = build_chat_graph()

        # Execute graph
        result = await chat_graph.ainvoke(state)

        return result

    except Exception as e:
        error_msg = f"Error in chat interaction: {str(e)}"
        print(error_msg)
        print("Stack trace:", traceback.format_exc())

        return {
            **state,
            "messages": state["messages"]
            + [AIMessage(content=f"An error occurred: {str(e)}")],
            "error": str(e),
        }


class RepoVisualizer:
    """Visualization tools for repository analysis"""

    @staticmethod
    def create_dependency_graph(dependencies: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Create visual dependency graph"""
        try:
            dot = Digraph(comment="Project Dependencies")
            dot.attr(rankdir="LR")

            # Add nodes for each package file
            for package_file in dependencies:
                file_name = Path(package_file).name
                dot.node(package_file, file_name)

                # Add dependency nodes and edges
                for dep in dependencies[package_file]:
                    dep_id = dep.replace("-", "_").replace(".", "_")
                    dot.node(dep_id, dep)
                    dot.edge(package_file, dep_id)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                dot.render(tmp.name, format="png", cleanup=True)
                with open(f"{tmp.name}.png", "rb") as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")

            return {
                "image_base64": image_base64,
                "image_format": "png",
            }

        except Exception as e:
            return {"error": str(e)}


def load_repository_node(state: RepoAnalysisState) -> Dict:
    """Load repository with improved error handling"""
    try:
        # Ensure SSL certificates are properly configured
        check_environment()

        # Get repo URL from query if not directly provided
        repo_url = state.get("repo_url") or extract_repo_url(state.get("query", ""))
        if not repo_url:
            return {
                "error": "No repository URL provided",
                "response": "Please provide a GitHub repository URL to analyze.",
            }

        # Initialize GitHub client with error handling
        try:
            g = Github(os.getenv("GITHUB_TOKEN"))
            parts = repo_url.split("github.com/")[-1].split("/")
            owner, repo_name = parts[0], parts[1].replace(".git", "")
            repo = g.get_repo(f"{owner}/{repo_name}")

            # Load repository contents
            files = []

            def process_contents(contents):
                for content in contents:
                    try:
                        if content.type == "dir":
                            process_contents(repo.get_contents(content.path))
                        else:
                            file_content = base64.b64decode(content.content).decode(
                                "utf-8"
                            )
                            extension = Path(content.path).suffix.lstrip(".")
                            files.append(
                                RepoFile(
                                    path=content.path,
                                    content=file_content,
                                    language=extension,
                                )
                            )
                    except Exception as e:
                        print(f"Error processing {content.path}: {str(e)}")
                        continue

            # Process all repository contents
            process_contents(repo.get_contents(""))

            if not files:
                return {
                    "error": "No files found in repository",
                    "response": "Repository appears to be empty or inaccessible.",
                }

            return {
                **state,
                "files": files,
                "repo_url": repo_url,
                "response": f"Successfully loaded repository with {len(files)} files.",
                "analysis_complete": False,
            }

        except Exception as e:
            return {
                "error": f"GitHub authentication error: {str(e)}",
                "response": "Failed to authenticate with GitHub. Please check your GITHUB_TOKEN.",
            }

    except Exception as e:
        error_msg = f"Error loading repository: {str(e)}"
        print(f"Detailed error: {traceback.format_exc()}")
        return {
            "error": error_msg,
            "response": f"Failed to load repository: {str(e)}. Please check your connection and try again.",
        }


async def handle_waiting_for_repo(
    message: str, state: Dict[str, Any], github_token: Optional[str]
) -> Dict[str, Any]:
    """Handle state when waiting for repo URL"""
    repo_url = extract_repo_url(message)
    if not repo_url:
        return {
            **state,
            "messages": state["messages"]
            + [
                AIMessage(
                    content="I still need a valid GitHub repository URL. "
                    "Please provide one in the format: https://github.com/owner/repo"
                )
            ],
        }

    # Process the repo URL
    state["waiting_for_repo"] = False
    state["source_type"] = "github"
    state["source_path"] = repo_url

    # Analyze repository
    analysis_result = await analyze_and_query_repo(
        repo_url, "initial_analysis", github_token
    )
    state["current_context"] = analysis_result.get("repo_context", {})

    # Process original query if exists
    if state["last_query"]:
        return await chat_interaction(state["last_query"], state, github_token)

    return state


async def handle_repo_request(
    message: str, state: Dict[str, Any], github_token: Optional[str]
) -> Dict[str, Any]:
    """Handle requests involving repository URLs"""
    repo_url = extract_repo_url(message)
    if repo_url:
        state["source_type"] = "github"
        state["source_path"] = repo_url
        analysis_result = await analyze_and_query_repo(
            repo_url, "initial_analysis", github_token
        )
        state["current_context"] = analysis_result.get("repo_context", {})
        return state
    else:
        return {
            **state,
            "waiting_for_repo": True,
            "last_query": message,
            "messages": state["messages"]
            + [
                AIMessage(
                    content="I notice you're asking about a repository. "
                    "Please provide the GitHub repository URL you'd like me to examine."
                )
            ],
        }


def build_enhanced_chat_graph() -> StateGraph:
    """Build an improved chat interaction graph with better query classification flow"""
    chat_builder = StateGraph(ChatFlowState)

    # Core nodes
    chat_builder.add_node("validate_input", validate_input_node)
    chat_builder.add_node("classify_query", classify_query_node)
    chat_builder.add_node("load_repository", load_repository_node)
    chat_builder.add_node("analyze_repository", analyze_repository_node)

    # Handler nodes
    chat_builder.add_node("handle_general", handle_general_query)
    chat_builder.add_node("handle_docs", handle_docs_query)
    chat_builder.add_node("handle_graph", handle_graph_query)
    chat_builder.add_node("handle_code", handle_code_query)

    # Error handling
    chat_builder.add_node("handle_error", handle_error_node)
    chat_builder.add_node("recover_state", recover_state_node)

    # Define the flow
    chat_builder.add_edge(START, "validate_input")
    chat_builder.add_edge("validate_input", "classify_query")

    # Route to repository loading if needed
    chat_builder.add_conditional_edges(
        "classify_query",
        needs_repository_load,
        {"load": "load_repository", "skip": "analyze_repository"},
    )

    # Connect repository nodes
    chat_builder.add_edge("load_repository", "analyze_repository")

    # Route to appropriate handler based on query type
    chat_builder.add_conditional_edges(
        "analyze_repository",
        route_to_handler,
        {
            "general": "handle_general",
            "docs": "handle_docs",
            "graph": "handle_graph",
            "code": "handle_code",
            "error": "handle_error",
        },
    )

    # All handlers lead to END
    chat_builder.add_edge("handle_general", END)
    chat_builder.add_edge("handle_docs", END)
    chat_builder.add_edge("handle_graph", END)
    chat_builder.add_edge("handle_code", END)

    # Error recovery flow
    chat_builder.add_edge("handle_error", "recover_state")
    chat_builder.add_edge(
        "recover_state", "classify_query"
    )  # Return to classification after recovery

    return chat_builder.compile()


def needs_repository_load(state: ChatFlowState) -> str:
    """Determine if repository needs to be loaded"""
    try:
        if not state.get("repo_context"):
            message = state["messages"][-1].content
            if "github.com" in message or needs_repo_url(message):
                return "load"
        return "skip"
    except Exception as e:
        return "error"


def route_to_handler(state: ChatFlowState) -> str:
    """Route to appropriate query handler with improved visualization detection"""
    try:
        query_type = state.get("query_type", "")
        message = state["messages"][-1].content.lower()

        # Special handling for visualization requests
        visualization_keywords = {
            "diagram",
            "graph",
            "chart",
            "flow",
            "visualize",
            "draw",
            "display",
            "plot",
            "structure",
            "visualization",
        }

        if any(keyword in message for keyword in visualization_keywords):
            return "graph"

        # Default handler mapping
        handler_map = {
            "general": "general",
            "documentation": "docs",
            "visualization": "graph",
            "code": "code",
        }

        return handler_map.get(query_type, "general")

    except Exception as e:
        print(f"Routing error: {str(e)}")
        return "error"


def classify_query_node(state: ChatFlowState) -> Dict[str, Any]:
    """Enhanced query classification node with better pattern matching"""
    try:
        message = state["messages"][-1].content.lower()

        # Define classification patterns with more comprehensive matches
        patterns = {
            # Visualization/Graph patterns
            r"(draw|create|generate|show|display|visualize|plot).*(diagram|graph|chart|flow|visualization|structure)": "visualization",
            r"(diagram|graph|chart|flow|visualization).*(repository|code|structure|dependency|architecture)": "visualization",
            # Documentation patterns
            r"(document|documentation|describe|explain|detail|summarize)": "documentation",
            # Code patterns
            r"(code|implement|function|class|method|snippet)": "code",
            # General patterns (default)
            r"(explain|what|how|why)": "general",
        }

        # Determine query type with priority
        query_type = "general"  # default
        for pattern, qtype in patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                query_type = qtype
                # Give priority to visualization if matches
                if qtype == "visualization":
                    break

        print(
            f"Classified query type: {query_type} for message: {message}"
        )  # Debug log

        return {**state, "query_type": query_type, "analysis_stage": "query_classified"}

    except Exception as e:
        return {**state, "error": f"Query classification failed: {str(e)}"}


def handle_error_node(state: ChatFlowState) -> Dict[str, Any]:
    """Improved error handling with recovery options"""
    try:
        if state.get("error"):
            error_msg = state["error"]
            recovery_options = determine_recovery_options(error_msg)

            return {
                **state,
                "messages": state["messages"]
                + [
                    AIMessage(
                        content=f"An error occurred: {error_msg}\n\nI can try to:"
                    )
                ]
                + [AIMessage(content=f"- {option}") for option in recovery_options],
                "pending_operations": recovery_options,
                "error": None,
            }
        return state
    except Exception as e:
        return {
            **state,
            "messages": state["messages"]
            + [AIMessage(content=f"Critical error in error handler: {str(e)}")],
            "error": str(e),
        }


def recover_state_node(state: ChatFlowState) -> Dict[str, Any]:
    """Handle state recovery after errors"""
    try:
        # Attempt to recover last valid state
        if state.get("repo_context"):
            return {
                **state,
                "analysis_stage": "recovered",
                "messages": state["messages"]
                + [
                    AIMessage(content="Successfully recovered previous analysis state.")
                ],
            }
        # Reinitialize if no valid state exists
        return initialize_chat()
    except Exception as e:
        return {**state, "error": f"Recovery failed: {str(e)}"}


def determine_recovery_options(error_msg: str) -> List[str]:
    """Determine possible recovery options based on error"""
    recovery_options = []

    error_patterns = {
        "token": ["Refresh GitHub token", "Use alternative authentication"],
        "rate limit": ["Wait and retry", "Use different token"],
        "not found": ["Verify repository URL", "Check repository visibility"],
        "permission": ["Request repository access", "Use public repository"],
        "timeout": ["Retry with longer timeout", "Analyze smaller portion"],
        "memory": ["Reduce analysis scope", "Process in batches"],
    }

    for pattern, options in error_patterns.items():
        if pattern.lower() in error_msg.lower():
            recovery_options.extend(options)

    if not recovery_options:
        recovery_options = ["Restart analysis", "Try different repository"]

    return recovery_options


def build_chat_graph() -> StateGraph:
    """Build the chat interaction graph"""
    return build_enhanced_chat_graph()


def analyze_structure(state: ChatFlowState) -> Dict[str, Any]:
    """Analyze repository structure in detail"""
    try:
        files = state.get("files", [])
        if not files:
            return {"error": "No files available for analysis"}

        analysis = {"file_tree": {}, "components": [], "patterns": [], "metrics": {}}

        # Build file tree
        for file in files:
            parts = Path(file.path).parts
            current = analysis["file_tree"]
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = {"type": "file", "language": file.language}

        # Analyze components and patterns
        for file in files:
            if file.language in ["py", "js", "java", "cpp", "ts"]:
                structure = _analyze_file_structure(file)
                if structure.get("classes"):
                    analysis["components"].extend(
                        [
                            {"type": "class", "name": cls, "file": file.path}
                            for cls in structure["classes"]
                        ]
                    )
                if structure.get("functions"):
                    analysis["components"].extend(
                        [
                            {"type": "function", "name": func, "file": file.path}
                            for func in structure["functions"]
                        ]
                    )

        # Calculate metrics
        analysis["metrics"] = {
            "total_files": len(files),
            "languages": dict(Counter(f.language for f in files if f.language)),
            "avg_file_size": (
                sum(len(f.content) for f in files) / len(files) if files else 0
            ),
        }

        return analysis

    except Exception as e:
        return {"error": f"Structure analysis failed: {str(e)}"}


def analyze_dependencies(state: ChatFlowState) -> Dict[str, Any]:
    """Analyze project dependencies and their relationships"""
    try:
        files = state.get("files", [])
        if not files:
            return {"error": "No files available for dependency analysis"}

        analysis = {
            "direct_dependencies": {},
            "dev_dependencies": {},
            "dependency_graph": {},
            "metrics": {},
        }

        # Analyze package files
        for file in files:
            if file.path.endswith(("requirements.txt", "package.json", "Cargo.toml")):
                deps = RepoAnalysisTools._parse_dependencies(file)
                analysis["direct_dependencies"][file.path] = deps

        # Build dependency graph
        for file_path, deps in analysis["direct_dependencies"].items():
            analysis["dependency_graph"][file_path] = {
                "dependencies": list(deps),
                "type": "production",
            }

        # Calculate metrics
        analysis["metrics"] = {
            "total_dependencies": sum(
                len(deps) for deps in analysis["direct_dependencies"].values()
            ),
            "files_with_deps": len(analysis["direct_dependencies"]),
            "unique_deps": (
                len(set().union(*analysis["direct_dependencies"].values()))
                if analysis["direct_dependencies"]
                else 0
            ),
        }

        return analysis

    except Exception as e:
        return {"error": f"Dependency analysis failed: {str(e)}"}


def generate_visualizations(state: ChatFlowState) -> Dict[str, Any]:
    """Generate various visualizations of the repository"""
    try:
        files = state.get("files", [])
        if not files:
            return {"error": "No files available for visualization"}

        visualizations = {
            "architecture": None,
            "dependencies": None,
            "components": None,
            "metrics": None,
        }

        # Generate architecture diagram
        arch_result = RepoAnalysisTools.generate_architecture_diagram(files)
        if "error" not in arch_result:
            visualizations["architecture"] = arch_result

        # Generate dependency graph if dependencies exist
        if state.get("dependency_analysis"):
            dep_result = RepoVisualizer.create_dependency_graph(
                state["dependency_analysis"].get("direct_dependencies", {})
            )
            if "error" not in dep_result:
                visualizations["dependencies"] = dep_result

        # Generate component diagram
        if state.get("structure_analysis", {}).get("components"):
            components_dot = Digraph(comment="Component Relationships")
            for comp in state["structure_analysis"]["components"]:
                components_dot.node(comp["name"], f"{comp['type']}: {comp['name']}")

            # Save component diagram
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                components_dot.render(tmp.name, format="png", cleanup=True)
                with open(f"{tmp.name}.png", "rb") as f:
                    image_data = f.read()
                visualizations["components"] = {
                    "image_base64": base64.b64encode(image_data).decode("utf-8"),
                    "image_format": "png",
                }

        return visualizations

    except Exception as e:
        return {"error": f"Visualization generation failed: {str(e)}"}


def validate_input_node(state: ChatFlowState) -> Dict[str, Any]:
    """Validate incoming chat messages and prepare for processing"""
    try:
        # Check for messages
        if not state["messages"]:
            return {
                **state,
                "error": "No messages found in state",
                "analysis_stage": "error",
            }

        last_message = state["messages"][-1]

        # Validate message type
        if not isinstance(last_message, HumanMessage):
            return {
                **state,
                "error": "Last message must be from human",
                "analysis_stage": "error",
            }

        # Validate content
        content = last_message.content.strip()
        if not content:
            return {
                **state,
                "error": "Message content cannot be empty",
                "analysis_stage": "error",
            }

        # Update state with validated content
        return {
            **state,
            "last_query": content,
            "analysis_stage": "input_validated",
            "pending_operations": ["classify_query"],
        }

    except Exception as e:
        return {
            **state,
            "error": f"Input validation failed: {str(e)}",
            "analysis_stage": "error",
        }


def analyze_repository_node(state: ChatFlowState) -> Dict[str, Any]:
    """Analyze repository contents and structure"""
    try:
        # Check if we have repository data
        if not state.get("source_path"):
            return {
                **state,
                "error": "No repository path provided",
                "analysis_stage": "error",
            }

        # Get repository data
        repo_url = state["source_path"]
        files = state.get("files", [])

        # Initialize analysis results
        analysis_results = {
            "structure": {},
            "dependencies": {},
            "metrics": {},
            "components": [],
        }

        # Analyze structure
        for file in files:
            if isinstance(file, dict):
                file = RepoFile(**file)
            structure = _analyze_file_structure(file)
            analysis_results["structure"][file.path] = structure

            # Extract components
            if structure.get("classes"):
                analysis_results["components"].extend(
                    [
                        {"type": "class", "name": cls, "file": file.path}
                        for cls in structure["classes"]
                    ]
                )
            if structure.get("functions"):
                analysis_results["components"].extend(
                    [
                        {"type": "function", "name": func, "file": file.path}
                        for func in structure["functions"]
                    ]
                )

        # Analyze dependencies
        analysis_results["dependencies"] = RepoAnalysisTools.analyze_dependencies(files)

        # Calculate metrics
        analysis_results["metrics"] = {
            "total_files": len(files),
            "languages": dict(Counter(f.language for f in files if f.language)),
            "components": len(analysis_results["components"]),
            "dependencies": len(analysis_results["dependencies"]),
        }

        # Update state with analysis results
        return {
            **state,
            "repo_context": analysis_results,
            "current_context": analysis_results,
            "analysis_stage": "repository_analyzed",
            "analysis_complete": True,
        }

    except Exception as e:
        return {
            **state,
            "error": f"Repository analysis failed: {str(e)}",
            "analysis_stage": "error",
        }


# Add this at the end of the file


def get_graph():
    """Return the compiled graph for use in LangGraph Studio"""
    return build_main_graph()


# Export the graph
graph = get_graph()


def check_environment() -> None:
    """Check if all required environment variables are set and configure SSL certificates"""
    # Set SSL certificate path using certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for language model",
        "GITHUB_TOKEN": "GitHub token for repository access",
        "TAVILY_API_KEY": "Tavily API key for web search",
        "LANGCHAIN_API_KEY": "LangChain API key for tracing",
    }

    missing_vars = {
        var: desc for var, desc in required_vars.items() if not os.getenv(var)
    }

    if missing_vars:
        raise EnvironmentError(
            "Missing required environment variables:\n"
            + "\n".join(f"- {var}: {desc}" for var, desc in missing_vars.items())
        )


def _extract_markdown_content(files: List[RepoFile]) -> Dict[str, str]:
    """Extract content from markdown files in the repository"""
    markdown_files = {}
    for file in files:
        if file.path.lower().endswith(".md"):
            markdown_files[file.path] = file.content
    return markdown_files


def _create_pull_request(
    repo_url: str, branch_name: str, file_path: str, content: str, commit_message: str
) -> Dict:
    """Create a pull request with updated documentation"""
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GitHub token not found")

        g = Github(github_token)
        parts = repo_url.split("github.com/")[-1].split("/")
        owner, repo_name = parts[0], parts[1].replace(".git", "")
        repo = g.get_repo(f"{owner}/{repo_name}")

        # Get the default branch
        default_branch = repo.default_branch
        default_branch_ref = repo.get_git_ref(f"heads/{default_branch}")

        # Create new branch
        new_branch_ref = repo.create_git_ref(
            ref=f"refs/heads/{branch_name}", sha=default_branch_ref.object.sha
        )

        try:
            # Try to get existing file
            existing_file = repo.get_contents(file_path, ref=branch_name)
            repo.update_file(
                file_path,
                commit_message,
                content,
                existing_file.sha,
                branch=branch_name,
            )
        except:
            # File doesn't exist, create new one
            repo.create_file(file_path, commit_message, content, branch=branch_name)

        # Create pull request
        pr = repo.create_pull(
            title=commit_message,
            body="Automated documentation update by GitHub Assistant",
            head=branch_name,
            base=default_branch,
        )

        return {"success": True, "pr_url": pr.html_url, "pr_number": pr.number}

    except Exception as e:
        return {"success": False, "error": str(e)}
