# RepoFellow Documentation

## Project Overview

RepoFellow is an application designed to serve as an AI Agent that assists users in documenting, answering questions, and generating diagrams from a given repository URL. Leveraging various functionalities, the AI Assistant provides intelligent support for tasks related to Github repositories.

## Key Features

- **Documentation Assistance**: The AI Agent helps in generating documentation for repositories.
- **Question Answering**: Provides answers to queries related to the repository.
- **Diagram Generation**: Generates diagrams based on the repository's structure.
- **Dependency Analysis**: Analyzes dependencies within the repository.
- **Chat Interface**: Supports interaction through a chat interface for user queries.

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/david-wagih/RepoFellow.git
   ```
2. Install the required dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the necessary environment variables following the structure in `.env.example`.

## Usage Examples

1. Initialize the AI Assistant and provide a repository URL for analysis:
   ```python
   python repo_fellow
   ```
2. Interact with the AI Assistant through the chat interface to ask questions or request documentation.

## Architecture Overview

The project consists of various classes and functions within the `repo_fellow.py` script. Key components include:

- **Classes**: Analyst, Perspectives, RepoFile, RepoMetadata, BaseState, RepoAnalysisState, InterviewState, ChatFlowState, GenerateAnalystsState, RepositoryMemory, RepoAnalysisTools, RepoVisualizer.
- **Functions**: persona, save_analysis, extract_repo_url, analyze_dependencies, generate_architecture_diagram, handle_general_query, handle_docs_query, handle_graph_query, handle_code_query, analyze_repository, process_chat_message, build_main_graph, perform_web_search, create_dependency_graph, analyze_structure, generate_visualizations, validate_input_node, check_environment, and more.

## Contributing Guidelines

We welcome and encourage contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) guide for:

- Detailed contribution process
- Development setup instructions
- Coding standards and style guide
- Testing requirements
- Pull request guidelines

Before contributing, please also review our [Code of Conduct](CODE_OF_CONDUCT.md).

Your contributions are welcome and encouraged!
