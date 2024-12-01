# RepoFellow Documentation

## Project Overview

RepoFellow is an application designed to serve as an AI Agent that assists users in documenting, answering questions, and generating diagrams from a given repository URL. Leveraging various functionalities, the AI Assistant provides intelligent support for tasks related to Github repositories.

## Key Features

- **Documentation Assistance**: The AI Agent helps in generating documentation for repositories.
- **Question Answering**: Provides answers to queries related to the repository.
- **Diagram Generation**: Generates diagrams based on the repository's structure.
- **Dependency Analysis**: Analyzes dependencies within the repository.
- **Chat Interface**: Supports interaction through a chat interface for user queries.

## Installation & Usage

There are two ways to use RepoFellow:

### Method 1: LangGraph Studio Desktop (Recommended for Development)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Open LangGraph Studio Desktop application on your machine

4. In LangGraph Studio:
   - Navigate to File > Open Project
   - Select the RepoFellow directory
   - The graph will be automatically loaded and ready to use

### Method 2: Docker Deployment

1. Make sure you have Docker and Docker Compose installed.

2. Clone and setup:

   ```bash
   git clone https://github.com/david-wagih/RepoFellow.git
   cd RepoFellow
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Build the LangGraph image:

   ```bash
   langgraph build -t repo_fellow_image:latest .
   ```

4. Build and run the services:

   ```bash
   docker-compose up --build
   ```

   The `--build` flag ensures Docker Compose builds the image locally before starting the services.

5. Access RepoFellow through any of these interfaces:
   - **API Service**: `http://localhost:8123`
   - **API Documentation**: `http://localhost:8123/docs`
   - **LangSmith Studio Web Interface**: [LangSmith Studio](https://smith.langchain.com/studio/thread?baseUrl=http%3A%2F%2F127.0.0.1%3A8123)

To stop the Docker containers:

```bash
docker-compose down
```

## Example Interactions

Once connected, you can interact with RepoFellow using queries like:

- "Analyze this repository: https://github.com/user/repo"
- "Generate a diagram of the project structure"
- "What are the main dependencies?"
- "Explain the core classes and their relationships"
- "Generate documentation for the API endpoints"

## Architecture Overview

The project consists of various components:

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
