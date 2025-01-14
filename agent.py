import requests
from typing import Dict, Optional, Tuple
import os
import logging

class CodeAssistant:
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        # Define models for different tasks
        self.models = {
            'code': "qwen2.5-coder",  # For code generation and modification
            'docs': "gemma2:2b",        # For documentation and explanation
            'general': "gemma2:2b"      # For general questions
        }
    
    def _generate_response(self, prompt: str, task_type: str = 'general') -> str:
        """Generate response from Ollama using the appropriate model for the task"""
        model = self.models.get(task_type, self.models['general'])
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            if "response" not in response_data:
                raise ValueError("Unexpected response format from Ollama")
            
            return response_data["response"]
            
        except requests.exceptions.ConnectionError:
            return f"Error: Cannot connect to Ollama. Make sure Ollama is running (use 'ollama serve') and the model {model} is installed"
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                return f"Error: Model '{model}' not found. Try running 'ollama pull {model}' first"
            return f"Error: HTTP error occurred: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def answer_question(self, question: str, codebase) -> str:
        """Answer questions about the codebase"""
        context = self._prepare_context(codebase)
        
        # Determine if the question is code-related
        code_keywords = {'how to', 'code', 'implement', 'function', 'class', 'method', 'syntax', 'error'}
        is_code_related = any(keyword in question.lower() for keyword in code_keywords)
        task_type = 'code' if is_code_related else 'general'
        
        prompt = f"""Given the following codebase context:

{context}

Question: {question}

Please provide a detailed answer based on the codebase context."""
        
        return self._generate_response(prompt, task_type)
    
    def suggest_modifications(
        self, 
        instruction: str, 
        codebase, 
        specific_file: Optional[str] = None,
        line_range: Optional[Tuple[int, int]] = None
    ) -> str:
        """Suggest code modifications based on instructions"""
        if specific_file:
            content = codebase.get_file_content(specific_file)
            if not content:
                return f"Error: File '{specific_file}' not found in the codebase"
            
            if line_range:
                lines = content.splitlines()
                start, end = line_range
                if 1 <= start <= len(lines) and 1 <= end <= len(lines):
                    selected_content = '\n'.join(lines[start-1:end])
                    context = f"Selected section (lines {start}-{end}) of {specific_file}:\n```\n{selected_content}\n```\n\nFull file context:\n```\n{content}\n```"
                else:
                    return f"Error: Line range {start}-{end} is out of bounds for file '{specific_file}'"
            else:
                context = f"File: {specific_file}\n```\n{content}\n```"
        else:
            context = self._prepare_context(codebase)
        
        prompt = f"""Given the following codebase:

{context}

Modification instruction: {instruction}

Please suggest specific code modifications to implement this change. Include the file path and the modified code sections."""
        
        modifications = self._generate_response(prompt, 'code')
        quality_report = self._check_code_quality(codebase)
        
        return f"{modifications}\n\nCode Quality Report:\n{quality_report}"
    
    def _prepare_context(self, codebase) -> str:
        """Prepare codebase context for the prompt"""
        files = codebase.get_all_files()
        context = []
        
        for file_path, content in files.items():
            context.append(f"File: {file_path}\n```\n{content}\n```\n")
        
        return "\n".join(context) 
    
    def generate_readme(self, codebase) -> str:
        """Generate a comprehensive README for the codebase"""
        # Get basic context
        context = self._prepare_context(codebase)
        
        # Get project structure
        try:
            project_structure = []
            for root, dirs, files in os.walk(codebase.base_path):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', '.venv']]
                level = root.replace(str(codebase.base_path), '').count(os.sep)
                indent = '  ' * level
                folder = os.path.basename(root)
                if level > 0:
                    project_structure.append(f"{indent}- 📁 {folder}/")
                for file in sorted(files):
                    if not file.startswith('.') and file not in ['__init__.py']:
                        project_structure.append(f"{indent}  - 📄 {file}")
            project_tree = '\n'.join(project_structure)
        except Exception as e:
            project_tree = "Error getting project structure"

        # Get dependencies from requirements.txt and pyproject.toml
        dependencies = set()
        if 'requirements.txt' in codebase.files:
            for line in codebase.files['requirements.txt'].splitlines():
                if line.strip() and not line.startswith('#'):
                    dependencies.add(line.strip())
        
        if 'pyproject.toml' in codebase.files:
            import tomli
            try:
                pyproject = tomli.loads(codebase.files['pyproject.toml'])
                if 'dependencies' in pyproject.get('project', {}):
                    for dep in pyproject['project']['dependencies']:
                        dependencies.add(dep)
            except:
                pass

        deps_list = '\n'.join([f"- {dep}" for dep in sorted(dependencies)])

        # Get CLI commands
        commands = []
        try:
            import ast
            for file_path, content in codebase.files.items():
                if file_path.endswith('main.py'):
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and any(
                            decorator.id == 'command' 
                            for decorator in node.decorator_list 
                            if isinstance(decorator, ast.Name)
                        ):
                            doc = ast.get_docstring(node) or "No description available"
                            commands.append(f"- `{node.name}`: {doc}")
        except:
            commands = ["Error parsing commands"]

        commands_list = '\n'.join(commands)

        prompt = f"""Given the following codebase context:

{context}

Project Structure:

{project_tree}

Dependencies

{deps_list}

CLI Commands

{commands_list}

Generate a comprehensive README.md file that includes:
1. Project Title and Description
2. Key Features and Technical Capabilities
3. Prerequisites and Dependencies
4. Installation Guide
5. Usage Instructions with Examples
6. Available Commands and Options
7. Project Structure
8. Configuration (if any)
9. Contributing Guidelines (basic)
10. License Information

Format the response in Markdown and make it professional and well-structured.
Include proper code blocks with syntax highlighting where needed.
Base all information strictly on the actual codebase content.
Include a section on code quality checks and how to run them."""

        return self._generate_response(prompt) 
    
    def inspect_code(self, file_path: str, question: str, codebase, line_range: Optional[Tuple[int, int]] = None) -> str:
        """Analyze specific file or code section"""
        content = codebase.get_file_content(file_path)
        if not content:
            return f"Error: File '{file_path}' not found in the codebase"
        
        # Extract specific lines if line range is provided
        if line_range:
            lines = content.splitlines()
            start, end = line_range
            if 1 <= start <= len(lines) and 1 <= end <= len(lines):
                content = '\n'.join(lines[start-1:end])
                context = f"Lines {start}-{end} of {file_path}:\n```python\n{content}\n```"
            else:
                return f"Error: Line range {start}-{end} is out of bounds for file '{file_path}'"
        else:
            context = f"File: {file_path}\n```python\n{content}\n```"
        
        prompt = f"""Analyze the following code section concisely and precisely:

        {context}

        Question: {question}

        Please provide a focused response that:
        1. Directly answers the question
        2. Only references the code shown above
        3. Keeps the explanation clear and concise
        4. Uses bullet points for multiple points
        5. Includes code examples if relevant
        6. Highlights key functions, classes, or variables mentioned

        Focus on the selected code section and avoid discussing code outside the markers."""

        
        # Use code model for code-related questions
        return self._generate_response(prompt, 'code') 
    
    def _check_code_quality(self, codebase) -> str:
        """Perform code quality checks using pylint"""
        try:
            # Get Python files from the codebase
            python_files = codebase.get_python_files()
            
            logging.debug(f"Found {len(python_files)} Python files to check")
            if not python_files:
                logging.warning("No Python files found to check")
                return "No Python files found to check."
            
            logging.debug(f"Files to check: {python_files}")
            
            # Run pylint with the JSON reporter
            from io import StringIO
            from pylint.lint import Run
            from pylint.reporters import JSONReporter
            
            # Capture pylint output
            output = StringIO()
            reporter = JSONReporter(output)
            
            # Run pylint with the JSON reporter
            Run(
                python_files,
                reporter=reporter,
                exit=False  # Only use exit parameter, not do_exit
            )
            
            return output.getvalue()
            
        except Exception as e:
            logging.error(f"Error running pylint: {str(e)}", exc_info=True)
            return f"Error running pylint: {str(e)}"
