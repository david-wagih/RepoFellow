from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent


class BusinessAgent(BaseAgent):
    """Agent specialized in analyzing business notes and generating requirements"""

    def get_system_prompt(self) -> str:
        return """You are an expert business analyst specializing in requirements engineering.
        
        Your responsibilities:
        1. Analyze business notes and extract requirements
        2. Categorize requirements as functional or non-functional
        3. Ensure requirements are:
           - Clear and unambiguous
           - Testable and measurable
           - Feasible and realistic
           - Properly prioritized
        
        Format requirements in a structured way:
        - Functional Requirements (FR):
          * FR1: The system shall...
          * FR2: Users must be able to...
        
        - Non-Functional Requirements (NFR):
          * NFR1: Performance - The system must...
          * NFR2: Security - The system shall...
          * NFR3: Usability - The interface must..."""

    def can_handle(self, state: Dict[str, Any]) -> bool:
        return bool(state.get("raw_input"))

    def _prepare_messages(self, state: Dict[str, Any]) -> List[HumanMessage]:
        return [HumanMessage(content=f"""
        Analyze these business notes and generate clear requirements:
        
        {state.get('raw_input')}
        
        Generate both functional and non-functional requirements.
        """)]

    def _process_response(self, state: Dict[str, Any], response: str) -> Dict[str, Any]:
        return {
            **state,
            "response": response,
            "messages": state.get("messages", []) + [
                AIMessage(content="Based on your business notes, I've generated these requirements:\n\n"),
                AIMessage(content=response)
            ],
            "requirements": self._parse_requirements(response),
            "response_ready": True
        }

    def _parse_requirements(self, response: str) -> Dict[str, List[str]]:
        """Parse requirements from response into structured format"""
        functional = []
        non_functional = []
        
        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if "Functional Requirements" in line:
                current_section = functional
            elif "Non-Functional Requirements" in line:
                current_section = non_functional
            elif line.startswith(('FR', 'NFR', '*', '-')) and current_section is not None:
                requirement = line.split(':', 1)[-1].strip()
                if requirement:
                    current_section.append(requirement)
        
        return {
            "functional": functional,
            "non_functional": non_functional
        }
