"""Business analysis tools and utilities"""
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from ..utils.config import llm
from ..utils.jira_client import JiraClient
from ..utils.image_analysis import analyze_design_image


class BusinessAnalysisTools:
    """Tools for business analysis and requirements processing"""

    @staticmethod
    def analyze_design(image_data: bytes) -> str:
        """Analyze design image and extract requirements"""
        return analyze_design_image(image_data)

    @staticmethod
    def analyze_requirements(text: str) -> List[Dict[str, Any]]:
        """Analyze textual requirements and convert to structured format"""
        messages = [
            SystemMessage(content="Convert the following requirements into user stories:"),
            HumanMessage(content=text)
        ]
        response = llm.invoke(messages).content
        return BusinessAnalysisTools.parse_user_stories(response)

    @staticmethod
    def parse_user_stories(text: str) -> List[Dict[str, Any]]:
        """Parse user stories from text into structured format"""
        stories = []
        # Basic parsing logic - can be enhanced
        for line in text.split('\n'):
            if line.strip().lower().startswith(('as a ', 'as an ')):
                parts = line.split('so that')
                if len(parts) >= 2:
                    who_what = parts[0].split('i want')
                    if len(who_what) >= 2:
                        stories.append({
                            'role': who_what[0].replace('as a', '').replace('as an', '').strip(),
                            'want': who_what[1].strip(),
                            'why': parts[1].strip()
                        })
        return stories

    @staticmethod
    def format_user_stories(stories: List[Dict[str, Any]]) -> str:
        """Format user stories for display"""
        if not stories:
            return "No user stories generated."
            
        formatted = "Generated User Stories:\n\n"
        for i, story in enumerate(stories, 1):
            formatted += f"{i}. As a {story['role']},\n"
            formatted += f"   I want {story['want']},\n"
            formatted += f"   so that {story['why']}\n\n"
        return formatted
