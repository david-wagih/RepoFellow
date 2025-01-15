from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from utils.config import llm
from utils.jira_client import JiraClient
from utils.image_analysis import analyze_design_image


class BusinessAnalysisTools:
    """Tools for business requirements analysis and user story generation"""

    @staticmethod
    def analyze_requirements(requirements_text: str) -> List[Dict[str, Any]]:
        """Analyze business requirements and generate user stories"""
        messages = [
            SystemMessage(
                content="""You are a skilled business analyst. 
                Analyze the requirements and create comprehensive user stories following this format:
                - Title: Clear and concise title
                - As a: Type of user
                - I want to: Clear action or feature
                - So that: Business value or benefit
                - Acceptance Criteria: List of specific criteria
                - Story Points: Estimated effort (1, 2, 3, 5, 8, 13)
                - Priority: Must-have, Should-have, Could-have, Won't-have
                """
            ),
            HumanMessage(
                content=f"Requirements:\n{requirements_text}\n\nGenerate user stories for these requirements."
            ),
        ]

        response = llm.invoke(messages)
        return BusinessAnalysisTools.parse_user_stories(response.content)

    @staticmethod
    def analyze_design(image_data: bytes) -> List[Dict[str, Any]]:
        """Analyze design screenshot and generate user stories"""
        design_context = analyze_design_image(image_data)
        return BusinessAnalysisTools.analyze_requirements(
            f"Design Analysis:\n{design_context}"
        )

    @staticmethod
    def parse_user_stories(content: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured user stories"""
        stories = []
        current_story = {}

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("- Title:"):
                if current_story:
                    stories.append(current_story)
                current_story = {"title": line.replace("- Title:", "").strip()}
            elif line.startswith("- As a:"):
                current_story["as_a"] = line.replace("- As a:", "").strip()
            elif line.startswith("- I want to:"):
                current_story["i_want"] = line.replace("- I want to:", "").strip()
            elif line.startswith("- So that:"):
                current_story["so_that"] = line.replace("- So that:", "").strip()
            elif line.startswith("- Acceptance Criteria:"):
                criteria_text = line.replace("- Acceptance Criteria:", "").strip()
                current_story["acceptance_criteria"] = [
                    c.strip() for c in criteria_text.split(",")
                ]
            elif line.startswith("- Story Points:"):
                points = line.replace("- Story Points:", "").strip()
                current_story["story_points"] = (
                    int(points) if points.isdigit() else None
                )
            elif line.startswith("- Priority:"):
                current_story["priority"] = line.replace("- Priority:", "").strip()

        if current_story:
            stories.append(current_story)

        return stories

    @staticmethod
    def format_user_stories(stories: List[Dict[str, Any]]) -> str:
        """Format user stories for display"""
        formatted = "Generated User Stories:\n\n"

        for i, story in enumerate(stories, 1):
            formatted += f"Story {i}:\n"
            formatted += f"Title: {story.get('title', 'N/A')}\n"
            formatted += f"As a: {story.get('as_a', 'N/A')}\n"
            formatted += f"I want to: {story.get('i_want', 'N/A')}\n"
            formatted += f"So that: {story.get('so_that', 'N/A')}\n"

            if criteria := story.get("acceptance_criteria"):
                formatted += "Acceptance Criteria:\n"
                for criterion in criteria:
                    formatted += f"- {criterion}\n"

            formatted += f"Story Points: {story.get('story_points', 'N/A')}\n"
            formatted += f"Priority: {story.get('priority', 'N/A')}\n\n"

        return formatted

    @staticmethod
    def format_jira_description(story: Dict[str, Any]) -> str:
        """Format user story for Jira description"""
        description = f"""
h2. User Story
*As a* {story.get('as_a', 'N/A')}
*I want to* {story.get('i_want', 'N/A')}
*So that* {story.get('so_that', 'N/A')}

h2. Acceptance Criteria
"""
        if criteria := story.get("acceptance_criteria"):
            for criterion in criteria:
                description += f"* {criterion}\n"

        return description

    @staticmethod
    def map_priority(priority: str) -> str:
        """Map user story priority to Jira priority"""
        priority_map = {
            "Must-have": "Highest",
            "Should-have": "High",
            "Could-have": "Medium",
            "Won't-have": "Low",
        }
        return priority_map.get(priority, "Medium")
