from typing import Dict, Any
from langchain_core.messages import AIMessage
from tools.business_analysis import BusinessAnalysisTools
from utils.jira_client import JiraClient


def analyze_business_requirements(state: Dict) -> Dict[str, Any]:
    """Analyze business requirements and generate user stories"""
    try:
        input_type = state.get("input_type", "text")
        raw_input = state.get("raw_input", "")

        # Generate user stories based on input type
        if input_type == "image":
            user_stories = BusinessAnalysisTools.analyze_design(raw_input)
        else:
            user_stories = BusinessAnalysisTools.analyze_requirements(raw_input)

        # Return state with interrupt for human confirmation
        return {
            **state,
            "user_stories": user_stories,
            "messages": state.get("messages", [])
            + [
                AIMessage(
                    content=BusinessAnalysisTools.format_user_stories(user_stories)
                ),
            ],
            "__interrupt": {
                "prompt": "Would you like to sync these stories to Jira? (yes/no)",
                "type": "confirmation",
                "data": {"stories": user_stories},
            },
        }
    except Exception as e:
        return {**state, "error": f"Business analysis failed: {str(e)}"}


def handle_sync_confirmation(state: Dict) -> Dict[str, Any]:
    """Handle the human confirmation response for Jira sync"""
    try:
        human_response = state.get("__human_response", "").lower()

        if human_response not in ["yes", "y"]:
            return {
                **state,
                "messages": state.get("messages", [])
                + [AIMessage(content="Okay, I won't sync the stories to Jira.")],
                "__interrupt": None,
            }

        # Proceed with Jira sync
        jira_client = JiraClient()
        sync_results = []

        for story in state.get("user_stories", []):
            issue = jira_client.create_issue(
                project_key=state.get("jira_project", "DEFAULT"),
                summary=story["title"],
                description=BusinessAnalysisTools.format_jira_description(story),
                issue_type="Story",
                story_points=story["story_points"],
                priority=BusinessAnalysisTools.map_priority(story["priority"]),
            )
            sync_results.append(
                {
                    "story_title": story["title"],
                    "jira_key": issue.key,
                    "status": "created",
                }
            )

        return {
            **state,
            "jira_sync_status": {"success": True, "results": sync_results},
            "messages": state.get("messages", [])
            + [
                AIMessage(
                    content=f"Successfully created {len(sync_results)} user stories in Jira."
                )
            ],
            "__interrupt": None,
        }
    except Exception as e:
        return {**state, "error": f"Jira sync failed: {str(e)}"}
