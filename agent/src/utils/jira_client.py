from jira import JIRA
import os
from typing import Dict, Any


class JiraClient:
    def __init__(self):
        self.jira = JIRA(
            server=os.getenv("JIRA_SERVER"),
            basic_auth=(os.getenv("JIRA_EMAIL"), os.getenv("JIRA_API_TOKEN")),
        )

    def create_issue(
        self,
        project_key: str,
        summary: str,
        description: str,
        issue_type: str = "Story",
        story_points: int = None,
        priority: str = None,
    ) -> Any:
        """Create a Jira issue"""
        fields = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }

        if story_points:
            fields["customfield_10016"] = story_points  # Adjust field ID as needed

        if priority:
            fields["priority"] = {"name": priority}

        return self.jira.create_issue(fields=fields)
