"""
jira_service.py
------------------
Milestone 3: Jira integration.

Creates a real Jira Cloud issue via Jira's REST API when a ticket is
escalated. Requires JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY
to be set (see .env.example). Without them, this gracefully reports
"not configured" instead of crashing - same pattern as the Google/Facebook
OAuth integration.
"""

import os
import requests


class JiraService:
    def __init__(self):
        self.url = os.environ.get("JIRA_URL", "").rstrip("/")
        self.email = os.environ.get("JIRA_EMAIL", "")
        self.api_token = os.environ.get("JIRA_API_TOKEN", "")
        self.project_key = os.environ.get("JIRA_PROJECT_KEY", "")

    @property
    def is_configured(self) -> bool:
        return all([self.url, self.email, self.api_token, self.project_key])

    def create_ticket(self, summary: str, description: str, category: str = "", priority: str = "") -> dict:
        if not self.is_configured:
            return {
                "created": False,
                "configured": False,
                "message": "Jira is not connected. Set JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, "
                            "and JIRA_PROJECT_KEY in .env to enable real ticket creation.",
            }

        endpoint = f"{self.url}/rest/api/3/issue"
        payload = {
            "fields": {
                "project": {"key": self.project_key},
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description}],
                        }
                    ],
                },
                "issuetype": {"name": "Task"},
                "labels": [category.replace(" ", "-"), priority] if category else [priority],
            }
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                auth=(self.email, self.api_token),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
        except requests.RequestException as exc:
            return {"created": False, "configured": True, "message": f"Jira request failed: {exc}"}

        if response.status_code in (200, 201):
            data = response.json()
            ticket_key = data.get("key")
            return {
                "created": True,
                "configured": True,
                "ticket_key": ticket_key,
                "ticket_url": f"{self.url}/browse/{ticket_key}" if ticket_key else None,
            }

        return {
            "created": False,
            "configured": True,
            "message": f"Jira returned {response.status_code}: {response.text[:200]}",
        }
