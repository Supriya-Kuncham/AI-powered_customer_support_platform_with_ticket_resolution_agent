"""
agents.py
----------
Milestone 3: Multi-Agent Workflows.

Five specialized agents, coordinated by SupportPilotOrchestrator:
  1. DiagnosisAgent   - classifies the ticket (wraps classifier.py, Milestone 1)
  2. RetrievalAgent   - searches the knowledge base (wraps knowledge_base.py, Milestone 2)
  3. ResolutionAgent  - generates a cited resolution (wraps rag_pipeline.py, Milestone 2)
  4. ValidationAgent  - scores confidence and decides AUTO_RESOLVE vs ESCALATE
  5. EscalationAgent  - creates a Jira ticket when validation says ESCALATE

Each agent exposes one clear method and returns a plain dict, so the
orchestrator can log every step as a workflow event (used for the
"Current Workflow Activity" feed on the AI Agent page) and pass each
agent's output into the next agent, exactly as the deck's architecture
diagram (slide 78) shows.
"""

from datetime import datetime, timezone

from classifier import process_ticket
from knowledge_base import get_retriever
from rag_pipeline import generate_resolution, MIN_RELEVANCE
from jira_service import JiraService
from email_service import EmailService

AUTO_RESOLVE_THRESHOLD = 70  # deck slide 19: confidence >= 70 -> AUTO_RESOLVE


class DiagnosisAgent:
    """Classifies the ticket: category, severity, priority, likely causes."""

    def diagnose(self, ticket_text: str) -> dict:
        result = process_ticket(ticket_text)
        return {
            "category": result["category"],
            "severity": result["severity"],
            "priority": result["priority"],
            "business_impact": result["business_impact"],
            "causes": result["possible_causes"],
            "confidence": round(result["category_confidence"] * 100, 2),  # as a percentage, matching the deck
        }


class RetrievalAgent:
    """Searches the knowledge base for the most relevant articles."""

    # TF-IDF cosine similarity between a short ticket and a longer KB article
    # rarely exceeds ~0.5, even for a near-perfect keyword match - this is a
    # well-known property of TF-IDF (sparse vectors, document-length dilution),
    # confirmed empirically on this project's own knowledge base. The deck's
    # illustrative example (slide 18) uses "82% retrieval similarity" as a
    # round demo number, not a value actually produced by a real TF-IDF
    # pipeline. Feeding the raw 0-~0.5 score straight into the confidence
    # formula below would mean almost every real ticket scores too low to
    # ever reach AUTO_RESOLVE - so it's rescaled here against the realistic
    # ceiling this retriever actually produces, the same way you'd calibrate
    # any model's raw score against its own real output range rather than
    # against a made-up illustrative range.
    REALISTIC_CEILING = 0.50

    def __init__(self):
        self.retriever = get_retriever()

    def retrieve(self, ticket_text: str, top_k: int = 3) -> dict:
        raw_results = self.retriever.search(ticket_text, top_k=top_k)
        filtered = [r for r in raw_results if r["score"] >= MIN_RELEVANCE]

        if filtered:
            raw_top = filtered[0]["score"]
            normalized = (raw_top - MIN_RELEVANCE) / (self.REALISTIC_CEILING - MIN_RELEVANCE)
            top_similarity = round(max(0.0, min(1.0, normalized)) * 100, 2)
        else:
            top_similarity = 0.0

        return {
            "documents": filtered,
            "count": len(filtered),
            "top_similarity": top_similarity,  # calibrated 0-100, see REALISTIC_CEILING note above
            "raw_top_score": filtered[0]["score"] if filtered else 0.0,
        }


class ResolutionAgent:
    """Generates a step-by-step, cited resolution from retrieved KB articles."""

    def generate(self, retrieved_docs: list) -> dict:
        resolution_text, steps = generate_resolution(retrieved_docs)
        return {
            "resolution": resolution_text,
            "steps": steps,
            "step_count": len(steps),
        }


class ValidationAgent:
    """
    Scores overall confidence in the generated resolution and decides
    whether it's safe to auto-resolve or whether a human should take over.
    Formula per deck slide 18:
        confidence = diagnosis_confidence*0.40 + retrieval_similarity*0.40
                     + min(steps/6, 1)*0.20
    """

    def validate(self, diagnosis_confidence: float, retrieval_similarity: float, number_of_steps: int) -> dict:
        confidence = (
            diagnosis_confidence * 0.40
            + retrieval_similarity * 0.40
            + min(number_of_steps / 6, 1) * 0.20
        )
        confidence = round(confidence, 2)
        status = "AUTO_RESOLVE" if confidence >= AUTO_RESOLVE_THRESHOLD else "ESCALATE"
        return {"confidence": confidence, "status": status}


class EscalationAgent:
    """Creates a Jira ticket when the Validation Agent says ESCALATE."""

    def __init__(self):
        self.jira = JiraService()

    def should_escalate(self, validation: dict) -> bool:
        return validation["status"] == "ESCALATE"

    def escalate(self, ticket_title, ticket_description, category, priority) -> dict:
        return self.jira.create_ticket(
            summary=ticket_title or ticket_description[:80],
            description=ticket_description,
            category=category,
            priority=priority,
        )


class SupportPilotOrchestrator:
    """
    Coordinates all five agents for one ticket, exactly matching the deck's
    workflow diagram: Diagnosis -> Retrieval -> Resolution -> Validation ->
    (AUTO_RESOLVE -> email) or (ESCALATE -> Jira + email).
    Every step is logged as a workflow event for the UI's activity feed.
    """

    def __init__(self):
        self.diagnosis_agent = DiagnosisAgent()
        self.retrieval_agent = RetrievalAgent()
        self.resolution_agent = ResolutionAgent()
        self.validation_agent = ValidationAgent()
        self.escalation_agent = EscalationAgent()
        self.email_service = EmailService()

    def _log(self, workflow_log, agent, message):
        workflow_log.append({
            "time": datetime.now(timezone.utc).strftime("%I:%M %p"),
            "agent": agent,
            "message": message,
        })

    def process_ticket(self, ticket_text: str, employee_name: str = "", requester_email: str = "",
                        title: str = "") -> dict:
        workflow_log = []

        # --- Agent 1: Diagnosis ---
        diagnosis = self.diagnosis_agent.diagnose(ticket_text)
        self._log(workflow_log, "Diagnosis Agent",
                   f"Identified {diagnosis['category'].lower()} issue "
                   f"({diagnosis['severity']} severity)")

        # --- Agent 2: Retrieval ---
        retrieval = self.retrieval_agent.retrieve(ticket_text)
        self._log(workflow_log, "Retrieval Agent",
                   f"Found {retrieval['count']} relevant knowledge article{'s' if retrieval['count'] != 1 else ''}")

        # --- Agent 3: Resolution ---
        resolution = self.resolution_agent.generate(retrieval["documents"])
        self._log(workflow_log, "Resolution Agent",
                   f"Generated {resolution['step_count']}-step troubleshooting resolution")

        # --- Agent 4: Validation ---
        validation = self.validation_agent.validate(
            diagnosis["confidence"], retrieval["top_similarity"], resolution["step_count"]
        )
        self._log(workflow_log, "System", f"Resolution confidence: {validation['confidence']}%")

        result = {
            "diagnosis": diagnosis,
            "retrieval": retrieval,
            "resolution": resolution,
            "validation": validation,
            "workflow_log": workflow_log,
            "jira": None,
            "email": None,
        }

        # --- Agent 5: Escalation (only if validation says so) ---
        if self.escalation_agent.should_escalate(validation):
            jira_result = self.escalation_agent.escalate(
                title, ticket_text, diagnosis["category"], diagnosis["priority"]
            )
            result["jira"] = jira_result
            if jira_result.get("created"):
                self._log(workflow_log, "Escalation Agent",
                           f"Created Jira ticket {jira_result.get('ticket_key')} for human review")
            else:
                self._log(workflow_log, "Escalation Agent",
                           "Low confidence - flagged for human review (Jira not connected)")

            if requester_email:
                email_result = self.email_service.send_email(
                    to_email=requester_email,
                    subject=f"Support Ticket Received: {title or 'Your request'}",
                    body=(
                        f"Hello {employee_name or ''},\n\n"
                        f"We've received your support ticket and it has been assigned to our "
                        f"support team for review. We'll follow up shortly.\n\n"
                        f"— SupportPilot"
                    ),
                )
                result["email"] = email_result
                if email_result.get("sent"):
                    self._log(workflow_log, "Email Service", f"Escalation notice sent to {requester_email}")
        else:
            if requester_email:
                email_result = self.email_service.send_email(
                    to_email=requester_email,
                    subject=f"Support Ticket Resolved: {title or 'Your request'}",
                    body=(
                        f"Hello {employee_name or ''},\n\n"
                        f"Your support ticket has been automatically resolved by our AI system.\n\n"
                        f"{resolution['resolution']}\n\n"
                        f"— SupportPilot"
                    ),
                )
                result["email"] = email_result
                if email_result.get("sent"):
                    self._log(workflow_log, "Email Service", f"Resolution email sent to {requester_email}")

        return result


# Module-level singleton so agents (and the KB's TF-IDF index) are built once
_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SupportPilotOrchestrator()
    return _orchestrator
