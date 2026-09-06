"""
rag_pipeline.py
-----------------
Milestone 2: Knowledge Retrieval & Resolution Generation.

Implements the pipeline from the deck (slides 4, 27-28, 47-50):
  1. Ticket analysis & query generation
  2. Knowledge base retrieval (TF-IDF + cosine similarity, see knowledge_base.py)
  3. Context augmentation
  4. Resolution generation, with a per-step citation back to its source article

No external LLM API is called. The "generator" step (deck slide 25-26)
extracts and numbers the actual troubleshooting steps from the retrieved
KB articles rather than inventing new ones - this keeps every claim in the
resolution traceable to a real source, which is the whole point of RAG
over asking a model to answer from memory.
"""

import re
from knowledge_base import get_retriever

# Below this relevance score, a KB article is considered not relevant enough
# to build a resolution from (deck, slide 49, "confidence/relevance threshold").
# The deck suggests 0.30 as an illustrative value. Tested empirically against
# this project's actual TF-IDF scores: short, real-world ticket phrasing
# ("laptop wont turn on") scores as low as ~0.10 even for a correct match,
# while genuinely irrelevant queries score at or near 0.0. 0.08 was chosen
# to keep short genuine tickets from being wrongly flagged as
# "insufficient knowledge" while still filtering out no-signal queries.
MIN_RELEVANCE = 0.08


def build_context(results):
    """Formats retrieved KB articles into a context block (deck, slide 21, 35)."""
    context = ""
    for doc in results:
        context += (
            f"\nSOURCE: {doc['id']}\n"
            f"TITLE: {doc['title']}\n"
            f"RELEVANCE: {doc['score']:.2f}\n\n"
            f"{doc['content']}\n"
            f"{'=' * 40}\n"
        )
    return context


MAX_RESOLUTION_STEPS = 8


def generate_resolution(retrieved_docs):
    """
    Builds a numbered, cited resolution from the retrieved KB articles
    (deck, slide 25-26 + slide 47-48 citation format).
    Returns (resolution_text, structured_steps) where structured_steps is a
    list of {step, text, source_id, source_title} used by the UI.
    Capped at MAX_RESOLUTION_STEPS so a multi-document match stays readable
    instead of dumping every step from every retrieved article.
    """
    if not retrieved_docs:
        return (
            "No sufficiently relevant knowledge-base articles were found for "
            "this ticket. Recommend routing to a human agent for manual review.",
            [],
        )

    structured_steps = []
    step_number = 1

    for doc in retrieved_docs:
        for line in doc["content"].split("\n"):
            if step_number > MAX_RESOLUTION_STEPS:
                break
            line = line.strip()
            match = re.match(r"^\d+\.\s*(.+)$", line)
            if match:
                structured_steps.append({
                    "step": step_number,
                    "text": match.group(1).strip(),
                    "source_id": doc["id"],
                    "source_title": doc["title"],
                })
                step_number += 1
        if step_number > MAX_RESOLUTION_STEPS:
            break

    if not structured_steps:
        return (
            "Relevant knowledge-base articles were found, but no structured "
            "steps could be extracted. Recommend manual review of the source articles.",
            [],
        )

    lines = ["Recommended Resolution:"]
    for s in structured_steps:
        lines.append(f"{s['step']}. {s['text']}")
        lines.append(f"   Source: {s['source_id']} \u2013 {s['source_title']}")

    return "\n".join(lines), structured_steps


def run_rag_pipeline(ticket_text: str, top_k: int = 3):
    """
    Full pipeline: retrieve -> filter by relevance -> build context ->
    generate cited resolution. Returns a dict with every intermediate
    artifact so the UI can show the same 4-stage workflow the deck specifies.
    """
    workflow = {
        "ticket_analysis": "completed",
        "knowledge_retrieval": "pending",
        "context_augmentation": "pending",
        "response_generation": "pending",
    }

    retriever = get_retriever()
    raw_results = retriever.search(ticket_text, top_k=top_k)
    workflow["knowledge_retrieval"] = "completed"

    filtered_results = [r for r in raw_results if r["score"] >= MIN_RELEVANCE]

    if not filtered_results:
        workflow["context_augmentation"] = "skipped"
        workflow["response_generation"] = "skipped"
        return {
            "status": "INSUFFICIENT_KNOWLEDGE",
            "message": "No sufficiently relevant knowledge-base articles were found.",
            "retrieved_documents": raw_results,
            "context": "",
            "resolution": (
                "No sufficiently relevant knowledge-base articles were found for "
                "this ticket. Recommend routing to a human agent for manual review."
            ),
            "steps": [],
            "workflow": workflow,
        }

    context = build_context(filtered_results)
    workflow["context_augmentation"] = "completed"

    resolution_text, steps = generate_resolution(filtered_results)
    workflow["response_generation"] = "completed"

    return {
        "status": "OK",
        "message": None,
        "retrieved_documents": filtered_results,
        "context": context,
        "resolution": resolution_text,
        "steps": steps,
        "workflow": workflow,
    }
