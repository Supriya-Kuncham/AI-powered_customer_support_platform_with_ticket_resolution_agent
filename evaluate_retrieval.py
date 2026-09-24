"""
evaluate_retrieval.py
------------------------
Measures real retrieval accuracy for the Milestone 2 RAG pipeline.

The deck (slide 55-57) shows "Retrieval Accuracy: 92%" as an illustrative
UI metric and explicitly says "these should be treated as system metrics,
not hard-coded values" (slide 55) and shows the actual formula:
    100 tickets tested, 92 retrieved the correct KB article -> 92%

This script does that for real: a hand-labeled set of (query, expected KB
article) pairs is run through the actual retriever, and accuracy is
computed as (# where the top-1 result matches the expected article) / total.

Run:
    python evaluate_retrieval.py
"""

import json
import os
from knowledge_base import get_retriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "rag_evaluation_report.json")

# Hand-labeled test set: realistic ticket phrasing mapped to the KB article
# a correct retrieval should surface as the top result.
TEST_CASES = [
    ("Unable to connect to VPN since this morning, connection timed out", "KB001"),
    ("VPN keeps disconnecting after a few minutes on corporate network", "KB001"),
    ("Firewall seems to be blocking VPN traffic on port 500", "KB002"),
    ("VPN login keeps failing with authentication error", "KB003"),
    ("wifi not connecting on my laptop", "KB004"),
    ("entire floor lost wifi connection this morning", "KB004"),
    ("I forgot my password and need it reset", "KB005"),
    ("account locked out after too many failed login attempts", "KB005"),
    ("software installation keeps failing with a permissions error", "KB006"),
    ("cannot install the new CRM tool, setup fails every time", "KB006"),
    ("application keeps crashing with an unexpected error", "KB007"),
    ("app crashed and shows a stack trace error", "KB007"),
    ("printer is not responding to print jobs", "KB008"),
    ("print queue is stuck and nothing is printing", "KB008"),
    ("laptop wont turn on at all", "KB009"),
    ("desktop computer beeping and not booting", "KB009"),
    ("I was charged twice for the same order", "KB010"),
    ("invoice amount does not match what I was quoted", "KB010"),
    ("requesting a refund for my last order", "KB011"),
    ("how long does a refund usually take to process", "KB011"),
    ("want to exchange this item for a different size", "KB012"),
    ("received the wrong item in my order", "KB012"),
    ("the item arrived damaged and broken", "KB013"),
    ("package was crushed during shipping", "KB013"),
    ("the server is down and the whole company cannot access email", "KB014"),
    ("is there a scheduled maintenance window today", "KB014"),
    ("what is the price difference between the basic and pro plan", "KB015"),
    ("is this product currently in stock", "KB015"),
    ("I already contacted support about this and it was not resolved", "KB016"),
    ("my leave balance looks incorrect in the portal", "KB017"),
    ("payroll seems to be missing from this month", "KB017"),
]


def evaluate():
    retriever = get_retriever()
    correct = 0
    details = []

    for query, expected_id in TEST_CASES:
        results = retriever.search(query, top_k=1)
        predicted_id = results[0]["id"] if results else None
        is_correct = predicted_id == expected_id
        correct += int(is_correct)
        details.append({
            "query": query,
            "expected": expected_id,
            "predicted": predicted_id,
            "score": results[0]["score"] if results else 0,
            "correct": is_correct,
        })

    accuracy = round((correct / len(TEST_CASES)) * 100, 1)

    report = {
        "test_set_size": len(TEST_CASES),
        "correct": correct,
        "retrieval_accuracy": accuracy,
        "details": details,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Retrieval accuracy: {correct}/{len(TEST_CASES)} = {accuracy}%")
    for d in details:
        if not d["correct"]:
            print(f"  MISS: \"{d['query']}\" -> expected {d['expected']}, got {d['predicted']} (score={d['score']})")

    return report


if __name__ == "__main__":
    evaluate()
