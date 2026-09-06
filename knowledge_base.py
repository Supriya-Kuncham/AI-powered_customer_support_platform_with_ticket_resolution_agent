"""
knowledge_base.py
-------------------
Milestone 2: Enterprise knowledge base + retriever.

In a real deployment this content would come from Confluence, SharePoint,
ServiceNow, internal PDFs/Word docs, etc. (deck, slide 10). For this build
it's a curated set of troubleshooting articles covering the same
departments the Milestone 1 classifier predicts, so every category of
ticket has something real to retrieve.

Retrieval uses TF-IDF + cosine similarity, exactly as specified in the
deck (slide 16-18) - "TF-IDF is excellent for your initial milestone
because it is easy to understand and deploy." Semantic embeddings are
flagged in the deck as the production upgrade path (slide 43-46), not
required for this milestone.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE = [
    {
        "id": "KB001",
        "title": "VPN Troubleshooting Guide",
        "category": "Network Connectivity",
        "content": """Step-by-step instructions for resolving VPN connection issues.
1. Verify that the user has an active internet connection.
2. Check the VPN client configuration and confirm the correct server address.
3. Verify the VPN server address is reachable from the user's network.
4. Verify authentication settings and confirm credentials have not expired.
5. Restart the VPN client service and reconnect.
6. If the issue persists, try connecting from a different network to isolate whether the problem is local or server-side.""",
    },
    {
        "id": "KB002",
        "title": "Network Firewall Configuration",
        "category": "IT Policies",
        "content": """Corporate firewalls must allow VPN traffic.
1. Verify that VPN traffic is permitted on the required ports, including ports 500 and 4500 for IPsec-based VPN connections.
2. Confirm outbound rules are not blocking the VPN gateway's IP range.
3. Check whether a recent firewall policy change coincides with the reported outage.""",
    },
    {
        "id": "KB003",
        "title": "VPN Authentication Problems",
        "category": "Authentication",
        "content": """If VPN authentication fails, verify the following.
1. Confirm the username and password have not expired or been locked after failed attempts.
2. Verify multi-factor authentication is correctly configured on the user's device.
3. Check whether the identity provider (SSO) is reporting an outage.
4. Reset the account password if lockout is confirmed.""",
    },
    {
        "id": "KB004",
        "title": "WiFi and Local Network Connectivity",
        "category": "Network Connectivity",
        "content": """Steps for resolving WiFi or local network issues.
1. Confirm the device is connected to the correct WiFi network name (SSID).
2. Restart the router or access point if multiple users report the same issue.
3. Forget and reconnect to the WiFi network to clear a stale connection profile.
4. Update network adapter drivers if the issue is isolated to one device.
5. Escalate to network infrastructure team if the WiFi outage affects an entire floor or building.""",
    },
    {
        "id": "KB005",
        "title": "Password Reset Guide",
        "category": "Authentication",
        "content": """How to help a user reset a forgotten or expired password.
1. Direct the user to the self-service password reset portal.
2. Verify their identity using the registered recovery email or phone number.
3. If self-service fails, manually trigger a reset from the admin console.
4. Confirm the account is not locked due to repeated failed attempts before resetting.""",
    },
    {
        "id": "KB006",
        "title": "Software Installation Failures",
        "category": "Technical Support",
        "content": """Resolving failed or blocked software installations.
1. Confirm the user has administrator rights on the device, or request an elevated install.
2. Re-download the installer from the official internal software repository to rule out a corrupted download.
3. Check antivirus or endpoint protection logs for a blocked installation event.
4. Clear any partial/failed install remnants before retrying.
5. Verify the software version is compatible with the user's operating system.""",
    },
    {
        "id": "KB007",
        "title": "Application Crash and Error Troubleshooting",
        "category": "Technical Support",
        "content": """General steps for application crashes or unexpected errors.
1. Collect the exact error message and the time it occurred.
2. Check application logs for a stack trace around the reported timestamp.
3. Confirm the application is on the latest supported version.
4. Clear the application cache/local data and restart.
5. Escalate to engineering with logs attached if the crash is reproducible.""",
    },
    {
        "id": "KB008",
        "title": "Hardware Troubleshooting - Printers",
        "category": "Hardware",
        "content": """Resolving common printer issues.
1. Confirm the printer is powered on and connected to the network.
2. Verify the correct printer driver is installed on the requesting device.
3. Clear the print queue and resubmit the print job.
4. Check for paper jams or low toner/ink warnings on the device itself.
5. Restart the print spooler service if jobs are stuck queued.""",
    },
    {
        "id": "KB009",
        "title": "Hardware Troubleshooting - Laptops and Desktops",
        "category": "Hardware",
        "content": """Resolving common laptop/desktop hardware issues.
1. Confirm the device powers on and check for any hardware error beep codes.
2. Test with a different power adapter/cable to rule out a charging fault.
3. Boot into safe mode to determine whether the issue is hardware or software related.
4. Check Device Manager for driver conflicts or disabled hardware components.
5. Escalate to the hardware repair team if a physical fault is confirmed.""",
    },
    {
        "id": "KB010",
        "title": "Billing and Payment Issue Resolution",
        "category": "Billing and Payments",
        "content": """Steps for resolving billing and payment disputes.
1. Verify the payment method on file is valid and not expired.
2. Check the transaction log for duplicate or failed charge attempts.
3. Confirm the invoice line items against the customer's order history.
4. If a duplicate charge is confirmed, initiate a refund through the payment processor.
5. Provide the customer an updated invoice summary once resolved.""",
    },
    {
        "id": "KB011",
        "title": "Refund Processing Guide",
        "category": "Billing and Payments",
        "content": """How to process a customer refund request.
1. Confirm the original transaction ID and payment method.
2. Verify the request falls within the refund policy window.
3. Submit the refund through the payment processor's dashboard.
4. Inform the customer that refunds typically take 5-7 business days to appear.""",
    },
    {
        "id": "KB012",
        "title": "Returns and Exchanges Policy Guide",
        "category": "Returns and Exchanges",
        "content": """Handling return and exchange requests.
1. Confirm the item and order date fall within the return policy window.
2. Verify the SKU received matches the SKU ordered; flag mismatches to the warehouse team.
3. Issue a prepaid return label if the item is eligible.
4. Process the exchange or refund once the returned item is received and inspected.""",
    },
    {
        "id": "KB013",
        "title": "Damaged Item Handling",
        "category": "Returns and Exchanges",
        "content": """Steps for handling a damaged item complaint.
1. Request photos of the damage and the shipping packaging from the customer.
2. Check the carrier's delivery scan for signs of mishandling.
3. Offer an immediate replacement or refund per policy, without requiring the item to be returned for low-value items.
4. File a claim with the shipping carrier if damage is confirmed to be in transit.""",
    },
    {
        "id": "KB014",
        "title": "Service Outage Response Playbook",
        "category": "Service Outages and Maintenance",
        "content": """Steps when the server is down or the entire company reports an outage.
1. Check the internal status/monitoring dashboard for an active incident affecting the server or service.
2. Confirm whether the server outage matches a scheduled maintenance window.
3. Check upstream third-party provider status pages for a dependency outage.
4. Post a status update to affected users with an estimated resolution time.
5. Escalate to the infrastructure on-call engineer immediately if no active incident is already logged and the entire company is affected.""",
    },
    {
        "id": "KB015",
        "title": "Sales and Pricing Inquiry Guide",
        "category": "Sales and Pre-Sales",
        "content": """Handling sales and pricing questions.
1. Confirm which plan or product the customer is asking about.
2. Send the current plan comparison sheet if the question is about pricing tiers.
3. Verify live stock/availability before quoting delivery timelines.
4. Escalate to the sales team for custom enterprise quotes.""",
    },
    {
        "id": "KB016",
        "title": "General Customer Service Escalation Guide",
        "category": "Customer Service",
        "content": """When to escalate a customer service ticket.
1. Review the ticket history for prior related contacts on the same issue.
2. If the customer's concern was not resolved in a previous interaction, escalate rather than repeat the same response.
3. Route billing-specific concerns to the billing team and technical concerns to technical support.
4. Document the resolution clearly so future agents have context.""",
    },
    {
        "id": "KB017",
        "title": "HR Payroll and Leave Query Guide",
        "category": "Human Resources",
        "content": """Resolving common HR payroll and leave balance queries.
1. Cross-check the employee's leave balance against the HRMS system directly, as portal caches can be stale.
2. For payroll discrepancies, verify the most recent pay cycle sync completed successfully.
3. Escalate to the payroll team with the specific pay period in question if a discrepancy is confirmed.""",
    },
]


class KnowledgeRetriever:
    """TF-IDF + cosine similarity retriever over the knowledge base."""

    def __init__(self, documents=None):
        self.documents = documents if documents is not None else KNOWLEDGE_BASE
        self.texts = [doc["title"] + " " + doc["content"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.document_vectors = self.vectorizer.fit_transform(self.texts)

    def search(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_vectors)[0]
        ranked_indices = scores.argsort()[::-1]

        results = []
        for index in ranked_indices[:top_k]:
            doc = self.documents[index]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "content": doc["content"],
                "score": round(float(scores[index]), 4),
            })
        return results


# Module-level singleton so the TF-IDF index is built once, not per-request
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever
