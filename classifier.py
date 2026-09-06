"""
classifier.py
--------------
Core AI logic for SupportPilot - Milestone 1.

Responsibilities:
1. Text pre-processing (cleaning raw ticket text before ML)
2. AI-based ticket classification (category) - TF-IDF + Logistic Regression
3. Severity prediction - ML model (trained on real historic priority labels)
   combined with a rule-based "critical keyword" override
4. Priority calculation - severity + business impact -> P1/P2/P3/P4
"""

import re
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

CATEGORY_MODEL_PATH = os.path.join(MODEL_DIR, "category_model.pkl")
CATEGORY_VECTORIZER_PATH = os.path.join(MODEL_DIR, "category_vectorizer.pkl")
SEVERITY_MODEL_PATH = os.path.join(MODEL_DIR, "severity_model.pkl")
SEVERITY_VECTORIZER_PATH = os.path.join(MODEL_DIR, "severity_vectorizer.pkl")

# Departments that are treated as "high business impact" when they appear as
# the predicted category -> used for priority calculation.
HIGH_IMPACT_DEPARTMENTS = {
    "Service Outages and Maintenance",
    "IT Support",
    "Technical Support",
}

# ---------------------------------------------------------------------------
# 1. TEXT PRE-PROCESSING  (Work 3 in the Milestone 1 deck)
# ---------------------------------------------------------------------------
BOILERPLATE_PATTERN = re.compile(
    r"dear (customer )?(support )?team,|dear support team,|dear customer support,|"
    r"i hope this message (reaches you|finds you).*?\.|thank you.*|best regards.*|"
    r"i am (writing|reaching out|submitting|reporting)",
    re.IGNORECASE,
)


def preprocess_text(text: str) -> str:
    """
    Cleans raw ticket text before feeding it to the ML pipeline.
    Steps:
      1. lowercase
      2. strip URLs
      3. strip generic email boilerplate ("Dear Support Team", "Thank you...",
         "Best regards...") which appears in almost every ticket regardless of
         category and otherwise dilutes the TF-IDF signal
      4. remove symbols/punctuation
      5. collapse whitespace
    (Stop-word removal + tokenization is handled internally by TfidfVectorizer.)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # strip urls
    text = BOILERPLATE_PATTERN.sub(" ", text)                # strip email boilerplate
    text = re.sub(r"[^a-z0-9\s]", " ", text)                # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    return text


# ---------------------------------------------------------------------------
# Lazy-loaded model cache so app.py / train_model.py can both import this file
# ---------------------------------------------------------------------------
_category_model = None
_category_vectorizer = None
_severity_model = None
_severity_vectorizer = None


def _load_models():
    global _category_model, _category_vectorizer, _severity_model, _severity_vectorizer
    if _category_model is None:
        _category_model = joblib.load(CATEGORY_MODEL_PATH)
        _category_vectorizer = joblib.load(CATEGORY_VECTORIZER_PATH)
        _severity_model = joblib.load(SEVERITY_MODEL_PATH)
        _severity_vectorizer = joblib.load(SEVERITY_VECTORIZER_PATH)


# ---------------------------------------------------------------------------
# 2. AI-BASED TICKET CLASSIFICATION  (Work 4)
#    ML prediction, with a rule-based safety net for short/unambiguous
#    tickets where the ML model has very little text to work with (e.g.
#    "wifi not connecting" is only 3 words - too little signal for TF-IDF,
#    which was trained on full-paragraph tickets).
# ---------------------------------------------------------------------------
CATEGORY_KEYWORD_OVERRIDES = [
    (["wifi", "wi-fi", "wireless network", "ethernet", "router", "ip address",
      "network down", "network is down", "no internet", "internet is down",
      "vpn"], "IT Support"),
    (["refund", "return item", "return my order", "exchange item",
      "wrong item shipped", "damaged item received", "wrong item received"], "Returns and Exchanges"),
    (["invoice", "billing charge", "payment failed", "card was charged",
      "charged twice", "subscription charge", "double charged"], "Billing and Payments"),
    (["password reset", "forgot my password", "account locked",
      "cannot log in", "can't log in", "login error"], "IT Support"),
]


def rule_based_category_hint(text: str):
    """Keyword override for short/unambiguous tickets. Returns a department or None."""
    t = text.lower()
    for keywords, department in CATEGORY_KEYWORD_OVERRIDES:
        for kw in keywords:
            if kw in t:
                return department
    return None


def predict_category(raw_text: str):
    """Returns (category, confidence 0-1)."""
    _load_models()
    cleaned = preprocess_text(raw_text)
    vector = _category_vectorizer.transform([cleaned])
    ml_category = _category_model.predict(vector)[0]
    proba = _category_model.predict_proba(vector)[0]
    ml_confidence = round(float(max(proba)), 4)

    # Short tickets (roughly < 6 words) give TF-IDF very little to work with,
    # since the model was trained on full-paragraph tickets. For those, trust
    # an unambiguous keyword match over a low-confidence ML guess.
    word_count = len(cleaned.split())
    hint = rule_based_category_hint(raw_text)
    if hint and (word_count < 6 or ml_confidence < 0.5):
        return hint, max(ml_confidence, 0.9)

    return ml_category, ml_confidence


# ---------------------------------------------------------------------------
# 3. SEVERITY PREDICTION  (Work 5)
#    ML model trained on historic tickets, plus a rule-based safety net for
#    unmistakably critical language (matches the Milestone 1 deck's
#    keyword-based severity engine).
# ---------------------------------------------------------------------------
CRITICAL_WORDS = [
    "server down", "entire company", "production down",
    "security breach", "data breach", "system down", "complete outage",
]

HIGH_WORDS = [
    "urgent", "cannot work", "business stopped", "client meeting",
    "vpn not working", "asap", "immediately", "critical",
]


def rule_based_severity_hint(text: str):
    """Keyword based override - returns a severity string or None."""
    t = text.lower()
    for word in CRITICAL_WORDS:
        if word in t:
            return "Critical"
    for word in HIGH_WORDS:
        if word in t:
            return "High"
    return None


def predict_severity(raw_text: str):
    """
    Returns (severity, confidence 0-1).
    Severity in {Low, Medium, High, Critical}.
    """
    _load_models()
    cleaned = preprocess_text(raw_text)

    # Rule-based override for unmistakably critical/urgent language
    hint = rule_based_severity_hint(raw_text)

    vector = _severity_vectorizer.transform([cleaned])
    ml_prediction = _severity_model.predict(vector)[0]        # low / medium / high
    proba = _severity_model.predict_proba(vector)[0]
    confidence = round(float(max(proba)), 4)

    severity_map = {"low": "Low", "medium": "Medium", "high": "High"}
    ml_severity = severity_map.get(ml_prediction, "Medium")
    severity_rank = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

    if hint == "Critical":
        return "Critical", max(confidence, 0.95)
    if hint == "High" and severity_rank[ml_severity] < severity_rank["High"]:
        # unmistakably urgent language -> raise severity to at least "High"
        return "High", max(confidence, 0.7)

    return ml_severity, confidence


# ---------------------------------------------------------------------------
# 4. PRIORITY CALCULATION  (Work 6)
# ---------------------------------------------------------------------------
def calculate_priority(severity: str, business_impact: str) -> str:
    if severity == "Critical" and business_impact == "High":
        return "P1"
    elif severity == "High" and business_impact == "High":
        return "P1"
    elif severity == "High":
        return "P2"
    elif severity == "Medium":
        return "P3"
    else:
        return "P4"


def infer_business_impact(category: str) -> str:
    """Simple mapping: some departments are inherently higher business impact."""
    return "High" if category in HIGH_IMPACT_DEPARTMENTS else "Medium"


# ---------------------------------------------------------------------------
# 4b. POSSIBLE ROOT CAUSES  (diagnostic layer, feeds into future
#     auto-resolution / suggested-fix milestones)
#
#     Two layers:
#       - keyword_causes: specific, high-confidence causes triggered by exact
#         phrases in the ticket text (checked first, most relevant)
#       - department_causes: general causes typical for that department,
#         used to fill out the list / as a fallback when no keyword hits
# ---------------------------------------------------------------------------
KEYWORD_CAUSES = [
    # (keyword/phrase, cause, suggested_next_step)
    ("vpn", "VPN client may be outdated or the connection profile misconfigured",
     "Reinstall/update the VPN client and re-import the connection profile"),
    ("password", "Account credentials may have expired or been locked after failed attempts",
     "Trigger a self-service password reset and check account lock status"),
    ("login", "Session/auth token may have expired or SSO provider is unreachable",
     "Clear cached credentials and verify SSO/identity provider status"),
    ("slow", "Possible network congestion or server-side performance bottleneck",
     "Check server load and network latency to the affected service"),
    ("install", "Installation may be blocked by missing permissions or a corrupted installer",
     "Verify admin rights and re-download the installer from the official source"),
    ("payment", "Payment gateway timeout or card authorization failure",
     "Check payment gateway status and confirm card/billing details"),
    ("charge", "Duplicate or incorrect billing charge",
     "Cross-check the transaction log against the customer's order history"),
    ("refund", "Refund may be delayed in the payment processor's queue",
     "Check refund status with the payment processor (usually 5-7 business days)"),
    ("server down", "Backend service outage or infrastructure failure",
     "Check server/infra monitoring dashboards for an active incident"),
    ("error", "Application exception, likely from bad input or a missing dependency",
     "Check application error logs around the reported timestamp"),
    ("access", "Permission/role assignment may be missing or incorrectly configured",
     "Review the user's role and access group assignment"),
    ("email", "Mail server delivery delay or spam-filter false positive",
     "Check mail server queue and spam filter logs for the affected address"),
    ("delivery", "Shipping carrier delay or incorrect address on file",
     "Track the shipment with the carrier and verify the delivery address"),
    ("damaged", "Item likely damaged in transit due to packaging or handling",
     "Request photos of the damage and initiate a replacement/return"),
]

DEPARTMENT_CAUSES = {
    "Technical Support": [
        ("Software bug or unhandled edge case in the application", "Reproduce the issue and check recent release notes/changelog"),
        ("Incompatible OS or browser version", "Confirm the user's OS/browser version against supported list"),
        ("Corrupted local cache or config file", "Clear cache/local storage and restart the application"),
    ],
    "IT Support": [
        ("Network connectivity or firewall rule blocking access", "Check firewall/proxy rules for the affected port or domain"),
        ("Outdated device drivers or OS patches", "Verify device is on the latest supported patch level"),
        ("Expired or misconfigured VPN/SSO credentials", "Validate credentials against the identity provider"),
    ],
    "Product Support": [
        ("Feature is being used in an unintended way / missing a config step", "Walk through the setup guide for the specific feature"),
        ("Version mismatch between client and server", "Confirm the product version and check for pending updates"),
        ("Known product limitation not yet documented", "Check the internal known-issues tracker"),
    ],
    "Billing and Payments": [
        ("Expired or declined payment method", "Ask the customer to verify/update their payment method"),
        ("Currency conversion or tax calculation mismatch", "Review the invoice breakdown line by line"),
        ("Duplicate transaction from a retried request", "Check for duplicate transaction IDs in the payment log"),
    ],
    "Returns and Exchanges": [
        ("Wrong item shipped from the warehouse", "Verify the SKU shipped against the SKU ordered"),
        ("Return window may have technically expired", "Check order date against the return policy window"),
        ("Warehouse processing backlog", "Check current warehouse processing queue/SLA"),
    ],
    "Service Outages and Maintenance": [
        ("Scheduled maintenance window overlapping with usage", "Check the maintenance calendar for overlapping windows"),
        ("Upstream third-party provider outage", "Check status pages of dependent third-party services"),
        ("Server overload from unexpected traffic spike", "Check auto-scaling logs and current load metrics"),
    ],
    "Sales and Pre-Sales": [
        ("Pricing or plan comparison unclear to the customer", "Send the current plan comparison sheet"),
        ("Stock/availability information out of date", "Confirm live stock levels before responding"),
        ("Quote approval pending internally", "Check quote approval workflow status"),
    ],
    "Customer Service": [
        ("General miscommunication or unclear policy explanation", "Clarify the relevant policy in plain language"),
        ("Previous response may not have fully resolved the concern", "Review ticket history for prior related tickets"),
        ("Escalation may be needed to a specialized team", "Assess whether this needs routing to a specialist queue"),
    ],
    "Human Resources": [
        ("Leave balance or payroll data not yet synced", "Cross-check with the HRMS payroll sync logs"),
        ("Onboarding document missing or pending approval", "Check onboarding checklist completion status"),
        ("Policy question needing an HR policy lookup", "Reference the current employee handbook section"),
    ],
    "General Inquiry": [
        ("Request may need routing to a specialized department", "Re-read the ticket to identify the best-fit department"),
        ("Missing information needed to act on the request", "Ask a clarifying follow-up question"),
    ],
}


def get_possible_causes(category: str, raw_text: str, max_causes: int = 3):
    """
    Returns a short, ranked list of {cause, suggestion} dicts.
    Keyword-triggered causes (specific to this ticket's wording) are ranked
    first; department-level causes fill the rest of the list.
    """
    text = raw_text.lower()
    causes = []
    seen = set()

    for keyword, cause, suggestion in KEYWORD_CAUSES:
        if keyword in text and cause not in seen:
            causes.append({"cause": cause, "suggestion": suggestion})
            seen.add(cause)
        if len(causes) >= max_causes:
            break

    if len(causes) < max_causes:
        for cause, suggestion in DEPARTMENT_CAUSES.get(category, []):
            if cause not in seen:
                causes.append({"cause": cause, "suggestion": suggestion})
                seen.add(cause)
            if len(causes) >= max_causes:
                break

    return causes


# ---------------------------------------------------------------------------
# 5. END-TO-END PIPELINE  (Complete Milestone 1 Demo)
# ---------------------------------------------------------------------------
def process_ticket(raw_text: str):
    category, cat_confidence = predict_category(raw_text)
    severity, sev_confidence = predict_severity(raw_text)
    business_impact = infer_business_impact(category)
    priority = calculate_priority(severity, business_impact)
    possible_causes = get_possible_causes(category, raw_text)

    return {
        "category": category,
        "category_confidence": cat_confidence,
        "severity": severity,
        "severity_confidence": sev_confidence,
        "business_impact": business_impact,
        "priority": priority,
        "possible_causes": possible_causes,
    }


if __name__ == "__main__":
    sample = "URGENT: VPN is not connecting. I cannot work and I have an important client meeting."
    print(process_ticket(sample))
