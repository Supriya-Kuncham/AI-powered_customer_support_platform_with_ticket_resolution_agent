"""
app.py
-------
SupportPilot - Flask backend.

Milestone 1: ticket classification, severity, priority, likely causes.
Milestone 2: knowledge-base retrieval + cited resolution generation (RAG).

Routes:
  GET  /                -> Ticket submission + AI classification + RAG resolution
  POST /submit          -> Runs the full pipeline, stores the ticket, shows results
  GET  /tickets         -> Table of all submitted tickets
  GET  /dashboard        -> Stats + charts
  GET  /ai-agent         -> Pipeline explanation + real evaluation metrics
  POST /ticket/<id>/feedback -> Records whether the resolution actually worked
  GET  /register, /login, /logout -> Email-based authentication
  POST /api/ticket       -> REST API: classify + generate a cited resolution
  GET  /api/tickets      -> REST API listing of stored tickets
  GET  /api/stats        -> REST API aggregate stats
"""

from functools import wraps
import os
import time
import json

from dotenv import load_dotenv
load_dotenv()  # reads a local .env file if present, before anything reads os.environ

from flask import Flask, request, jsonify, render_template, redirect, url_for, g
from authlib.integrations.flask_client import OAuth

import database
from classifier import process_ticket
from rag_pipeline import run_rag_pipeline
from auth_jwt import generate_token, decode_token, get_token_from_request, TOKEN_COOKIE_NAME, TOKEN_EXPIRY_HOURS
from agents import get_orchestrator
from jira_service import JiraService
from email_service import EmailService

app = Flask(__name__)

# Needed for login sessions to work. For a real deployment, set this via an
# environment variable instead of leaving the hardcoded fallback in place.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-this-before-deploying")

# ---------------------------------------------------------------------------
# OAuth (Google / Facebook "Sign in with...")
#
# These only activate if you set the corresponding environment variables -
# see README.md for how to get real credentials from Google Cloud Console /
# Facebook Developers. Without them, the login page just shows the email
# form and skips the social buttons instead of crashing.
# ---------------------------------------------------------------------------
oauth = OAuth(app)

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

if GOOGLE_ENABLED:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

FACEBOOK_CLIENT_ID = os.environ.get("FACEBOOK_CLIENT_ID")
FACEBOOK_CLIENT_SECRET = os.environ.get("FACEBOOK_CLIENT_SECRET")
FACEBOOK_ENABLED = bool(FACEBOOK_CLIENT_ID and FACEBOOK_CLIENT_SECRET)

if FACEBOOK_ENABLED:
    oauth.register(
        name="facebook",
        client_id=FACEBOOK_CLIENT_ID,
        client_secret=FACEBOOK_CLIENT_SECRET,
        access_token_url="https://graph.facebook.com/oauth/access_token",
        authorize_url="https://www.facebook.com/dialog/oauth",
        api_base_url="https://graph.facebook.com/",
        client_kwargs={"scope": "email public_profile"},
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.json")
RETRIEVAL_REPORT_PATH = os.path.join(BASE_DIR, "rag_evaluation_report.json")

# Ensure DB + tables exist on startup
database.init_db()

# Warm up the ML models once at startup, not on the first user's request -
# loading the pickled models from disk takes ~1-2 seconds; without this,
# whoever submits the first ticket after a server restart would eat that
# cost. After warm-up, real classification takes ~1-2ms per ticket.
from classifier import _load_models as _warm_up_classifier
_warm_up_classifier()
get_orchestrator()  # builds the KB retriever's TF-IDF index once, not on the first request


def load_json_report(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# JWT-based authentication
#
# No server-side session store: every request carries a signed JWT (in an
# httponly cookie for browser pages, or an Authorization: Bearer header for
# API clients). g.current_user is populated once per request by decoding
# and verifying that token against app.secret_key.
# ---------------------------------------------------------------------------
@app.before_request
def load_current_user():
    token = get_token_from_request(request)
    payload = decode_token(token, app.secret_key) if token else None
    g.current_user = payload  # dict with user_id/email/name, or None


@app.context_processor
def inject_current_user():
    return {"current_user": g.get("current_user")}


def issue_token_cookie(response, user):
    token = generate_token(user, app.secret_key)
    response.set_cookie(
        TOKEN_COOKIE_NAME, token,
        httponly=True, samesite="Lax",
        max_age=TOKEN_EXPIRY_HOURS * 3600,
    )
    return response


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not g.get("current_user"):
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped


def api_login_required(view_func):
    """JSON-friendly version for API routes: 401 instead of a redirect."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not g.get("current_user"):
            return jsonify({"error": "Missing or invalid JWT. Send it as 'Authorization: Bearer <token>'."}), 401
        return view_func(*args, **kwargs)
    return wrapped


# ---------------------------------------------------------------------------
# Authentication  (email is the login identifier, per request)
# ---------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")

    if not email or not password:
        return render_template("register.html", error="Email and password are required.",
                                google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)
    if "@" not in email or "." not in email.split("@")[-1]:
        return render_template("register.html", error="Enter a valid email address.",
                                google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)
    if password != confirm:
        return render_template("register.html", error="Passwords do not match.",
                                google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)
    if len(password) < 6:
        return render_template("register.html", error="Password must be at least 6 characters.",
                                google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)

    user_id = database.create_user(email, password, name)
    if user_id is None:
        return render_template("register.html", error="An account with that email already exists.",
                                google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)

    user = database.get_user_by_id(user_id)
    response = redirect(url_for("index"))
    return issue_token_cookie(response, user)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", next=request.args.get("next", ""),
                                google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    next_url = request.form.get("next") or url_for("index")

    user = database.verify_login(email, password)
    if not user:
        return render_template("login.html", error="Incorrect email or password.", next=next_url,
                                google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)

    response = redirect(next_url)
    return issue_token_cookie(response, user)


@app.route("/logout")
def logout():
    response = redirect(url_for("login"))
    response.delete_cookie(TOKEN_COOKIE_NAME)
    return response


# ---------------------------------------------------------------------------
# JWT API endpoints - for external/API clients (e.g. Postman, a mobile app)
# rather than the browser session flow above.
# ---------------------------------------------------------------------------
@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True, silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = database.verify_login(email, password)
    if not user:
        return jsonify({"error": "Incorrect email or password"}), 401

    token = generate_token(user, app.secret_key)
    return jsonify({
        "token": token,
        "token_type": "Bearer",
        "expires_in_hours": TOKEN_EXPIRY_HOURS,
        "user": {"user_id": user["user_id"], "email": user["email"], "name": user.get("name")},
    })


@app.route("/api/me", methods=["GET"])
@api_login_required
def api_me():
    return jsonify(g.current_user)


# ---------------------------------------------------------------------------
# OAuth routes ("Sign in with Google" / "Sign in with Facebook")
# ---------------------------------------------------------------------------
@app.route("/auth/google/login")
def google_login():
    if not GOOGLE_ENABLED:
        return render_template("login.html", error="Google sign-in isn't configured on this server yet.",
                                google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)
    redirect_uri = url_for("google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    token = oauth.google.authorize_access_token()
    userinfo = token.get("userinfo") or oauth.google.parse_id_token(token)
    email = userinfo["email"]
    name = userinfo.get("name", "")
    google_id = userinfo.get("sub")

    user = database.get_or_create_oauth_user(email, name, "google", google_id)
    response = redirect(url_for("index"))
    return issue_token_cookie(response, user)


@app.route("/auth/facebook/login")
def facebook_login():
    if not FACEBOOK_ENABLED:
        return render_template("login.html", error="Facebook sign-in isn't configured on this server yet.",
                                google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED)
    redirect_uri = url_for("facebook_callback", _external=True)
    return oauth.facebook.authorize_redirect(redirect_uri)


@app.route("/auth/facebook/callback")
def facebook_callback():
    token = oauth.facebook.authorize_access_token()
    resp = oauth.facebook.get("me?fields=id,name,email", token=token)
    profile = resp.json()
    email = profile.get("email")
    if not email:
        return render_template(
            "login.html",
            error="Facebook did not share an email for this account. Try Google or email sign-in instead.",
            google_enabled=GOOGLE_ENABLED, facebook_enabled=FACEBOOK_ENABLED,
        )
    name = profile.get("name", "")
    facebook_id = profile.get("id")

    user = database.get_or_create_oauth_user(email, name, "facebook", facebook_id)
    response = redirect(url_for("index"))
    return issue_token_cookie(response, user)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
@login_required
def submit_ticket():
    employee_name = request.form.get("employee_name", "").strip() or g.current_user.get("name") or g.current_user.get("email")
    requester_email = g.current_user.get("email")
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    department = request.form.get("department", "").strip()

    if not description:
        return render_template("index.html", error="Ticket description is required.")

    full_text = f"{title}. {description}" if title else description

    # --- Milestone 3: run the full multi-agent workflow ---
    # Diagnosis -> Retrieval -> Resolution -> Validation -> (auto-resolve or
    # escalate to Jira), with every step timed and logged for the UI.
    start = time.perf_counter()
    agent_result = get_orchestrator().process_ticket(
        full_text, employee_name=employee_name, requester_email=requester_email, title=title
    )
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    diagnosis = agent_result["diagnosis"]
    retrieval = agent_result["retrieval"]
    resolution = agent_result["resolution"]
    validation = agent_result["validation"]
    jira = agent_result["jira"] or {}
    email = agent_result["email"] or {}

    ticket_id = database.insert_ticket(
        employee_name=employee_name,
        email=requester_email,
        title=title,
        description=description,
        department=department,
        category=diagnosis["category"],
        severity=diagnosis["severity"],
        priority=diagnosis["priority"],
        confidence=diagnosis["confidence"] / 100,  # stored 0-1 for consistency with earlier tickets
        causes=diagnosis["causes"],
        resolution=resolution["resolution"],
        retrieved_docs=retrieval["documents"],
        rag_duration_ms=duration_ms,
        validation_confidence=validation["confidence"],
        validation_status=validation["status"],
        jira_ticket_key=jira.get("ticket_key"),
        jira_ticket_url=jira.get("ticket_url"),
        email_sent="yes" if email.get("sent") else ("no" if email else None),
        workflow_log=agent_result["workflow_log"],
        status="Auto-Resolved" if validation["status"] == "AUTO_RESOLVE" else "Escalated",
    )

    # Build the same "result"/"rag" shape the template already expects,
    # so the existing Submit page rendering keeps working unchanged.
    result = {
        "category": diagnosis["category"],
        "category_confidence": diagnosis["confidence"] / 100,
        "severity": diagnosis["severity"],
        "priority": diagnosis["priority"],
        "business_impact": diagnosis["business_impact"],
        "possible_causes": diagnosis["causes"],
    }
    rag = {
        "status": "OK" if retrieval["documents"] else "INSUFFICIENT_KNOWLEDGE",
        "message": None if retrieval["documents"] else "No sufficiently relevant knowledge-base articles were found.",
        "retrieved_documents": retrieval["documents"],
        "resolution": resolution["resolution"],
        "steps": resolution["steps"],
        "workflow": {
            "ticket_analysis": "completed",
            "knowledge_retrieval": "completed" if retrieval["documents"] else "completed",
            "context_augmentation": "completed" if retrieval["documents"] else "skipped",
            "response_generation": "completed" if resolution["steps"] else "skipped",
        },
    }

    return render_template(
        "index.html",
        result=result,
        rag=rag,
        rag_duration_ms=duration_ms,
        ticket_id=ticket_id,
        submitted_title=title,
        validation=validation,
        jira=jira,
        email=email,
    )


@app.route("/ticket/<int:ticket_id>/feedback", methods=["POST"])
@login_required
def ticket_feedback(ticket_id):
    resolved = request.form.get("resolved") == "yes"
    database.set_ticket_resolved(ticket_id, resolved)
    return redirect(url_for("list_tickets_page"))



@app.route("/tickets", methods=["GET"])
@login_required
def list_tickets_page():
    tickets = database.get_all_tickets(limit=50, email=g.current_user.get("email"))
    return render_template("index.html", tickets=tickets, show_tickets=True)


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    stats = database.get_stats(email=g.current_user.get("email"))
    return render_template("index.html", stats=stats, show_dashboard=True)


@app.route("/ai-agent", methods=["GET"])
@login_required
def ai_agent():
    classification_report = load_json_report(CLASSIFICATION_REPORT_PATH)
    retrieval_report = load_json_report(RETRIEVAL_REPORT_PATH)
    stats = database.get_stats(email=g.current_user.get("email"))

    # Milestone 3: multi-agent workflow status and the current user's most
    # recently processed ticket's Jira/email outcome (mirrors the deck's
    # "Multi-Agent Workflow" screenshot layout).
    recent_tickets = database.get_all_tickets(limit=1, email=g.current_user.get("email"))
    latest_ticket = recent_tickets[0] if recent_tickets else None

    agent_status = {
        "diagnosis": "Active",
        "retrieval": "Active",
        "resolution": "Active",
        "escalation": "Standby" if not latest_ticket or latest_ticket.get("validation_status") != "ESCALATE" else "Active",
    }

    return render_template(
        "index.html",
        show_ai_agent=True,
        classification_report=classification_report,
        retrieval_report=retrieval_report,
        stats=stats,
        agent_status=agent_status,
        latest_ticket=latest_ticket,
    )


@app.route("/integrations", methods=["GET"])
@login_required
def integrations():
    jira = JiraService()
    email = EmailService()
    return render_template(
        "index.html",
        show_integrations=True,
        jira_configured=jira.is_configured,
        email_configured=email.is_configured,
    )


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------
@app.route("/api/ticket", methods=["POST"])
def api_create_ticket():
    data = request.get_json(force=True, silent=True) or {}
    description = data.get("description", "").strip()
    title = data.get("title", "")
    employee_name = data.get("employee", data.get("employee_name", ""))
    requester_email = data.get("email", "")
    department = data.get("department", "")

    if not description:
        return jsonify({"error": "'description' is required"}), 400

    full_text = f"{title}. {description}" if title else description

    start = time.perf_counter()
    agent_result = get_orchestrator().process_ticket(
        full_text, employee_name=employee_name, requester_email=requester_email, title=title
    )
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    diagnosis = agent_result["diagnosis"]
    retrieval = agent_result["retrieval"]
    resolution = agent_result["resolution"]
    validation = agent_result["validation"]
    jira = agent_result["jira"] or {}
    email = agent_result["email"] or {}

    ticket_id = database.insert_ticket(
        employee_name=employee_name,
        email=requester_email,
        title=title,
        description=description,
        department=department,
        category=diagnosis["category"],
        severity=diagnosis["severity"],
        priority=diagnosis["priority"],
        confidence=diagnosis["confidence"] / 100,
        causes=diagnosis["causes"],
        resolution=resolution["resolution"],
        retrieved_docs=retrieval["documents"],
        rag_duration_ms=duration_ms,
        validation_confidence=validation["confidence"],
        validation_status=validation["status"],
        jira_ticket_key=jira.get("ticket_key"),
        jira_ticket_url=jira.get("ticket_url"),
        email_sent="yes" if email.get("sent") else ("no" if email else None),
        workflow_log=agent_result["workflow_log"],
        status="Auto-Resolved" if validation["status"] == "AUTO_RESOLVE" else "Escalated",
    )

    response = {
        "ticket_id": ticket_id,
        "ticket": description,
        "diagnosis": diagnosis,
        "retrieval": {"documents": retrieval["documents"], "count": retrieval["count"],
                      "top_similarity": retrieval["top_similarity"]},
        "resolution": resolution["resolution"],
        "resolution_steps": resolution["steps"],
        "validation": validation,
        "jira": jira,
        "email": email,
        "workflow_log": agent_result["workflow_log"],
        "processing_time_ms": duration_ms,
        "status": "Auto-Resolved" if validation["status"] == "AUTO_RESOLVE" else "Escalated",
    }
    return jsonify(response), 201


@app.route("/api/tickets", methods=["GET"])
def api_list_tickets():
    tickets = database.get_all_tickets(limit=100)
    return jsonify(tickets)


@app.route("/api/stats", methods=["GET"])
def api_stats():
    return jsonify(database.get_stats())


if __name__ == "__main__":
    import webbrowser
    import threading

    # Opens your default browser (Chrome, if that's your default) automatically
    # a moment after the server starts, so you land on the app instead of
    # staring at the VS Code terminal. Only fires once, not on Flask's
    # debug-mode auto-reload.
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.25, lambda: webbrowser.open("http://127.0.0.1:5000")).start()

    app.run(debug=True, host="0.0.0.0", port=5000)
