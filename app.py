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

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from authlib.integrations.flask_client import OAuth

import database
from classifier import process_ticket
from rag_pipeline import run_rag_pipeline

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


def load_json_report(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
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

    session["user_id"] = user_id
    session["email"] = email
    session["name"] = name
    return redirect(url_for("index"))


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

    session["user_id"] = user["user_id"]
    session["email"] = user["email"]
    session["name"] = user.get("name")
    return redirect(next_url)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


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
    session["user_id"] = user["user_id"]
    session["email"] = user["email"]
    session["name"] = user.get("name")
    return redirect(url_for("index"))


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
    session["user_id"] = user["user_id"]
    session["email"] = user["email"]
    session["name"] = user.get("name")
    return redirect(url_for("index"))


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
    employee_name = request.form.get("employee_name", "").strip() or session.get("name") or session.get("email")
    requester_email = session.get("email")
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    department = request.form.get("department", "").strip()

    if not description:
        return render_template("index.html", error="Ticket description is required.")

    full_text = f"{title}. {description}" if title else description

    # --- Milestone 1: classification, severity, priority, likely causes ---
    classification = process_ticket(full_text)

    # --- Milestone 2: knowledge-base retrieval + cited resolution (RAG) ---
    rag_start = time.perf_counter()
    rag_result = run_rag_pipeline(full_text)
    rag_duration_ms = round((time.perf_counter() - rag_start) * 1000, 1)

    ticket_id = database.insert_ticket(
        employee_name=employee_name,
        email=requester_email,
        title=title,
        description=description,
        department=department,
        category=classification["category"],
        severity=classification["severity"],
        priority=classification["priority"],
        confidence=classification["category_confidence"],
        causes=classification["possible_causes"],
        resolution=rag_result["resolution"],
        retrieved_docs=rag_result["retrieved_documents"],
        rag_duration_ms=rag_duration_ms,
    )

    return render_template(
        "index.html",
        result=classification,
        rag=rag_result,
        rag_duration_ms=rag_duration_ms,
        ticket_id=ticket_id,
        submitted_title=title,
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
    tickets = database.get_all_tickets(limit=50)
    return render_template("index.html", tickets=tickets, show_tickets=True)


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    stats = database.get_stats()
    return render_template("index.html", stats=stats, show_dashboard=True)


@app.route("/ai-agent", methods=["GET"])
@login_required
def ai_agent():
    classification_report = load_json_report(CLASSIFICATION_REPORT_PATH)
    retrieval_report = load_json_report(RETRIEVAL_REPORT_PATH)
    stats = database.get_stats()
    return render_template(
        "index.html",
        show_ai_agent=True,
        classification_report=classification_report,
        retrieval_report=retrieval_report,
        stats=stats,
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

    classification = process_ticket(full_text)

    rag_start = time.perf_counter()
    rag_result = run_rag_pipeline(full_text)
    rag_duration_ms = round((time.perf_counter() - rag_start) * 1000, 1)

    ticket_id = database.insert_ticket(
        employee_name=employee_name,
        email=requester_email,
        title=title,
        description=description,
        department=department,
        category=classification["category"],
        severity=classification["severity"],
        priority=classification["priority"],
        confidence=classification["category_confidence"],
        causes=classification["possible_causes"],
        resolution=rag_result["resolution"],
        retrieved_docs=rag_result["retrieved_documents"],
        rag_duration_ms=rag_duration_ms,
    )

    response = {
        "ticket_id": ticket_id,
        "ticket": description,
        "category": classification["category"],
        "category_confidence": classification["category_confidence"],
        "severity": classification["severity"],
        "severity_confidence": classification["severity_confidence"],
        "priority": classification["priority"],
        "possible_causes": classification["possible_causes"],
        "rag_status": rag_result["status"],
        "retrieved_documents": rag_result["retrieved_documents"],
        "resolution": rag_result["resolution"],
        "resolution_steps": rag_result["steps"],
        "workflow": rag_result["workflow"],
        "processing_time_ms": rag_duration_ms,
        "status": "Open",
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
    app.run(debug=True, host="0.0.0.0", port=5000)
