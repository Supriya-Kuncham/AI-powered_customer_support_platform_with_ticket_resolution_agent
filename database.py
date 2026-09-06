"""
database.py
------------
SQLite persistence layer for SupportPilot.

Tables:
  users   - real authentication, login by EMAIL (not username)
  tickets - ticket data, Milestone 1 classification fields, and Milestone 2
            RAG fields (retrieved KB docs, generated resolution, feedback)
"""

import sqlite3
import os
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "tickets.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_name VARCHAR(100),
            email VARCHAR(150),
            title VARCHAR(255),
            description TEXT,
            department VARCHAR(50),
            category VARCHAR(100),
            severity VARCHAR(20),
            priority VARCHAR(10),
            confidence FLOAT,
            causes TEXT,
            resolution TEXT,
            retrieved_docs TEXT,
            rag_duration_ms FLOAT,
            resolved VARCHAR(10),
            status VARCHAR(30) DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # lightweight migrations for DBs created before these columns existed
    cur.execute("PRAGMA table_info(tickets)")
    columns = {row[1] for row in cur.fetchall()}
    for col, coltype in [
        ("causes", "TEXT"),
        ("resolution", "TEXT"),
        ("retrieved_docs", "TEXT"),
        ("rag_duration_ms", "FLOAT"),
        ("resolved", "VARCHAR(10)"),
    ]:
        if col not in columns:
            cur.execute(f"ALTER TABLE tickets ADD COLUMN {col} {coltype}")

    # ---------------- users table (real authentication, login by email) ----------------
    # If an older username-based users table exists (from before this milestone),
    # drop and recreate it - the old schema's username NOT NULL constraint is
    # incompatible with email-based signup and would break every registration.
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if cur.fetchone():
        cur.execute("PRAGMA table_info(users)")
        user_columns = {row[1] for row in cur.fetchall()}
        if "username" in user_columns and "email" not in user_columns:
            cur.execute("DROP TABLE users")
        elif "username" in user_columns:
            cur.execute("DROP TABLE users")  # old schema, rebuild with email as the key

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email VARCHAR(150) UNIQUE NOT NULL,
            name VARCHAR(100),
            password_hash VARCHAR(255),
            provider VARCHAR(20) DEFAULT 'local',
            oauth_id VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # migration: older DBs may have password_hash NOT NULL / lack provider columns
    cur.execute("PRAGMA table_info(users)")
    user_cols = {row[1] for row in cur.fetchall()}
    if "provider" not in user_cols or "oauth_id" not in user_cols:
        # SQLite can't easily relax NOT NULL / add complex columns in place;
        # since this is local dev data, rebuild the table with the new schema.
        cur.execute("ALTER TABLE users RENAME TO users_old")
        cur.execute(
            """
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                email VARCHAR(150) UNIQUE NOT NULL,
                name VARCHAR(100),
                password_hash VARCHAR(255),
                provider VARCHAR(20) DEFAULT 'local',
                oauth_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "INSERT INTO users (user_id, email, name, password_hash, created_at) "
            "SELECT user_id, email, name, password_hash, created_at FROM users_old"
        )
        cur.execute("DROP TABLE users_old")

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Auth helpers  (email is the login identifier)
# ---------------------------------------------------------------------------
def create_user(email, password, name=None):
    """Returns the new user_id, or None if the email is already registered.
    Used for local (email+password) signup."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (email, name, password_hash, provider, created_at) VALUES (?, ?, ?, 'local', ?)",
            (email.strip().lower(), name, generate_password_hash(password), datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_or_create_oauth_user(email, name, provider, oauth_id):
    """
    Logs in an existing account by email, or creates a new passwordless
    account for a Google/Facebook sign-in. Returns the user dict.
    """
    email = email.strip().lower()
    existing = get_user_by_email(email)
    if existing:
        return existing

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (email, name, password_hash, provider, oauth_id, created_at) "
        "VALUES (?, ?, NULL, ?, ?, ?)",
        (email, name, provider, oauth_id, datetime.utcnow().isoformat()),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return get_user_by_id(user_id)


def get_user_by_email(email):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def verify_login(email, password):
    """Returns the user dict if the email/password combo is correct, else None.
    OAuth-only accounts (no password_hash) can't log in this way."""
    user = get_user_by_email(email)
    if user and user.get("password_hash") and check_password_hash(user["password_hash"], password):
        return user
    return None


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------
def insert_ticket(employee_name, email, title, description, category,
                   severity, priority, confidence, department=None,
                   status="Open", causes=None, resolution=None,
                   retrieved_docs=None, rag_duration_ms=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tickets
            (employee_name, email, title, description, department,
             category, severity, priority, confidence, causes,
             resolution, retrieved_docs, rag_duration_ms, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            employee_name, email, title, description, department or category,
            category, severity, priority, confidence,
            json.dumps(causes or []), resolution,
            json.dumps(retrieved_docs or []), rag_duration_ms, status,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    ticket_id = cur.lastrowid
    conn.close()
    return ticket_id


def _deserialize_ticket(row):
    t = dict(row)
    try:
        t["causes"] = json.loads(t.get("causes") or "[]")
    except (TypeError, json.JSONDecodeError):
        t["causes"] = []
    try:
        t["retrieved_docs"] = json.loads(t.get("retrieved_docs") or "[]")
    except (TypeError, json.JSONDecodeError):
        t["retrieved_docs"] = []
    return t


def get_all_tickets(limit=100):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickets ORDER BY ticket_id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [_deserialize_ticket(r) for r in rows]


def get_ticket(ticket_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    row = cur.fetchone()
    conn.close()
    return _deserialize_ticket(row) if row else None


def set_ticket_resolved(ticket_id, resolved: bool):
    """Records whether the AI-generated resolution actually solved the issue.
    This is the real, honest source for the 'resolution rate' metric."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE tickets SET resolved = ? WHERE ticket_id = ?",
        ("yes" if resolved else "no", ticket_id),
    )
    conn.commit()
    conn.close()


def get_stats():
    """Aggregate counts used by the dashboard charts."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT category, COUNT(*) as c FROM tickets GROUP BY category ORDER BY c DESC")
    by_category = [{"label": r["category"], "count": r["c"]} for r in cur.fetchall()]

    cur.execute("SELECT severity, COUNT(*) as c FROM tickets GROUP BY severity")
    by_severity = [{"label": r["severity"], "count": r["c"]} for r in cur.fetchall()]

    cur.execute("SELECT priority, COUNT(*) as c FROM tickets GROUP BY priority ORDER BY priority")
    by_priority = [{"label": r["priority"], "count": r["c"]} for r in cur.fetchall()]

    cur.execute("SELECT COUNT(*) as c FROM tickets")
    total = cur.fetchone()["c"]

    cur.execute("SELECT AVG(rag_duration_ms) as a FROM tickets WHERE rag_duration_ms IS NOT NULL")
    avg_row = cur.fetchone()
    avg_response_time_ms = round(avg_row["a"], 0) if avg_row["a"] is not None else None

    cur.execute("SELECT COUNT(*) as c FROM tickets WHERE resolved IS NOT NULL")
    feedback_total = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) as c FROM tickets WHERE resolved = 'yes'")
    feedback_resolved = cur.fetchone()["c"]
    resolution_rate = round((feedback_resolved / feedback_total) * 100, 1) if feedback_total > 0 else None

    conn.close()
    return {
        "total": total,
        "by_category": by_category,
        "by_severity": by_severity,
        "by_priority": by_priority,
        "avg_response_time_ms": avg_response_time_ms,
        "feedback_total": feedback_total,
        "resolution_rate": resolution_rate,
    }


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
