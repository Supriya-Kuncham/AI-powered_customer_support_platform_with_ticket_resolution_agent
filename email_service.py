"""
email_service.py
-------------------
Milestone 3: Email automation.

Sends real emails via SMTP when a ticket is auto-resolved or escalated.
Requires SMTP_EMAIL, SMTP_PASSWORD (an app password, not your real
password, if using Gmail), SMTP_SERVER, SMTP_PORT - see .env.example.
Without them, this gracefully reports "not configured" instead of
crashing, same pattern as Jira and OAuth.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmailService:
    def __init__(self):
        self.email = os.environ.get("SMTP_EMAIL", "")
        self.password = os.environ.get("SMTP_PASSWORD", "")
        self.server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.port = int(os.environ.get("SMTP_PORT", "587"))

    @property
    def is_configured(self) -> bool:
        return bool(self.email and self.password)

    def send_email(self, to_email: str, subject: str, body: str) -> dict:
        if not self.is_configured:
            return {
                "sent": False,
                "configured": False,
                "message": "Email is not connected. Set SMTP_EMAIL and SMTP_PASSWORD in .env "
                            "to enable real email sending.",
            }
        if not to_email:
            return {"sent": False, "configured": True, "message": "No recipient email address on this ticket."}

        message = MIMEMultipart()
        message["From"] = self.email
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.server, self.port, timeout=10) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.sendmail(self.email, to_email, message.as_string())
            return {"sent": True, "configured": True, "to": to_email}
        except Exception as exc:  # noqa: BLE001 - report any SMTP failure back to the caller
            return {"sent": False, "configured": True, "message": f"Email send failed: {exc}"}
