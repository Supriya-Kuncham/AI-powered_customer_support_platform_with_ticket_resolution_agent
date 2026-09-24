"""
auth_jwt.py
------------
Real JWT (JSON Web Token) authentication, using PyJWT.

How it works:
  - On login/register/OAuth success, a signed JWT is issued containing the
    user's id/email/name and an expiry time, and stored in an httponly cookie.
  - On every request, the token is read back from that cookie (or from an
    Authorization: Bearer <token> header, for API clients), verified against
    the app's SECRET_KEY, and decoded to identify the logged-in user.
  - No server-side session store is used to track "who is logged in" - the
    token itself carries that information, signed so it can't be tampered
    with. This is what makes it JWT-based auth rather than Flask's default
    session-cookie auth.
"""

import jwt
from datetime import datetime, timedelta, timezone

TOKEN_COOKIE_NAME = "access_token"
TOKEN_EXPIRY_HOURS = 24 * 7  # 7 days


def generate_token(user: dict, secret_key: str) -> str:
    payload = {
        "user_id": user["user_id"],
        "email": user["email"],
        "name": user.get("name"),
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS),
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")


def decode_token(token: str, secret_key: str):
    """Returns the decoded payload dict, or None if invalid/expired/missing."""
    if not token:
        return None
    try:
        return jwt.decode(token, secret_key, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_token_from_request(request):
    """Checks the httponly cookie first (browser pages), then an
    Authorization: Bearer <token> header (API clients)."""
    token = request.cookies.get(TOKEN_COOKIE_NAME)
    if token:
        return token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[len("Bearer "):]
    return None
