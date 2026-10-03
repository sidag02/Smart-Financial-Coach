"""Bearer tokens for the MCP server: who is calling, signed by the app, with an expiry.

A signed-in user gets a personal access token from the "Connect an assistant" page for an outside
assistant (FR-19); the coach gets a short-lived one for the session's user on every question. The
token is the only place the MCP server learns whose data to read: no tool takes a `user_id`
(Technical Design, "Security and data isolation"). Tokens are signed with the session secret, so
rotating `SFC_SESSION_SECRET` revokes them all.

    tokens = AccessTokens(secret, known_users)
    token = tokens.issue("u_te_yp_0030", timedelta(days=7), feedback_subject=session_feedback_id)
    tokens.user(token)  # "u_te_yp_0030", or None when forged, expired or for an unknown user
    tokens.feedback_subject(token)  # whose category feedback the calls read and write

The feedback subject (FR-5, FR-6) is the user in production; in the demo, where visitors share
accounts, it's the browser session that issued the token, so an assistant connected from a
session sees and changes that session's corrections only. It's signed like the user.
"""

import time
from collections.abc import Collection
from datetime import timedelta

from itsdangerous import BadSignature, URLSafeSerializer
from mcp.server.auth.provider import AccessToken

PREFIX = "sfc_"
SALT = "sfc-mcp-access-token"


class AccessTokens:
    def __init__(self, secret: str, known_users: Collection[str]) -> None:
        self._signer = URLSafeSerializer(secret, salt=SALT)
        self._users = frozenset(known_users)

    def issue(
        self,
        user_id: str,
        lifetime: timedelta,
        *,
        client: str = "assistant",
        feedback_subject: str | None = None,
    ) -> str:
        if user_id not in self._users:
            raise ValueError(f"unknown user {user_id!r}")
        expires = int(time.time() + lifetime.total_seconds())
        claims = {
            "sub": user_id,
            "exp": expires,
            "client": client,
            "fb": feedback_subject or user_id,
        }
        return PREFIX + self._signer.dumps(claims)

    def claims(self, token: str) -> dict[str, object] | None:
        """The token's claims if it's genuine, unexpired and for a known user."""
        if not token.startswith(PREFIX):
            return None
        try:
            claims = self._signer.loads(token[len(PREFIX) :])
        except BadSignature:
            return None
        if not isinstance(claims, dict) or claims.get("sub") not in self._users:
            return None
        expires = claims.get("exp")
        if not isinstance(expires, int) or expires <= time.time():
            return None
        return claims

    def user(self, token: str) -> str | None:
        claims = self.claims(token)
        return str(claims["sub"]) if claims else None

    def feedback_subject(self, token: str) -> str | None:
        """Whose category feedback the token's calls use. None for a token issued before
        feedback existed: falling back to the account would put every visitor's assistant on one
        shared subject, so such a token gets read-only tools (review on #36)."""
        claims = self.claims(token)
        fb = claims.get("fb") if claims else None
        return str(fb) if fb else None

    async def verify_token(self, token: str) -> AccessToken | None:
        """The MCP SDK's `TokenVerifier`: a valid token's user becomes the token's subject."""
        claims = self.claims(token)
        if claims is None:
            return None
        return AccessToken(
            token=token,
            client_id=str(claims.get("client", "assistant")),
            scopes=[],
            expires_at=int(str(claims["exp"])),
            subject=str(claims["sub"]),
        )
