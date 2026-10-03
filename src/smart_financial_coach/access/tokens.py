"""Bearer tokens for the MCP server: who is calling, signed by the app, with an expiry.

A signed-in user gets a personal access token from the "Connect an assistant" page for an outside
assistant (FR-19); the coach gets a short-lived one for the session's user on every question. The
token is the only place the MCP server learns whose data to read: no tool takes a `user_id`
(Technical Design, "Security and data isolation"). Tokens are signed with the session secret, so
rotating `SFC_SESSION_SECRET` revokes them all.

    tokens = AccessTokens(secret, known_users)
    token = tokens.issue("u_te_yp_0030", timedelta(days=7))
    tokens.user(token)  # "u_te_yp_0030", or None when forged, expired or for an unknown user
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

    def issue(self, user_id: str, lifetime: timedelta, *, client: str = "assistant") -> str:
        if user_id not in self._users:
            raise ValueError(f"unknown user {user_id!r}")
        expires = int(time.time() + lifetime.total_seconds())
        return PREFIX + self._signer.dumps({"sub": user_id, "exp": expires, "client": client})

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
