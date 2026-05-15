"""OAuth provider factory for yugabytedb-mcp-server.

Wraps FastMCP's OIDCProxy + JWTVerifier. Supports two providers:
- cognito: tested against AWS Cognito user pools
- oidc:    generic OIDC issuer; code present but untested

No scope validation: any authenticated request can call any tool. Auth gates
access to the server itself; tool-level permissions are not enforced.

Also exposes `cognito_password_login()` — a helper that calls Cognito's
USER_PASSWORD_AUTH flow directly so users can fetch a token with email +
password (no browser). Useful for curl-based smoke tests and CI. Exposed via
the `/auth/login` HTTP route in server.py when MCP_AUTH_PROVIDER=cognito.

For multi-replica deployments behind a load balancer, the default per-process
file-tree client_storage is not shared across pods, which breaks DCR mid-flow.
A Postgres-backed shared store exists in meko-mcp-server and can be ported
here when a self-hoster needs it; defer until then.
"""

import base64
import hashlib
import hmac
import logging
import os
from typing import Any

logger = logging.getLogger("yugabytedb-mcp.auth")

SUPPORTED_PROVIDERS = ["cognito", "oidc"]


def create_auth_provider(name: str | None):
    """Factory that returns a configured auth provider instance based on name,
    or None when auth is disabled.
    """
    if name is None:
        logger.info("No auth provider configured, auth disabled")
        return None
    logger.info("Creating auth provider: %s", name)
    logger.debug("MCP_BASE_URL=%s", os.environ.get("MCP_BASE_URL", "<unset>"))
    if name == "cognito":
        return _create_cognito()
    elif name == "oidc":
        return _create_oidc()
    else:
        logger.error("Unknown auth provider: %s", name)
        raise ValueError(
            f"Unknown auth provider: {name!r}. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )


def _create_cognito():
    from fastmcp.server.auth.oidc_proxy import OIDCProxy
    from fastmcp.server.auth.auth import MultiAuth
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    class _CognitoProxy(OIDCProxy):
        """OIDCProxy tuned for AWS Cognito's quirks:

        - Cognito's token endpoint rejects `scope` in authorization_code exchange
        - The `resource` (RFC 8707) parameter in the authorize URL causes invalid_grant
        - Confidential clients don't need PKCE on the upstream leg
        - `client_secret_basic` is what Cognito expects (`client_secret_post` may fail)
        """

        def _prepare_scopes_for_token_exchange(self, scopes: list[str]) -> list[str]:
            return []

    pool_id = os.environ["COGNITO_USER_POOL_ID"]
    region = os.environ["COGNITO_AWS_REGION"]
    logger.debug("Configuring Cognito provider (pool=%s, region=%s)", pool_id, region)
    config_url = (
        f"https://cognito-idp.{region}.amazonaws.com/{pool_id}/"
        ".well-known/openid-configuration"
    )

    scopes = "openid email profile"
    base_url = os.environ.get("MCP_BASE_URL", "http://localhost:8000")
    logger.debug("Cognito base_url=%s, scopes=%s", base_url, scopes)

    proxy = _CognitoProxy(
        config_url=config_url,
        client_id=os.environ["COGNITO_CLIENT_ID"],
        client_secret=os.environ["COGNITO_CLIENT_SECRET"],
        base_url=base_url,
        token_endpoint_auth_method="client_secret_basic",
        extra_authorize_params={"scope": scopes},
        forward_resource=False,
    )
    proxy._forward_pkce = False

    proxy.client_registration_options.valid_scopes = scopes.split()
    proxy.client_registration_options.default_scopes = scopes.split()
    proxy._default_scope_str = scopes
    if getattr(proxy, "_cimd_manager", None) is not None:
        proxy._cimd_manager.default_scope = scopes

    raw_jwt_verifier = JWTVerifier(
        jwks_uri=str(proxy.oidc_config.jwks_uri),
        issuer=str(proxy.oidc_config.issuer),
    )

    logger.info("Cognito auth provider created (pool=%s, region=%s)", pool_id, region)
    return MultiAuth(server=proxy, verifiers=[raw_jwt_verifier])


def _create_oidc():
    """Generic OIDC provider. Untested — exercise at your own risk and please
    report findings."""
    from fastmcp.server.auth.oidc_proxy import OIDCProxy
    from fastmcp.server.auth.auth import MultiAuth
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    config_url = os.environ["OIDC_CONFIG_URL"]
    client_id = os.environ["OIDC_CLIENT_ID"]
    client_secret = os.environ["OIDC_CLIENT_SECRET"]
    audience = os.environ.get("OIDC_AUDIENCE")

    proxy = OIDCProxy(
        config_url=config_url,
        client_id=client_id,
        client_secret=client_secret,
        audience=audience,
        base_url=os.environ.get("MCP_BASE_URL", "http://localhost:8000"),
    )

    raw_jwt_verifier = JWTVerifier(
        jwks_uri=str(proxy.oidc_config.jwks_uri),
        issuer=str(proxy.oidc_config.issuer),
    )

    logger.info("OIDC auth provider created (config_url=%s)", config_url)
    return MultiAuth(server=proxy, verifiers=[raw_jwt_verifier])


# ---------------------------------------------------------------------------
# Email/password → token helper for Cognito (USER_PASSWORD_AUTH flow)
# ---------------------------------------------------------------------------

class CognitoLoginError(Exception):
    """Raised when Cognito rejects credentials or initiate_auth fails."""

    def __init__(self, code: str, detail: str, status: int = 401):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status


def _cognito_secret_hash(username: str, client_id: str, client_secret: str) -> str:
    """Cognito's SECRET_HASH = Base64(HMAC-SHA256(client_secret, username + client_id))."""
    msg = (username + client_id).encode("utf-8")
    secret = client_secret.encode("utf-8")
    return base64.b64encode(hmac.new(secret, msg, hashlib.sha256).digest()).decode()


def cognito_password_login(email: str, password: str) -> dict[str, Any]:
    """Exchange email + password for Cognito tokens via USER_PASSWORD_AUTH.

    Requires `COGNITO_USER_POOL_ID`, `COGNITO_AWS_REGION`, `COGNITO_CLIENT_ID`
    and (if the app client has a client secret) `COGNITO_CLIENT_SECRET` to be
    set. The app client must have the `ALLOW_USER_PASSWORD_AUTH` auth flow
    enabled — toggle it in Cognito → App integration → App client settings.

    Returns the Cognito `AuthenticationResult` dict directly (keys:
    `AccessToken`, `IdToken`, `RefreshToken`, `ExpiresIn`, `TokenType`).
    Raises `CognitoLoginError` on any failure with an HTTP-appropriate status.
    """
    import boto3
    from botocore.exceptions import ClientError

    try:
        client_id = os.environ["COGNITO_CLIENT_ID"]
        region = os.environ["COGNITO_AWS_REGION"]
    except KeyError as e:
        raise CognitoLoginError(
            "server_misconfigured",
            f"Missing env var: {e.args[0]}",
            status=503,
        ) from e

    auth_params: dict[str, str] = {"USERNAME": email, "PASSWORD": password}
    client_secret = os.environ.get("COGNITO_CLIENT_SECRET")
    if client_secret:
        auth_params["SECRET_HASH"] = _cognito_secret_hash(email, client_id, client_secret)

    logger.info("Cognito login attempt for %s", email)
    cognito = boto3.client("cognito-idp", region_name=region)
    try:
        resp = cognito.initiate_auth(
            AuthFlow="USER_PASSWORD_AUTH",
            ClientId=client_id,
            AuthParameters=auth_params,
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "unknown")
        detail = e.response.get("Error", {}).get("Message", str(e))
        logger.warning("Cognito login failed for %s: %s — %s", email, code, detail)
        # Map common Cognito error codes to clearer client-facing codes
        client_facing = {
            "NotAuthorizedException": "invalid_credentials",
            "UserNotFoundException": "invalid_credentials",
            "UserNotConfirmedException": "user_not_confirmed",
            "PasswordResetRequiredException": "password_reset_required",
            "TooManyRequestsException": "rate_limited",
            "InvalidParameterException": "invalid_request",
        }.get(code, "auth_error")
        status = 429 if code == "TooManyRequestsException" else 401
        raise CognitoLoginError(client_facing, detail, status=status) from e

    result = resp.get("AuthenticationResult")
    if not result:
        # When the user has additional challenges (MFA, new password), Cognito returns
        # ChallengeName instead of AuthenticationResult. We don't implement those flows.
        challenge = resp.get("ChallengeName", "unknown")
        logger.warning("Cognito returned challenge %s for %s (not handled)", challenge, email)
        raise CognitoLoginError(
            "challenge_required",
            f"Cognito requires the '{challenge}' challenge which is not supported by this endpoint. "
            "Use the browser-based OAuth flow instead.",
            status=401,
        )

    logger.info("Cognito login succeeded for %s", email)
    return result
