import json
from fastmcp.server.dependencies import get_access_token


def get_user_permissions() -> set[str]:
    """Extract permissions from the current access token.

    Normalizes across providers by checking multiple claim locations:
    - "permissions" (Auth0 RBAC)
    - "roles" (Azure Entra app roles)
    - "scope" / scopes (Cognito, generic OIDC)
    """
    token = get_access_token()
    print("Token:", json.dumps(token.model_dump(), indent=2, default=str))
    if not token:
        return set()

    result: set[str] = set()

    claims = token.claims or {}
    permissions = claims.get("permissions", [])
    if isinstance(permissions, list):
        result.update(permissions)

    roles = claims.get("roles", [])
    if isinstance(roles, list):
        result.update(roles)

    if token.scopes:
        result.update(token.scopes)

    scope_str = claims.get("scope", "")
    if isinstance(scope_str, str) and scope_str:
        result.update(scope_str.split())

    # Strip resource server prefixes (e.g., Cognito "mcp-api/read" → "read")
    unprefixed = {s.rsplit("/", 1)[-1] for s in result if "/" in s}
    result.update(unprefixed)

    return result


def require_permission(permission: str) -> None:
    """Raise PermissionError if the current user lacks the given permission."""
    perms = get_user_permissions()
    if permission not in perms:
        raise PermissionError(
            f"Access denied: requires '{permission}' permission. "
            f"User has: {perms or '{none}'}"
        )
