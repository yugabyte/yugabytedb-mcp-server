import os

SUPPORTED_PROVIDERS = ["auth0", "azure", "cognito", "oidc"]


def create_auth_provider(name: str):
    """Factory that returns a configured auth provider instance based on name."""
    if name == "auth0":
        return _create_auth0()
    elif name == "azure":
        return _create_azure()
    elif name == "cognito":
        return _create_cognito()
    elif name == "oidc":
        return _create_oidc()
    else:
        raise ValueError(
            f"Unknown auth provider: {name!r}. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )


def _create_auth0():
    from fastmcp.server.auth.providers.auth0 import Auth0Provider

    return Auth0Provider(
        config_url=os.environ["AUTH0_CONFIG_URL"],
        client_id=os.environ["AUTH0_CLIENT_ID"],
        client_secret=os.environ["AUTH0_CLIENT_SECRET"],
        audience=os.environ["AUTH0_AUDIENCE"],
        base_url=os.environ.get("MCP_BASE_URL", "http://localhost:8000"),
    )


def _create_azure():
    from fastmcp.server.auth.providers.azure import AzureProvider

    return AzureProvider(
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
        tenant_id=os.environ["AZURE_TENANT_ID"],
        base_url=os.environ.get("MCP_BASE_URL", "http://localhost:8000"),
    )


def _create_cognito():
    from fastmcp.server.auth.oidc_proxy import OIDCProxy

    pool_id = os.environ["COGNITO_USER_POOL_ID"]
    region = os.environ["COGNITO_AWS_REGION"]
    config_url = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/openid-configuration"

    resource_id = os.environ.get("COGNITO_RESOURCE_SERVER_ID", "https://mcp-api")
    scopes = f"openid {resource_id}/read {resource_id}/write {resource_id}/admin"

    return OIDCProxy(
        config_url=config_url,
        client_id=os.environ["COGNITO_CLIENT_ID"],
        client_secret=os.environ["COGNITO_CLIENT_SECRET"],
        base_url=os.environ.get("MCP_BASE_URL", "http://localhost:8000"),
        token_endpoint_auth_method="client_secret_post",
        extra_authorize_params={"scope": scopes, "resource": resource_id},
    )


def _create_oidc():
    from fastmcp.server.auth.oidc_proxy import OIDCProxy

    return OIDCProxy(
        config_url=os.environ["OIDC_CONFIG_URL"],
        client_id=os.environ["OIDC_CLIENT_ID"],
        client_secret=os.environ["OIDC_CLIENT_SECRET"],
        audience=os.environ.get("OIDC_AUDIENCE"),
        base_url=os.environ.get("MCP_BASE_URL", "http://localhost:8000"),
    )
