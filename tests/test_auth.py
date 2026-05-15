"""Unit tests for yugabytedb_mcp_server.auth — Cognito provider construction + JWT verification.

No network: OIDC discovery is mocked; JWT verification uses a self-signed RSA
key pair, never a real Cognito JWKS.

Run with: uv run pytest tests/test_auth.py
"""
import datetime
from unittest.mock import patch

import httpx
import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from yugabytedb_mcp_server.auth import create_auth_provider, _create_cognito


# ---------------------------------------------------------------------------
# Mocked Cognito OIDC discovery
# ---------------------------------------------------------------------------

FAKE_ISSUER = "https://cognito-idp.us-west-2.amazonaws.com/us-west-2_FAKE"
FAKE_JWKS_URI = f"{FAKE_ISSUER}/.well-known/jwks.json"
FAKE_OIDC_CONFIG = {
    "issuer": FAKE_ISSUER,
    "authorization_endpoint": f"{FAKE_ISSUER}/oauth2/authorize",
    "token_endpoint": f"{FAKE_ISSUER}/oauth2/token",
    "jwks_uri": FAKE_JWKS_URI,
    "response_types_supported": ["code"],
    "subject_types_supported": ["public"],
    "id_token_signing_alg_values_supported": ["RS256"],
    "scopes_supported": ["openid", "email", "profile"],
}

COGNITO_ENV = {
    "COGNITO_USER_POOL_ID": "us-west-2_FAKE",
    "COGNITO_AWS_REGION": "us-west-2",
    "COGNITO_CLIENT_ID": "fakeclient",
    "COGNITO_CLIENT_SECRET": "fakesecret",
    "MCP_BASE_URL": "http://localhost:8000",
}


def _mock_httpx_get(url, **kwargs):
    request = httpx.Request("GET", str(url))
    if "openid-configuration" in str(url):
        return httpx.Response(200, json=FAKE_OIDC_CONFIG, request=request)
    elif "jwks" in str(url):
        return httpx.Response(200, json={"keys": []}, request=request)
    return httpx.Response(404, request=request)


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------

class TestAuthFactory:
    def test_no_provider_returns_none(self):
        assert create_auth_provider(None) is None

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown auth provider"):
            create_auth_provider("not-a-real-provider")

    def test_cognito_missing_env_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(KeyError):
                _create_cognito()


class TestCognitoConstruction:
    @pytest.fixture(scope="class")
    def provider(self):
        with patch.dict("os.environ", COGNITO_ENV), \
             patch("httpx.get", side_effect=_mock_httpx_get):
            return _create_cognito()

    def test_returns_multiauth(self, provider):
        from fastmcp.server.auth.auth import MultiAuth
        assert isinstance(provider, MultiAuth)

    def test_inner_is_oidc_proxy(self, provider):
        from fastmcp.server.auth.oidc_proxy import OIDCProxy
        assert isinstance(provider.server, OIDCProxy)

    def test_default_scopes_set(self, provider):
        proxy = provider.server
        assert proxy._default_scope_str == "openid email profile"
        assert set(proxy.client_registration_options.default_scopes) == {
            "openid", "email", "profile",
        }


# ---------------------------------------------------------------------------
# JWT verification round-trip (self-signed; no real Cognito)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rsa_keypair():
    """Generate one RSA key pair for the test module."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return {"private_pem": private_pem, "public_pem": public_pem}


def _sign_jwt(claims: dict, private_pem: bytes) -> str:
    return pyjwt.encode(claims, private_pem, algorithm="RS256")


@pytest.mark.asyncio
async def test_jwt_verifier_accepts_valid_token(rsa_keypair):
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    issuer = FAKE_ISSUER
    audience = "test-audience"
    verifier = JWTVerifier(
        public_key=rsa_keypair["public_pem"].decode(),
        issuer=issuer,
        audience=audience,
        algorithm="RS256",
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    claims = {
        "iss": issuer,
        "sub": "test-user",
        "aud": audience,
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(claims, rsa_keypair["private_pem"])

    result = await verifier.verify_token(token)
    assert result is not None
    assert result.client_id == "test-user"


@pytest.mark.asyncio
async def test_jwt_verifier_rejects_expired_token(rsa_keypair):
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    verifier = JWTVerifier(
        public_key=rsa_keypair["public_pem"].decode(),
        issuer=FAKE_ISSUER,
        audience="aud",
        algorithm="RS256",
    )

    past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
    claims = {
        "iss": FAKE_ISSUER,
        "sub": "expired-user",
        "aud": "aud",
        "exp": past,
        "iat": past - datetime.timedelta(minutes=5),
    }
    token = _sign_jwt(claims, rsa_keypair["private_pem"])

    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_jwt_verifier_rejects_wrong_issuer(rsa_keypair):
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    verifier = JWTVerifier(
        public_key=rsa_keypair["public_pem"].decode(),
        issuer=FAKE_ISSUER,
        audience="aud",
        algorithm="RS256",
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    claims = {
        "iss": "https://attacker.example.com",
        "sub": "bad-user",
        "aud": "aud",
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(claims, rsa_keypair["private_pem"])

    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_jwt_verifier_rejects_tampered_signature(rsa_keypair):
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    verifier = JWTVerifier(
        public_key=rsa_keypair["public_pem"].decode(),
        issuer=FAKE_ISSUER,
        audience="aud",
        algorithm="RS256",
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    claims = {
        "iss": FAKE_ISSUER,
        "sub": "tampered",
        "aud": "aud",
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(claims, rsa_keypair["private_pem"])

    # Flip a character in the signature
    parts = token.split(".")
    sig = parts[2]
    parts[2] = sig[:-2] + ("A" if sig[-1] != "A" else "B") + sig[-1]
    tampered = ".".join(parts)

    result = await verifier.verify_token(tampered)
    assert result is None
