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


# ---------------------------------------------------------------------------
# audience validation
# ---------------------------------------------------------------------------
# The pre-fix code passed no `audience=` to JWTVerifier, so a token minted
# for a different app client in the same Cognito user pool was accepted at
# /mcp. Fix: pass the expected audience (Cognito's app-client_id) to
# JWTVerifier. These tests exercise the raw JWTVerifier with `audience=`
# because that's the guarantee we rely on in _create_cognito.

@pytest.mark.asyncio
async def test_jwt_verifier_rejects_wrong_audience(rsa_keypair):
    """ token minted for another client (different `aud`) must be
    rejected once audience validation is enabled."""
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    verifier = JWTVerifier(
        public_key=rsa_keypair["public_pem"].decode(),
        issuer=FAKE_ISSUER,
        audience="expected-client-id",
        algorithm="RS256",
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    claims = {
        "iss": FAKE_ISSUER,
        "sub": "some-user",
        "aud": "OTHER-app-client",       # different client in same pool
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(claims, rsa_keypair["private_pem"])
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_jwt_verifier_accepts_matching_audience(rsa_keypair):
    """Regression: correct audience continues to pass."""
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    verifier = JWTVerifier(
        public_key=rsa_keypair["public_pem"].decode(),
        issuer=FAKE_ISSUER,
        audience="expected-client-id",
        algorithm="RS256",
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    claims = {
        "iss": FAKE_ISSUER,
        "sub": "some-user",
        "aud": "expected-client-id",
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(claims, rsa_keypair["private_pem"])
    result = await verifier.verify_token(token)
    assert result is not None


def test_cognito_provider_wires_client_id_as_expected_audience():
    """ _create_cognito must pass COGNITO_CLIENT_ID as the
    verifier's expected audience so tokens for other same-pool clients are
    rejected. The custom `_CognitoJWTVerifier` stores this on
    `_expected_audience` (the parent JWTVerifier's `audience` attribute is
    intentionally None because Cognito access tokens omit the `aud` claim
    and put the identifier in `client_id`; the subclass handles both)."""
    with patch.dict("os.environ", COGNITO_ENV, clear=False), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        provider = _create_cognito()
    verifier = provider.verifiers[0]
    assert verifier._expected_audience == COGNITO_ENV["COGNITO_CLIENT_ID"]
    # Parent's audience is disabled — our subclass handles it.
    assert getattr(verifier, "audience", None) is None


# ---------------------------------------------------------------------------
# token_use=access enforcement (Cognito, opt-in)
# ---------------------------------------------------------------------------
# _CognitoJWTVerifier rejects tokens where `token_use != "access"` when
# YB_MCP_REQUIRE_ACCESS_TOKEN=true. Default off for backward compat with
# email-in-ID-token deployments.

@pytest.mark.asyncio
async def test_id_token_accepted_when_require_access_off(rsa_keypair, caplog):
    """Default config: ID tokens are accepted; a WARNING is logged so
    operators see they should migrate."""
    with patch.dict("os.environ", {**COGNITO_ENV}, clear=False), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        # Ensure the toggle is off (default)
        with patch.dict("os.environ", {"YB_MCP_REQUIRE_ACCESS_TOKEN": "false"}):
            provider = _create_cognito()

    verifier = provider.verifiers[0]
    # Swap the verifier's JWKS-based key path for our test RSA key so we can
    # sign tokens locally. JWTVerifier accepts `public_key` at construction;
    # since we're testing the subclass logic, patch its key resolution.
    verifier._public_key = rsa_keypair["public_pem"].decode()
    verifier.public_key = rsa_keypair["public_pem"].decode()
    verifier.algorithm = "RS256"

    now = datetime.datetime.now(datetime.timezone.utc)
    id_token_claims = {
        "iss": FAKE_ISSUER,
        "sub": "user",
        "aud": COGNITO_ENV["COGNITO_CLIENT_ID"],
        "token_use": "id",                  # ID token, not access
        "email": "alice@example.com",
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(id_token_claims, rsa_keypair["private_pem"])

    with caplog.at_level("WARNING"):
        result = await verifier.verify_token(token)

    assert result is not None, "ID token accepted when require_access is off"
    assert any("ID token" in r.message for r in caplog.records), \
        "expected a WARNING about accepted ID token"


@pytest.mark.asyncio
async def test_id_token_rejected_by_default(rsa_keypair, caplog):
    """Post-review requirement: the new default rejects ID tokens without
    any opt-in. Verified by not touching any of the token_use env vars —
    default of ``YB_MCP_REQUIRE_ACCESS_TOKEN`` is now True."""
    # Explicitly remove any env leakage from other tests that might have
    # set the toggle in the shared os.environ.
    clean_env = {**COGNITO_ENV}
    with patch.dict("os.environ", clean_env, clear=True), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        provider = _create_cognito()

    verifier = provider.verifiers[0]
    assert verifier._require_access_token is True, (
        "default should reject ID tokens (was False before )"
    )
    verifier._public_key = rsa_keypair["public_pem"].decode()
    verifier.public_key = rsa_keypair["public_pem"].decode()
    verifier.algorithm = "RS256"

    now = datetime.datetime.now(datetime.timezone.utc)
    id_token_claims = {
        "iss": FAKE_ISSUER,
        "sub": "user",
        "aud": COGNITO_ENV["COGNITO_CLIENT_ID"],
        "token_use": "id",
        "email": "alice@example.com",
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(id_token_claims, rsa_keypair["private_pem"])
    with caplog.at_level("WARNING"):
        result = await verifier.verify_token(token)
    assert result is None, "default config should reject ID token"


@pytest.mark.asyncio
async def test_id_token_accepted_when_legacy_flag_set(rsa_keypair, caplog):
    """ compat flag: ``YB_MCP_LEGACY_ACCEPT_ID_TOKENS=true`` restores
    the old behavior (accept ID tokens with a warning)."""
    env = {**COGNITO_ENV, "YB_MCP_LEGACY_ACCEPT_ID_TOKENS": "true"}
    with patch.dict("os.environ", env, clear=True), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        provider = _create_cognito()

    verifier = provider.verifiers[0]
    assert verifier._require_access_token is False, (
        "legacy flag should flip require_access_token back to False"
    )
    verifier._public_key = rsa_keypair["public_pem"].decode()
    verifier.public_key = rsa_keypair["public_pem"].decode()
    verifier.algorithm = "RS256"

    now = datetime.datetime.now(datetime.timezone.utc)
    id_token_claims = {
        "iss": FAKE_ISSUER,
        "sub": "user",
        "aud": COGNITO_ENV["COGNITO_CLIENT_ID"],
        "token_use": "id",
        "email": "alice@example.com",
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(id_token_claims, rsa_keypair["private_pem"])
    with caplog.at_level("WARNING"):
        result = await verifier.verify_token(token)
    assert result is not None, "legacy flag should accept ID tokens"


@pytest.mark.asyncio
async def test_explicit_require_access_false_overrides_legacy_default(rsa_keypair):
    """ an explicit ``YB_MCP_REQUIRE_ACCESS_TOKEN=false`` continues
    to opt in to ID tokens even without the compat flag — the explicit
    value wins over the new default."""
    env = {**COGNITO_ENV, "YB_MCP_REQUIRE_ACCESS_TOKEN": "false"}
    with patch.dict("os.environ", env, clear=True), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        provider = _create_cognito()

    verifier = provider.verifiers[0]
    assert verifier._require_access_token is False


@pytest.mark.asyncio
async def test_id_token_rejected_when_require_access_on(rsa_keypair, caplog):
    """`YB_MCP_REQUIRE_ACCESS_TOKEN=true` rejects ID tokens outright."""
    with patch.dict("os.environ", {**COGNITO_ENV, "YB_MCP_REQUIRE_ACCESS_TOKEN": "true"}), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        provider = _create_cognito()

    verifier = provider.verifiers[0]
    verifier._public_key = rsa_keypair["public_pem"].decode()
    verifier.public_key = rsa_keypair["public_pem"].decode()
    verifier.algorithm = "RS256"

    now = datetime.datetime.now(datetime.timezone.utc)
    id_token_claims = {
        "iss": FAKE_ISSUER,
        "sub": "user",
        "aud": COGNITO_ENV["COGNITO_CLIENT_ID"],
        "token_use": "id",
        "email": "alice@example.com",
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(id_token_claims, rsa_keypair["private_pem"])

    with caplog.at_level("WARNING"):
        result = await verifier.verify_token(token)

    assert result is None, "ID token should be rejected when require_access is on"
    assert any(
        "token_use" in r.message and "id" in r.message.lower()
        for r in caplog.records
    ), "expected a WARNING mentioning token_use=id"


@pytest.mark.asyncio
async def test_access_token_accepted_when_require_access_on(rsa_keypair):
    """Access tokens with the Cognito-native shape (no `aud`; identity in
    `client_id`) must pass with require_access=true. This is the real
    shape Cognito mints — see the module docstring on
    _CognitoJWTVerifier for why."""
    with patch.dict("os.environ", {**COGNITO_ENV, "YB_MCP_REQUIRE_ACCESS_TOKEN": "true"}), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        provider = _create_cognito()

    verifier = provider.verifiers[0]
    verifier._public_key = rsa_keypair["public_pem"].decode()
    verifier.public_key = rsa_keypair["public_pem"].decode()
    verifier.algorithm = "RS256"

    now = datetime.datetime.now(datetime.timezone.utc)
    # Real-Cognito access token: no `aud`, identity in `client_id`.
    access_token_claims = {
        "iss": FAKE_ISSUER,
        "sub": "user",
        "client_id": COGNITO_ENV["COGNITO_CLIENT_ID"],
        "token_use": "access",
        "cognito:groups": ["writer", "reader"],
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(access_token_claims, rsa_keypair["private_pem"])

    result = await verifier.verify_token(token)
    assert result is not None, "Cognito-shape access token (no aud, has client_id) was rejected"


@pytest.mark.asyncio
async def test_access_token_wrong_client_id_rejected(rsa_keypair, caplog):
    """ an access token minted for a different app client in the
    same pool must be rejected. Real Cognito tokens carry the identifier
    in `client_id`, not `aud`, so the check must inspect both."""
    with patch.dict("os.environ", COGNITO_ENV, clear=False), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        provider = _create_cognito()
    verifier = provider.verifiers[0]
    verifier._public_key = rsa_keypair["public_pem"].decode()
    verifier.public_key = rsa_keypair["public_pem"].decode()
    verifier.algorithm = "RS256"

    now = datetime.datetime.now(datetime.timezone.utc)
    # Wrong client_id — pretends to be another app client in the pool.
    access_token_claims = {
        "iss": FAKE_ISSUER,
        "sub": "user",
        "client_id": "another-client-in-same-pool",
        "token_use": "access",
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(access_token_claims, rsa_keypair["private_pem"])

    with caplog.at_level("WARNING"):
        result = await verifier.verify_token(token)

    assert result is None, "wrong-client_id access token should be rejected"
    assert any("audience mismatch" in r.message for r in caplog.records), \
        "expected an audience-mismatch WARNING"


@pytest.mark.asyncio
async def test_refresh_token_rejected_when_require_access_on(rsa_keypair):
    """token_use=refresh presented as a bearer is also rejected."""
    with patch.dict("os.environ", {**COGNITO_ENV, "YB_MCP_REQUIRE_ACCESS_TOKEN": "true"}), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        provider = _create_cognito()

    verifier = provider.verifiers[0]
    verifier._public_key = rsa_keypair["public_pem"].decode()
    verifier.public_key = rsa_keypair["public_pem"].decode()
    verifier.algorithm = "RS256"

    now = datetime.datetime.now(datetime.timezone.utc)
    refresh_claims = {
        "iss": FAKE_ISSUER,
        "sub": "user",
        "aud": COGNITO_ENV["COGNITO_CLIENT_ID"],
        "token_use": "refresh",
        "exp": now + datetime.timedelta(minutes=5),
        "iat": now,
    }
    token = _sign_jwt(refresh_claims, rsa_keypair["private_pem"])
    result = await verifier.verify_token(token)
    assert result is None


# ---------------------------------------------------------------------------
# OIDC path forwards OIDC_AUDIENCE to the verifier
# ---------------------------------------------------------------------------

def test_oidc_provider_wires_audience_env_to_verifier():
    """_create_oidc must pass OIDC_AUDIENCE (already read for OIDCProxy)
    to JWTVerifier as well, so signature+issuer isn't the only check."""
    from yugabytedb_mcp_server.auth import _create_oidc

    oidc_env = {
        "OIDC_CONFIG_URL": f"{FAKE_ISSUER}/.well-known/openid-configuration",
        "OIDC_CLIENT_ID": "oidc-client",
        "OIDC_CLIENT_SECRET": "oidc-secret",
        "OIDC_AUDIENCE": "expected-oidc-audience",
        "MCP_BASE_URL": "http://localhost:8000",
    }
    with patch.dict("os.environ", oidc_env, clear=False), \
         patch("httpx.get", side_effect=_mock_httpx_get):
        provider = _create_oidc()

    verifier = provider.verifiers[0]
    assert getattr(verifier, "audience", None) == "expected-oidc-audience"


def test_oidc_provider_warns_when_audience_missing(caplog):
    """No OIDC_AUDIENCE → a startup WARNING logs that only signature +
    issuer are checked; provider still constructs (backward compat)."""
    from yugabytedb_mcp_server.auth import _create_oidc

    oidc_env = {
        "OIDC_CONFIG_URL": f"{FAKE_ISSUER}/.well-known/openid-configuration",
        "OIDC_CLIENT_ID": "oidc-client",
        "OIDC_CLIENT_SECRET": "oidc-secret",
        # OIDC_AUDIENCE deliberately absent
        "MCP_BASE_URL": "http://localhost:8000",
    }
    with patch.dict("os.environ", oidc_env, clear=True), \
         patch("httpx.get", side_effect=_mock_httpx_get), \
         caplog.at_level("WARNING"):
        provider = _create_oidc()

    verifier = provider.verifiers[0]
    assert getattr(verifier, "audience", None) is None
    assert any(
        "OIDC_AUDIENCE" in r.message for r in caplog.records
    ), "expected a WARNING mentioning OIDC_AUDIENCE"
