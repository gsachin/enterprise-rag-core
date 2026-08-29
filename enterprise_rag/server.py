# §6 MCP Server — lifespan-wired over the EngineConfig-built stack.
# OIDCJWTVerifier and security_from_token are the verbatim verified seed
# transcript; the module-scope SDK singletons of the seed are replaced by
# EngineConfig.build_stack() wiring (see config.py) with two auth modes:
#   oidc — OAuth2 bearer required, SecurityContext derived from the JWT claims
#   none — no token layer, every request runs as the configured default tenant
import asyncio
import json

import jwt
from pydantic import AnyHttpUrl

from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.context import Context

from enterprise_rag.security import SecurityContext                          # §1
from enterprise_rag.orchestrator import AgentContextRequest                  # §3
from enterprise_rag.config import EngineConfig


class OIDCJWTVerifier(TokenVerifier):
    """Validates RS256 OIDC access tokens against the issuer's JWKS and
    exposes the verified claims to tool handlers."""

    def __init__(self, issuer_url: str, audience: str):
        self._jwks = jwt.PyJWKClient(f"{issuer_url.rstrip('/')}/.well-known/jwks.json")
        self._issuer = issuer_url
        self._audience = audience

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            # PyJWKClient caches keys by kid; signature check + claim validation
            # are CPU-bound, so run them off the event loop.
            key = await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)
            claims = jwt.decode(
                token, key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience=self._audience,
                options={"require": ["exp"]},
            )
            if "sub" not in claims:
                return None
        except Exception:
            return None    # invalid signature / expired / wrong iss+aud -> 401 upstream
        return AccessToken(
            token=token,
            client_id=claims.get("azp") or claims.get("sub"),
            scopes=str(claims.get("scope", "")).split(),
            claims=claims,          # full OIDC claims reach the tool handlers
        )


def security_from_token(token: AccessToken) -> SecurityContext:
    """Claim mapping per §1.1 — deny-by-default: a missing tenant_id maps to "",
    which matches no chunk in either backend."""
    claims = token.claims or {}
    return SecurityContext(
        principal_id=str(claims["sub"]),
        tenant_id=str(claims.get("tenant_id", "")),
        roles=[str(r) for r in claims.get("roles", [])],
        departments=[str(d) for d in claims.get("departments", [])],
        clearance_level=int(claims.get("clearance_level", 0)),
        allowed_groups=[str(g) for g in claims.get("allowed_groups", [])],
    )


# ── Orchestrator seam ─────────────────────────────────────────────────────
# Wired by build_app() (or replaced wholesale in tests, cf. test_mcp_boot.py).
agent_orchestrator = None


def _set_orchestrator(orchestrator) -> None:
    global agent_orchestrator
    agent_orchestrator = orchestrator


async def _orchestrator() -> object:
    if agent_orchestrator is None:
        raise RuntimeError(
            "server not wired — call build_app() (or set server.agent_orchestrator)"
        )
    return agent_orchestrator


# ── Engine seam ────────────────────────────────────────────────────────────
# Wired by build_app() alongside the orchestrator (replaced in tests).
agent_engine = None


def _set_engine(engine) -> None:
    global agent_engine
    agent_engine = engine


async def _engine() -> object:
    if agent_engine is None:
        raise RuntimeError(
            "server not wired — call build_app() (or set server.agent_engine)"
        )
    return agent_engine


# ── Tool: OIDC mode ───────────────────────────────────────────────────────

async def execute_agent_context(
    resume_text: str,
    job_description: str,
    rubric_query: str,
    channel: str = "voice",
    ctx: Context | None = None,     # SDK-injected; union-typed Context params are
                                    # classified and excluded from the input schema
) -> str:
    """Executes low-latency agentic context orchestration for job agents and
    voice mock interviews. Requires OAuth2 bearer token with rag:retrieve."""
    token = get_access_token()      # SDK middleware contextvar — NOT context.session.*
    if token is None:               # e.g. stdio transport — refuses rather than fabricating identity
        raise ValueError("unauthenticated: this tool requires an OAuth2 bearer access token")
    security = security_from_token(token)

    req = AgentContextRequest(
        sec_ctx=security,
        resume_text=resume_text,
        job_description=job_description,
        conversation_history=[],
        rubric_query=rubric_query,
        channel=channel,
    )
    result = await (await _orchestrator()).execute_agent_context(req)
    return json.dumps(result, indent=2)


# ── Tool: generic retrieval (OIDC mode) ────────────────────────────────────

def _chunks_payload(chunks) -> dict:
    return {
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id,
                "tenant_id": c.tenant_id,
                "section_title": c.section_title,
                "content": c.content,
                "score": c.score,
                "required_clearance": c.required_clearance,
                "department": c.department,
            }
            for c in chunks
        ],
        "count": len(chunks),
        "hit_source": "retrieval",
    }


async def retrieve_context(
    query: str,
    top_k: int = 5,
    ctx: Context | None = None,     # SDK-injected; excluded from the input schema
) -> str:
    """Retrieves tenant-scoped context chunks for a question (hybrid dense +
    keyword legs). Requires OAuth2 bearer token with rag:retrieve."""
    token = get_access_token()
    if token is None:               # e.g. stdio transport — refuses rather than fabricating identity
        raise ValueError("unauthenticated: this tool requires an OAuth2 bearer access token")
    security = security_from_token(token)

    chunks = await (await _engine()).retrieve_parallel(query, security, top_k)
    return json.dumps(_chunks_payload(chunks), indent=2)


# ── Tool: generic retrieval (none auth mode) ───────────────────────────────

def _make_none_auth_retrieve_tool(config: EngineConfig):
    """Tool variant for ``none`` auth mode: no bearer token, every request
    runs as the configured default tenant. Same input schema as the OIDC
    tool."""

    async def retrieve_context_defaults(
        query: str,
        top_k: int = 5,
        ctx: Context | None = None,
    ) -> str:
        """Retrieves tenant-scoped context chunks for a question (hybrid dense
        + keyword legs). No-auth mode: runs as the configured default tenant."""
        chunks = await (await _engine()).retrieve_parallel(
            query, config.default_security(), top_k)
        return json.dumps(_chunks_payload(chunks), indent=2)

    return retrieve_context_defaults


# ── Tool: none auth mode ──────────────────────────────────────────────────

def _make_none_auth_tool(config: EngineConfig):
    """Tool variant for ``none`` auth mode: no bearer token, every request runs
    as the configured default tenant (RAG_CORE_DEFAULT_TENANT /
    RAG_CORE_DEFAULT_CLEARANCE). Same input schema as the OIDC tool."""

    async def execute_agent_context_defaults(
        resume_text: str,
        job_description: str,
        rubric_query: str,
        channel: str = "voice",
        ctx: Context | None = None,
    ) -> str:
        """Executes low-latency agentic context orchestration for job agents
        and voice mock interviews. No-auth mode: runs as the configured
        default tenant."""
        req = AgentContextRequest(
            sec_ctx=config.default_security(),
            resume_text=resume_text,
            job_description=job_description,
            conversation_history=[],
            rubric_query=rubric_query,
            channel=channel,
        )
        result = await (await _orchestrator()).execute_agent_context(req)
        return json.dumps(result, indent=2)

    return execute_agent_context_defaults


# ── Server construction ───────────────────────────────────────────────────

def build_mcp(config: EngineConfig | None = None) -> MCPServer:
    """Constructs the MCPServer for a configuration.

    ``oidc`` mode: OIDCJWTVerifier + AuthSettings (401 without a valid
    ``rag:retrieve`` bearer), tool derives SecurityContext from the claims.
    ``none`` mode: no token layer, tool runs as the default tenant."""
    config = config or EngineConfig.from_env()
    if config.auth_mode == "oidc":
        if not config.oidc_issuer or not config.oidc_audience:
            raise ValueError(
                "RAG_CORE_OIDC_ISSUER and RAG_CORE_OIDC_AUDIENCE are required "
                "for auth_mode=oidc"
            )
        mcp = MCPServer(
            name="enterprise-rag-core",
            token_verifier=OIDCJWTVerifier(
                issuer_url=config.oidc_issuer,
                audience=config.oidc_audience,
            ),
            auth=AuthSettings(
                issuer_url=AnyHttpUrl(config.oidc_issuer),
                resource_server_url=AnyHttpUrl(config.resource_server_url),
                required_scopes=["rag:retrieve"],
            ),
        )
        mcp.add_tool(execute_agent_context)
        mcp.add_tool(retrieve_context)
    else:
        mcp = MCPServer(name="enterprise-rag-core")
        mcp.add_tool(_make_none_auth_tool(config), name="execute_agent_context")
        mcp.add_tool(_make_none_auth_retrieve_tool(config), name="retrieve_context")
    return mcp


def build_app(config: EngineConfig | None = None):
    """Builds the stack, wires the module-level orchestrator, and returns the
    streamable-HTTP ASGI app. The stack is attached as ``app.state.stack`` —
    long-running servers should ``await app.state.stack.aclose()`` on shutdown
    to close Qdrant/Elasticsearch clients."""
    config = config or EngineConfig.from_env()
    stack = config.build_stack()
    _set_orchestrator(stack.orchestrator)
    _set_engine(stack.engine)
    mcp = build_mcp(config)
    app = mcp.streamable_http_app(streamable_http_path="/mcp")
    app.state.stack = stack
    return app


# ── Module-scope conveniences (seed-doc parity) ───────────────────────────
# MCPServer construction is cheap (no SDK clients); the engine stack is NOT
# built at import — call build_app() for that.

mcp = build_mcp()
app = mcp.streamable_http_app(streamable_http_path="/mcp")

#   Serve (streamable HTTP, the OAuth2 bearer transport):
#     .venv/Scripts/enterprise-rag-core serve --host 0.0.0.0 --port 8000
#   Dev-only stdio (NO auth layer — the OIDC tool refuses get_access_token()
#   None; use RAG_CORE_AUTH_MODE=none for stdio):
#     .venv/Scripts/enterprise-rag-core serve-stdio
