# tools.py
import json
import os
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import Context
from mem0 import Memory


def _get_conn(ctx: Context):
    """Helper to extract the psycopg2 connection from the MCP lifespan context."""
    return ctx.request_context.lifespan_context.conn


# ---------------------------------------------------------------------------
# Existing tools
# ---------------------------------------------------------------------------

def summarize_database(ctx: Context, schema: str = "public") -> List[Dict[str, Any]]:
    """
    Summarize the database: list tables with schema and row counts.
    """
    summary = []
    conn = _get_conn(ctx)
    with conn.cursor() as cur:
        try:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name
            """, (schema,))
            tables = [row[0] for row in cur.fetchall()]

            for table in tables:
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema, table,))
                schema_info = [{"column_name": col, "data_type": dtype} for col, dtype in cur.fetchall()]

                cur.execute(f"SELECT COUNT(*) FROM {schema}.\"{table}\"")
                row_count = cur.fetchone()[0]

                summary.append({
                    "table": table,
                    "row_count": row_count,
                    "schema": schema_info
                })

        except Exception as e:
            summary.append({"error": str(e)})

    return summary


def run_read_only_query(ctx: Context, query: str) -> str:
    """
    Run a read-only SQL query and return the results as JSON.
    """
    conn = _get_conn(ctx)
    with conn.cursor() as cur:
        try:
            cur.execute("BEGIN READ ONLY")
            cur.execute(query)
            rows = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]
            result = [dict(zip(column_names, row)) for row in rows]
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error executing query: {e}"
        finally:
            try:
                cur.execute("ROLLBACK")
            except Exception as e:
                return f"Couldn't ROLLBACK transaction: {e}"


# ---------------------------------------------------------------------------
# pg_dist_rag tools
# ---------------------------------------------------------------------------

# def add_document_source(ctx: Context, source_uri: str) -> str:
#     """
#     Add a document source to pg_dist_rag for RAG pipeline processing.

#     This registers a new source location (e.g. an S3 bucket path) so that
#     its documents can later be indexed.  The call inserts a row into
#     ``dist_rag.sources`` and returns the generated ``source_id``.

#     Args:
#         ctx: MCP context (injected automatically).
#         source_uri: URI of the document source (e.g. ``s3://bucket/path/``).

#     Returns:
#         JSON string with the created ``source_id``, or an error message.
#     """
#     conn = _get_conn(ctx)
#     with conn.cursor() as cur:
#         try:
#             cur.execute(
#                 """
#                 SELECT dist_rag.create_source(%s)
#                 """,
#                 (source_uri,),
#             )
#             result = cur.fetchone()
#             conn.commit()
#             source_id = str(result[0]) if result else None
#             return json.dumps({"source_id": source_id})
#         except Exception as e:
#             conn.rollback()
#             return f"Error adding document source: {e}"


# def init_vector_index(
#     ctx: Context,
#     index_name: str,
#     source_ids: str,
#     ai_provider: str = "OPENAI",
#     embedding_model: str = "text-embedding-3-large",
#     embedding_dimensions: int = 1536,
#     chunk_size: int = 1024,
#     chunk_overlap: int = 256,
# ) -> str:
#     """
#     Initialise a new vector index in pg_dist_rag.

#     Creates the index metadata and associates it with the given sources so it
#     is ready to be built.

#     Args:
#         ctx: MCP context (injected automatically).
#         index_name: Name for the new vector index.
#         source_ids: Comma-separated list of source UUIDs to include.
#         ai_provider: AI / embedding provider name (e.g. ``OPENAI``).
#         embedding_model: Embedding model name (e.g. ``text-embedding-3-large``).
#         embedding_dimensions: Dimensions of the embedding vector (e.g. ``1536``).
#         chunk_size: Number of characters per chunk (e.g. ``1024``).
#         chunk_overlap: Number of overlapping characters between chunks (e.g. ``256``).

#     Returns:
#         JSON string with the created ``index_id``, or an error message.
#     """
#     conn = _get_conn(ctx)
#     source_id_list = [s.strip() for s in source_ids.split(",") if s.strip()]
#     embedding_model_params = json.dumps({
#         "model": embedding_model,
#         "dimensions": embedding_dimensions,
#     })
#     chunk_params = json.dumps({
#         "chunk_size": chunk_size,
#         "chunk_overlap": chunk_overlap,
#     })
#     with conn.cursor() as cur:
#         try:
#             cur.execute(
#                 """
#                 SELECT dist_rag.init_vector_index(
#                     r_index_name := %s,
#                     r_sources := %s::UUID[],
#                     r_ai_provider := %s,
#                     r_embedding_model_params := %s,
#                     r_chunk_params := %s
#                 ) AS index_id
#                 """,
#                 (
#                     index_name,
#                     source_id_list,
#                     ai_provider,
#                     embedding_model_params,
#                     chunk_params,
#                 ),
#             )
#             result = cur.fetchone()
#             conn.commit()
#             index_id = str(result[0]) if result else None
#             return json.dumps({"index_id": index_id})
#         except Exception as e:
#             conn.rollback()
#             return f"Error initializing vector index: {e}"


# def trigger_index_build(ctx: Context, index_name: str) -> str:
#     """
#     Start building (or rebuilding) a vector index.

#     Triggers the pg_dist_rag preprocessing pipeline for the given index.
#     Workers will begin processing documents, generating chunks, and creating
#     embeddings.

#     Args:
#         ctx: MCP context (injected automatically).
#         index_name: Name of the vector index to build.

#     Returns:
#         JSON string confirming the build was triggered, or an error message.
#     """
#     conn = _get_conn(ctx)
#     with conn.cursor() as cur:
#         try:
#             cur.execute(
#                 """
#                 SELECT dist_rag.build_index(r_index_name := %s)
#                 """,
#                 (index_name,),
#             )
#             conn.commit()
#             return json.dumps({
#                 "status": "triggered",
#                 "index_name": index_name,
#                 "message": f"Index build triggered for '{index_name}'.",
#             })
#         except Exception as e:
#             conn.rollback()
#             return f"Error triggering index build: {e}"


def check_index_status(ctx: Context, index_name: str) -> str:
    """
    Check the status of a vector index build.

    Queries the ``dist_rag.vector_index_pipeline_details`` view to return
    per-document processing progress, then augments with aggregated stats
    from ``dist_rag.pipeline_stats``.

    Args:
        ctx: MCP context (injected automatically).
        index_name: Name of the vector index to inspect.

    Returns:
        JSON string with pipeline details and stats, or an error message.
    """
    conn = _get_conn(ctx)
    with conn.cursor() as cur:
        try:
            cur.execute("BEGIN READ ONLY")

            cur.execute(
                """
                SELECT index_name, source_uri, document_name,
                       document_status, chunks_processed
                FROM dist_rag.vector_index_pipeline_details
                WHERE index_name = %s
                """,
                (index_name,),
            )
            pipeline_rows = cur.fetchall()
            pipeline_cols = [desc[0] for desc in cur.description]
            pipeline_details = [dict(zip(pipeline_cols, row)) for row in pipeline_rows]

            cur.execute(
                """
                SELECT index_name, document_name, calls,
                       total_embeddings_persisted, completion_rate_percent
                FROM dist_rag.pipeline_stats
                WHERE index_name = %s
                """,
                (index_name,),
            )
            stats_rows = cur.fetchall()
            stats_cols = [desc[0] for desc in cur.description]
            stats = [dict(zip(stats_cols, row)) for row in stats_rows]

            return json.dumps({
                "index_name": index_name,
                "pipeline_details": pipeline_details,
                "pipeline_stats": stats,
            }, indent=2)
        except Exception as e:
            return f"Error checking index status: {e}"
        finally:
            try:
                cur.execute("ROLLBACK")
            except Exception:
                pass


def add_source_to_index(
    ctx: Context,
    index_name: str,
    source_id: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
) -> str:
    """
    Add an additional document source to an existing vector index.

    This is useful when you want to enrich an already-initialised index with
    documents from a new source location without re-creating the index.

    Args:
        ctx: MCP context (injected automatically).
        index_name: Name of the existing vector index.
        source_id: UUID of the source to attach.
        chunk_size: Number of characters per chunk (e.g. ``1024``).
        chunk_overlap: Number of overlapping characters between chunks (e.g. ``256``).

    Returns:
        JSON string confirming the source was added, or an error message.
    """
    conn = _get_conn(ctx)
    chunk_params = json.dumps({
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    })
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT dist_rag.add_source_to_index(
                    r_index_name := %s,
                    r_source_id := %s::UUID,
                    r_chunk_params := %s
                )
                """,
                (index_name, source_id, chunk_params),
            )
            conn.commit()
            return json.dumps({
                "status": "added",
                "index_name": index_name,
                "source_id": source_id,
                "message": f"Source '{source_id}' added to index '{index_name}'.",
            })
        except Exception as e:
            conn.rollback()
            return f"Error adding source to index: {e}"


def run_write_query(ctx: Context, query: str) -> str:
    """
    Execute a write SQL statement against the database.

    Runs the given SQL (e.g. INSERT, UPDATE, DELETE, CALL) inside a
    transaction.  On success the transaction is committed and the number of
    affected rows is returned.  On failure the transaction is rolled back
    automatically.

    Args:
        ctx: MCP context (injected automatically).
        query: The SQL statement to execute.

    Returns:
        JSON string with the number of rows affected, or an error message.
    """
    conn = _get_conn(ctx)
    with conn.cursor() as cur:
        try:
            cur.execute(query)
            conn.commit()
            rows_affected = cur.rowcount
            return json.dumps({
                "rows_affected": rows_affected,
            })
        except Exception as e:
            conn.rollback()
            return f"Error executing write query: {e}"


def trigger_knowledge_base_build(
    ctx: Context,
    source_uris: str,
    index_name: str,
    ai_provider: str = "OPENAI",
    embedding_model: str = "text-embedding-3-large",
    embedding_dimensions: int = 1536,
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
) -> str:
    """
    End-to-end RAG index creation: registers one or more document sources,
    initialises a vector index for them, and triggers the index build -- all
    in one call.

    This is a convenience wrapper that runs add_document_source (for each
    source), init_vector_index, and trigger_index_build in sequence.

    Args:
        ctx: MCP context (injected automatically).
        source_uris: Comma-separated URIs of document sources
            (e.g. ``s3://bucket/path1/,s3://bucket/path2/``).
        index_name: Name for the new vector index.
        ai_provider: AI / embedding provider name (e.g. ``OPENAI``).
        embedding_model: Embedding model name (e.g. ``text-embedding-3-large``).
        embedding_dimensions: Dimensions of the embedding vector (e.g. ``1536``).
        chunk_size: Number of characters per chunk (e.g. ``1024``).
        chunk_overlap: Number of overlapping characters between chunks (e.g. ``256``).

    Returns:
        JSON string with source_ids, index_id, and build status, or an error
        message indicating which step failed.
    """
    conn = _get_conn(ctx)
    uri_list = [u.strip() for u in source_uris.split(",") if u.strip()]

    # Step 1: register each document source
    source_ids = []
    for uri in uri_list:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT dist_rag.create_source(%s)",
                    (uri,),
                )
                result = cur.fetchone()
                conn.commit()
                source_ids.append(str(result[0]) if result else None)
            except Exception as e:
                conn.rollback()
                return (
                    f"Error adding document source '{uri}' "
                    f"(sources already created: {source_ids}): {e}"
                )

    # Step 2: init vector index with all sources
    embedding_model_params = json.dumps({
        "model": embedding_model,
        "dimensions": embedding_dimensions,
    })
    chunk_params = json.dumps({
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    })
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT dist_rag.init_vector_index(
                    r_index_name := %s,
                    r_sources := %s::UUID[],
                    r_ai_provider := %s,
                    r_embedding_model_params := %s,
                    r_chunk_params := %s
                ) AS index_id
                """,
                (
                    index_name,
                    source_ids,
                    ai_provider,
                    embedding_model_params,
                    chunk_params,
                ),
            )
            result = cur.fetchone()
            conn.commit()
            index_id = str(result[0]) if result else None
        except Exception as e:
            conn.rollback()
            return (
                f"Error initializing vector index "
                f"(sources created: {source_ids}): {e}"
            )

    # Step 3: trigger index build
    with conn.cursor() as cur:
        try:
            cur.execute(
                "SELECT dist_rag.build_index(r_index_name := %s)",
                (index_name,),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            return (
                f"Error triggering index build (sources {source_ids} and "
                f"index '{index_id}' were created): {e}"
            )

    return json.dumps({
        "source_ids": source_ids,
        "index_id": index_id,
        "index_name": index_name,
        "status": "build_triggered",
        "message": (
            f"{len(source_ids)} source(s) registered, index '{index_name}' "
            f"initialised, and build triggered."
        ),
    })


# ---------------------------------------------------------------------------
# mem0 tools
# ---------------------------------------------------------------------------

_mem0_instance: Optional[Memory] = None


def _get_mem0() -> Memory:
    """Lazily initialise the local Mem0 Memory instance from config.json.

    Uses pgvector as the vector store and Apache AGE as the graph store,
    both configured in config.json.  Requires OPENAI_API_KEY in the
    environment (used by the default embedder and LLM).
    """
    global _mem0_instance
    if _mem0_instance is None:
        config_path = os.environ.get(
            "MEM0_CONFIG_PATH",
            os.path.join(os.path.dirname(__file__), "..", "config.json"),
        )
        with open(config_path) as f:
            config = json.load(f)
        _mem0_instance = Memory.from_config(config["mem0"])
    return _mem0_instance


_transport_mode: str = "stdio"

_USER_ID_REQUIRED_ERROR = json.dumps({
    "error": "user_id_required",
    "detail": "In HTTP mode, user_id must be provided explicitly by the client.",
})


def set_transport_mode(mode: str) -> None:
    """Called by server.py at startup to set the transport mode."""
    global _transport_mode
    _transport_mode = mode


def _resolve_user_id(
    user_id: Optional[str],
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve user_id based on transport mode.

    In stdio mode: fall back to MEM0_USER_ID env var.
    In HTTP mode: user_id must be provided explicitly (no env var fallback).
    Returns None when another scoping param (agent_id / run_id) is set.
    """
    if user_id:
        return user_id
    if agent_id or run_id:
        return None
    if _transport_mode == "stdio":
        return os.environ.get("MEM0_USER_ID", "yugabytedb-mcp")
    return None


def add_memory(
    ctx: Context,
    text: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    metadata: Optional[str] = None,
    messages: Optional[str] = None,
    enable_graph: bool = True,
) -> str:
    """
    Store a memory using Mem0 (a fact, preference, or conversation snippet).

    Memories are stored in pgvector and, when graph is enabled, also
    extracted as entity-relationship triples in Apache AGE.

    Args:
        ctx: MCP context (injected automatically).
        text: Plain text content to store as a memory.
        user_id: User ID to scope the memory to. Defaults to the
            MEM0_USER_ID environment variable.
        agent_id: Optional agent identifier for scoping.
        app_id: Optional app identifier for scoping.
        run_id: Optional run identifier for scoping.
        metadata: Optional JSON string of arbitrary metadata to attach.
        messages: Optional JSON string of conversation history as a list of
            ``{"role": "...", "content": "..."}`` objects.  When provided,
            this is used instead of wrapping ``text`` as a single user message.
        enable_graph: Whether to also store graph relations via Apache AGE.
            Defaults to True.

    Returns:
        JSON string with the created memory details, or an error message.
    """
    m = _get_mem0()

    kwargs: Dict[str, Any] = {"enable_graph": enable_graph}

    uid = _resolve_user_id(user_id, agent_id, run_id)
    if uid:
        kwargs["user_id"] = uid
    if agent_id:
        kwargs["agent_id"] = agent_id
    if app_id:
        kwargs["app_id"] = app_id
    if run_id:
        kwargs["run_id"] = run_id

    if not uid and not agent_id and not run_id:
        return _USER_ID_REQUIRED_ERROR

    if metadata:
        try:
            kwargs["metadata"] = json.loads(metadata)
        except json.JSONDecodeError:
            return "Error: metadata must be a valid JSON string."

    if messages:
        try:
            conversation = json.loads(messages)
        except json.JSONDecodeError:
            return "Error: messages must be a valid JSON array of {role, content} objects."
    elif text:
        conversation = [{"role": "user", "content": text}]
    else:
        return json.dumps({
            "error": "messages_missing",
            "detail": "Provide either text or messages so Mem0 knows what to store.",
        })

    try:
        result = m.add(conversation, **kwargs)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"Error adding memory: {e}"


def delete_memory(ctx: Context, memory_id: str) -> str:
    """
    Delete a single memory by its ID.

    Args:
        ctx: MCP context (injected automatically).
        memory_id: The exact ID of the memory to delete.

    Returns:
        JSON string confirming deletion, or an error message.
    """
    m = _get_mem0()
    try:
        result = m.delete(memory_id)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"Error deleting memory: {e}"


def search_memories(
    ctx: Context,
    query: str,
    user_id: Optional[str] = None,
    limit: int = 10,
    enable_graph: bool = True,
) -> str:
    """
    Semantic search across stored memories and graph relations.

    Returns both vector-matched memories and, when graph is enabled,
    entity-relationship triples from Apache AGE.

    Args:
        ctx: MCP context (injected automatically).
        query: Natural language description of what to find.
        user_id: User ID to scope the search to. Defaults to the
            MEM0_USER_ID environment variable.
        limit: Maximum number of results to return.
        enable_graph: Whether to also return graph relations.
            Defaults to True.

    Returns:
        JSON string with matching memories (and relations when graph is
        enabled), or an error message.
    """
    m = _get_mem0()
    uid = _resolve_user_id(user_id)
    if not uid:
        return _USER_ID_REQUIRED_ERROR
    try:
        result = m.search(
            query=query,
            user_id=uid,
            limit=limit,
            enable_graph=enable_graph,
        )
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"Error searching memories: {e}"


def get_memory_by_id(ctx: Context, memory_id: str) -> str:
    """
    Retrieve a single memory by its exact ID.

    Args:
        ctx: MCP context (injected automatically).
        memory_id: The exact ID of the memory to fetch.

    Returns:
        JSON string with the memory details, or an error message.
    """
    m = _get_mem0()
    try:
        result = m.get(memory_id)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"Error fetching memory: {e}"


def get_memories(
    ctx: Context,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    List all memories for a given scope (user, agent, app, or run).

    Args:
        ctx: MCP context (injected automatically).
        user_id: User ID to list memories for. Defaults to the
            MEM0_USER_ID environment variable.
        agent_id: Optional agent identifier to filter by.
        app_id: Optional app identifier to filter by.
        run_id: Optional run identifier to filter by.

    Returns:
        JSON string with the list of memories, or an error message.
    """
    m = _get_mem0()
    kwargs: Dict[str, Any] = {}
    uid = _resolve_user_id(user_id, agent_id, run_id)
    if uid:
        kwargs["user_id"] = uid
    if agent_id:
        kwargs["agent_id"] = agent_id
    if app_id:
        kwargs["app_id"] = app_id
    if run_id:
        kwargs["run_id"] = run_id

    if not kwargs:
        return _USER_ID_REQUIRED_ERROR

    try:
        result = m.get_all(**kwargs)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"Error listing memories: {e}"


def update_memory(ctx: Context, memory_id: str, text: str) -> str:
    """
    Overwrite an existing memory's text.

    Args:
        ctx: MCP context (injected automatically).
        memory_id: The exact ID of the memory to update.
        text: The replacement text for the memory.

    Returns:
        JSON string with the updated memory details, or an error message.
    """
    m = _get_mem0()
    try:
        result = m.update(memory_id=memory_id, data=text)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"Error updating memory: {e}"


def delete_all_memories(
    ctx: Context,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    Delete all memories in the given scope.

    Provide at least one of user_id, agent_id, app_id, or run_id to
    define which memories to delete.

    Args:
        ctx: MCP context (injected automatically).
        user_id: Delete all memories for this user.
        agent_id: Delete all memories for this agent.
        app_id: Delete all memories for this app.
        run_id: Delete all memories for this run.

    Returns:
        JSON string confirming deletion, or an error message.
    """
    m = _get_mem0()
    kwargs: Dict[str, Any] = {}
    uid = _resolve_user_id(user_id, agent_id, run_id)
    if uid:
        kwargs["user_id"] = uid
    if agent_id:
        kwargs["agent_id"] = agent_id
    if app_id:
        kwargs["app_id"] = app_id
    if run_id:
        kwargs["run_id"] = run_id

    if not kwargs:
        return _USER_ID_REQUIRED_ERROR

    try:
        result = m.delete_all(**kwargs)
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"Error deleting all memories: {e}"
