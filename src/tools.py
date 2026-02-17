# tools.py
import json
from typing import List, Dict, Any

from mcp.server.fastmcp import Context


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

def add_document_source(ctx: Context, source_uri: str) -> str:
    """
    Add a document source to pg_dist_rag for RAG pipeline processing.

    This registers a new source location (e.g. an S3 bucket path) so that
    its documents can later be indexed.  The call inserts a row into
    ``dist_rag.sources`` and returns the generated ``source_id``.

    Args:
        ctx: MCP context (injected automatically).
        source_uri: URI of the document source (e.g. ``s3://bucket/path/``).

    Returns:
        JSON string with the created ``source_id``, or an error message.
    """
    conn = _get_conn(ctx)
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT dist_rag.create_source(%s)
                """,
                (source_uri,),
            )
            result = cur.fetchone()
            conn.commit()
            source_id = str(result[0]) if result else None
            return json.dumps({"source_id": source_id})
        except Exception as e:
            conn.rollback()
            return f"Error adding document source: {e}"


def init_vector_index(
    ctx: Context,
    index_name: str,
    source_ids: str,
    ai_provider: str = "OPENAI",
    embedding_model: str = "text-embedding-3-large",
    embedding_dimensions: int = 1536,
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
) -> str:
    """
    Initialise a new vector index in pg_dist_rag.

    Creates the index metadata and associates it with the given sources so it
    is ready to be built.

    Args:
        ctx: MCP context (injected automatically).
        index_name: Name for the new vector index.
        source_ids: Comma-separated list of source UUIDs to include.
        ai_provider: AI / embedding provider name (e.g. ``OPENAI``).
        embedding_model: Embedding model name (e.g. ``text-embedding-3-large``).
        embedding_dimensions: Dimensions of the embedding vector (e.g. ``1536``).
        chunk_size: Number of characters per chunk (e.g. ``1024``).
        chunk_overlap: Number of overlapping characters between chunks (e.g. ``256``).

    Returns:
        JSON string with the created ``index_id``, or an error message.
    """
    conn = _get_conn(ctx)
    source_id_list = [s.strip() for s in source_ids.split(",") if s.strip()]
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
                    source_id_list,
                    ai_provider,
                    embedding_model_params,
                    chunk_params,
                ),
            )
            result = cur.fetchone()
            conn.commit()
            index_id = str(result[0]) if result else None
            return json.dumps({"index_id": index_id})
        except Exception as e:
            conn.rollback()
            return f"Error initializing vector index: {e}"


def trigger_index_build(ctx: Context, index_name: str) -> str:
    """
    Start building (or rebuilding) a vector index.

    Triggers the pg_dist_rag preprocessing pipeline for the given index.
    Workers will begin processing documents, generating chunks, and creating
    embeddings.

    Args:
        ctx: MCP context (injected automatically).
        index_name: Name of the vector index to build.

    Returns:
        JSON string confirming the build was triggered, or an error message.
    """
    conn = _get_conn(ctx)
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT dist_rag.build_index(r_index_name := %s)
                """,
                (index_name,),
            )
            conn.commit()
            return json.dumps({
                "status": "triggered",
                "index_name": index_name,
                "message": f"Index build triggered for '{index_name}'.",
            })
        except Exception as e:
            conn.rollback()
            return f"Error triggering index build: {e}"


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
