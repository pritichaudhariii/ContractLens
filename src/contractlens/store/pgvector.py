"""Postgres + pgvector store: HNSW cosine search and GIN-indexed full-text search."""

from __future__ import annotations

import json
from importlib import resources

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from contractlens.models import Chunk, Document
from contractlens.store.base import DocumentStore
from contractlens.store.memory import tokenize


class PgVectorStore(DocumentStore):
    name = "pgvector"

    def __init__(self, database_url: str, *, dimension: int, min_size: int = 1, max_size: int = 8) -> None:
        self.dimension = dimension
        # The vector type must exist before register_vector() can run on pooled connections.
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.pool = ConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            configure=self._configure,
            open=True,
        )
        try:
            self.ensure_schema()
        except Exception:
            self.pool.close()
            raise

    @staticmethod
    def _configure(conn: psycopg.Connection) -> None:
        register_vector(conn)

    # ---- schema -------------------------------------------------------------------

    def ensure_schema(self) -> None:
        sql = resources.files("contractlens.store").joinpath("schema.sql").read_text(encoding="utf-8")
        with self.pool.connection() as conn:
            conn.execute(sql.replace("{dim}", str(self.dimension)))
            row = conn.execute("SELECT value FROM contractlens_meta WHERE key = 'embedding_dimension'").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO contractlens_meta (key, value) VALUES ('embedding_dimension', %s)",
                    (str(self.dimension),),
                )
            elif int(row["value"]) != self.dimension:
                raise RuntimeError(
                    f"Database was created for {row['value']}-dimensional embeddings but the configured "
                    f"embedder produces {self.dimension}. Drop the chunks table or switch provider."
                )
            conn.commit()

    # ---- writes --------------------------------------------------------------------

    def upsert_document(self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document.id,))
            conn.execute(
                """
                INSERT INTO documents (id, title, doc_type, source, metadata, created_at, chunk_count)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title, doc_type = EXCLUDED.doc_type, source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata, created_at = EXCLUDED.created_at, chunk_count = EXCLUDED.chunk_count
                """,
                (
                    document.id,
                    document.title,
                    document.doc_type,
                    document.source,
                    json.dumps(document.metadata),
                    document.created_at,
                    len(chunks),
                ),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (id, document_id, ordinal, section, page, text, char_start, char_end, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (c.id, c.document_id, c.ordinal, c.section, c.page, c.text, c.char_start, c.char_end, emb)
                        for c, emb in zip(chunks, embeddings, strict=True)
                    ],
                )
            conn.commit()

    def delete_document(self, document_id: str) -> bool:
        with self.pool.connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            conn.commit()
            return cur.rowcount > 0

    # ---- reads -----------------------------------------------------------------------

    def list_documents(self) -> list[Document]:
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY title").fetchall()
        return [self._document(r) for r in rows]

    def get_document(self, document_id: str) -> Document | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = %s", (document_id,)).fetchone()
        return self._document(row) if row else None

    def count_chunks(self) -> int:
        with self.pool.connection() as conn:
            return int(conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"])

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT c.*, d.title AS document_title, d.doc_type FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.id = ANY(%s)",
                (chunk_ids,),
            ).fetchall()
        by_id = {r["id"]: self._chunk(r) for r in rows}
        return [by_id[c] for c in chunk_ids if c in by_id]

    def document_chunks(self, document_id: str) -> list[Chunk]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT c.*, d.title AS document_title, d.doc_type FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.document_id = %s ORDER BY c.ordinal",
                (document_id,),
            ).fetchall()
        return [self._chunk(r) for r in rows]

    def vector_search(
        self,
        query_embedding: list[float],
        k: int,
        *,
        doc_types: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        from pgvector import Vector

        params: list[object] = [Vector(query_embedding)]
        clauses: list[str] = []
        if doc_types:
            clauses.append("d.doc_type = ANY(%s)")
            params.append(doc_types)
        if document_ids:
            clauses.append("c.document_id = ANY(%s)")
            params.append(document_ids)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(k)
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, d.title AS document_title, d.doc_type,
                       1 - (c.embedding <=> %s) AS score
                FROM chunks c JOIN documents d ON d.id = c.document_id
                {where}
                ORDER BY score DESC
                LIMIT %s
                """,
                params,
            ).fetchall()
        return [(self._chunk(r), float(r["score"])) for r in rows]

    def keyword_search(
        self, query: str, k: int, *, doc_types: list[str] | None = None, document_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Full-text search in two passes: the strict `websearch_to_tsquery` form first (all terms
        must match), then, if that leaves room, an OR-of-terms query so partial matches still
        rank. Both use the GIN-indexed generated `tsv` column and `ts_rank_cd`."""
        filters = ""
        filter_params: list[object] = []
        if doc_types:
            filters += " AND d.doc_type = ANY(%s)"
            filter_params.append(doc_types)
        if document_ids:
            filters += " AND c.document_id = ANY(%s)"
            filter_params.append(document_ids)

        results: list[tuple[Chunk, float]] = []
        seen: set[str] = set()
        with self.pool.connection() as conn:
            for tsquery, arg in (
                ("websearch_to_tsquery('english', %s)", query),
                ("to_tsquery('english', %s)", _or_query(query)),
            ):
                if not arg or len(results) >= k:
                    continue
                rows = conn.execute(
                    f"""
                    SELECT c.*, d.title AS document_title, d.doc_type, ts_rank_cd(c.tsv, {tsquery}) AS score
                    FROM chunks c JOIN documents d ON d.id = c.document_id
                    WHERE c.tsv @@ {tsquery}{filters}
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    [arg, arg, *filter_params, k],
                ).fetchall()
                for r in rows:
                    if r["id"] not in seen:
                        seen.add(r["id"])
                        results.append((self._chunk(r), float(r["score"])))
        return results[:k]

    def close(self) -> None:
        self.pool.close()

    # ---- mapping -----------------------------------------------------------------------

    @staticmethod
    def _document(row: dict) -> Document:
        return Document(
            id=row["id"],
            title=row["title"],
            doc_type=row["doc_type"],
            source=row["source"],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
            chunk_count=row["chunk_count"],
        )

    @staticmethod
    def _chunk(row: dict) -> Chunk:
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            document_title=row["document_title"],
            doc_type=row["doc_type"],
            ordinal=row["ordinal"],
            section=row["section"],
            page=row["page"],
            text=row["text"],
            char_start=row["char_start"],
            char_end=row["char_end"],
        )


def _or_query(query: str) -> str:
    """'liability cap' -> 'liability | cap' for to_tsquery; tokens are alphanumeric so no operator injection."""
    terms = [t for t in dict.fromkeys(tokenize(query)) if len(t) > 1]
    return " | ".join(terms)
