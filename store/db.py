"""PostgreSQL connection (127.0.0.1:5433) and schema setup."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

DSN = os.environ.get("DQA_DSN", "host=127.0.0.1 port=5433 user=dqa dbname=dqa")
SCHEMA = Path(__file__).with_name("schema.sql")


@contextmanager
def connect(autocommit: bool = False):
    conn = psycopg.connect(DSN, autocommit=autocommit)
    try:
        register_vector(conn)
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def init_schema():
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(SCHEMA.read_text("utf-8"))


def reset():
    """Drop everything (cold test)."""
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    init_schema()


if __name__ == "__main__":
    import sys
    reset() if "--reset" in sys.argv else init_schema()
    print("schema ready")
