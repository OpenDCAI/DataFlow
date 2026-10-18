"""Regression tests for ``DatabaseManager.batch_compare_queries``.

A triple pointing at an unregistered database must only fail that single triple;
it must not make every other comparison in the batch fail (see issue #556).
"""
import os
import sqlite3

import pytest

from dataflow.utils.text2sql.database_manager import DatabaseManager


@pytest.fixture
def manager(tmp_path):
    """A manager with one SQLite database named ``concert``."""
    db_dir = tmp_path / "concert"
    db_dir.mkdir()
    conn = sqlite3.connect(db_dir / "concert.sqlite")
    conn.execute("CREATE TABLE singer (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO singer VALUES (1, 'a'), (2, 'b')")
    conn.commit()
    conn.close()

    return DatabaseManager(db_type="sqlite", config={"root_path": str(tmp_path)})


@pytest.mark.cpu
def test_missing_db_only_fails_its_own_triple(manager):
    results = manager.batch_compare_queries([
        ("concert", "SELECT name FROM singer", "SELECT name FROM singer"),
        ("missing_db", "SELECT 1", "SELECT 1"),
    ])

    assert len(results) == 2
    # The valid triple is still compared and equal.
    assert results[0]["equal"] is True
    assert results[0]["differences"] == []
    # Only the triple whose database is missing carries the error.
    assert results[1]["equal"] is False
    assert results[1]["differences"] == ["Database 'missing_db' not found"]


@pytest.mark.cpu
def test_missing_db_isolated_regardless_of_order(manager):
    """Output order must follow input order even when the bad row comes first."""
    results = manager.batch_compare_queries([
        ("missing_db", "SELECT 1", "SELECT 1"),
        ("concert", "SELECT name FROM singer", "SELECT name FROM singer"),
    ])

    assert results[0]["equal"] is False
    assert results[0]["differences"] == ["Database 'missing_db' not found"]
    assert results[1]["equal"] is True
    assert results[1]["differences"] == []


@pytest.mark.cpu
def test_each_missing_db_gets_its_own_error(manager):
    results = manager.batch_compare_queries([
        ("db_a", "SELECT 1", "SELECT 1"),
        ("db_b", "SELECT 1", "SELECT 1"),
    ])

    assert results[0]["differences"] == ["Database 'db_a' not found"]
    assert results[1]["differences"] == ["Database 'db_b' not found"]


@pytest.mark.cpu
def test_valid_but_unequal_queries_are_still_compared(manager):
    """The isolation change must not break the normal comparison path."""
    results = manager.batch_compare_queries([
        ("concert", "SELECT name FROM singer", "SELECT id FROM singer"),
    ])

    assert results[0]["equal"] is False
    assert results[0]["differences"]  # a column mismatch is reported


@pytest.mark.cpu
def test_empty_batch_returns_empty_list(manager):
    assert manager.batch_compare_queries([]) == []
