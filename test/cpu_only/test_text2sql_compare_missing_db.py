import sqlite3

import pytest

from dataflow.utils.text2sql.database_manager import DatabaseManager


@pytest.fixture
def database_manager(tmp_path):
    db_dir = tmp_path / "concert"
    db_dir.mkdir()
    conn = sqlite3.connect(db_dir / "concert.sqlite")
    conn.execute("CREATE TABLE singer (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO singer VALUES (1, 'a'), (2, 'b')")
    conn.commit()
    conn.close()
    return DatabaseManager(db_type="sqlite", config={"root_path": str(tmp_path)})


@pytest.mark.cpu
def test_missing_database_only_fails_its_own_comparison(database_manager):
    results = database_manager.batch_compare_queries([
        ("concert", "SELECT name FROM singer", "SELECT name FROM singer"),
        ("missing_db", "SELECT 1", "SELECT 1"),
        ("concert", "SELECT name FROM singer", "SELECT name FROM singer WHERE id = 1"),
    ])

    assert len(results) == 3
    assert results[0]["equal"] is True
    assert results[1]["equal"] is False
    assert results[1]["differences"] == ["Database 'missing_db' not found"]
    assert results[2]["equal"] is False
    assert results[2]["result1_success"] and results[2]["result2_success"]
