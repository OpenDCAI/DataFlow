"""MySQLConnector must decide "read or write" from the server, not the leading keyword.

``SQLExecutabilityFilter``/``SQLExecutionFilter`` deliberately keep CTE queries
(``filter_select_sql`` accepts ``select`` *and* ``with``), and the SQLite
connector supports them because it reads ``cursor.description``. The MySQL
connector instead sniffed ``sql.strip().upper().startswith(("SELECT", ...))``
*after* executing the statement, so every read that does not literally begin
with one of those four keywords was reported as a failed write.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymysql")

from dataflow.utils.text2sql.database_connector.mysql_connector import (  # noqa: E402
    MySQLConnector,
)


class _FakeCursor:
    """Minimal pymysql cursor: ``description`` is what the server sets for a read."""

    def __init__(self, *, returns_rows: bool, rows: list[dict] | None = None):
        self._returns_rows = returns_rows
        self._rows = rows if rows is not None else [{"a": 1}]
        self.description = None
        self.executed: list[str] = []
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self.description = [("a", 3, None, None, None, None, None)] if self._returns_rows else None
        return 1

    def fetchall(self):
        return self._rows

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _run(sql: str, *, returns_rows: bool):
    cursor = _FakeCursor(returns_rows=returns_rows)
    result = MySQLConnector().execute_query(_FakeConnection(cursor), sql)
    return result, cursor


@pytest.mark.cpu
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a FROM t",
        "  select a from t  ",
        "WITH recent AS (SELECT a FROM t) SELECT a FROM recent",
        "with recursive r(n) as (select 1) select n from r",
        "/* leading hint */ SELECT a FROM t",
        "-- why: cte is standard in Spider/BIRD\nSELECT a FROM t",
        "(SELECT a FROM t) UNION (SELECT a FROM t)",
        "SHOW TABLES",
        "EXPLAIN SELECT a FROM t",
    ],
)
def test_reads_are_accepted_however_the_statement_is_spelled(sql):
    result, cursor = _run(sql, returns_rows=True)

    assert result.success is True, result.error
    assert result.data == [{"a": 1}]
    assert cursor.closed is True


@pytest.mark.cpu
@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM t",
        "UPDATE t SET a = 1",
        "INSERT INTO t (a) VALUES (1)",
        "WITH gone AS (SELECT a FROM t) DELETE FROM t",
    ],
)
def test_writes_are_still_rejected(sql):
    result, _ = _run(sql, returns_rows=False)

    assert result.success is False
    assert "read-only" in result.error
