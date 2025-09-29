import threading
import logging
import time
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger(__name__)



class ClickHouseConnectionPool:

    class ClickHousePoolError(Exception):
        """Base exception for ClickHouse pool."""

    class TooManyConnections(ClickHousePoolError):
        """Raised when the pool is exhausted."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        min_connections: int = 5,
        max_connections: int = 20,
        health_check_interval: int = 30,
        connect_timeout: int = 10,
        **kwargs
    ):
        # Lazy import to avoid dependency if not used
        try:
            from clickhouse_driver import Client
            from clickhouse_driver.errors import Error as ClickHouseError
            # Store references for use in other methods
            self._Client = Client
            self._ClickHouseError = ClickHouseError
        except ImportError as e:
            raise ImportError("clickhouse-driver is required for ClickHouseConnectionPool, please install it via `pip install clickhouse-driver`") from e
        
        self._config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "connect_timeout": connect_timeout,
            **kwargs
        }
        self._min = min_connections
        self._max = max_connections
        self._health_check_interval = health_check_interval

        self._pool = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._closed = False
        
        # 修复：添加正确的连接计数
        self._in_use_connections = 0
        self._total_created = 0

        # 初始化最小连接
        for _ in range(self._min):
            self._pool.append(self._create_connection())

    def _create_connection(self):
        try:
            self._total_created += 1
            return self._Client(**self._config)
        except self._ClickHouseError as e:
            logger.error("Failed to create ClickHouse client: %s", e)
            raise

    def _is_connection_alive(self, client) -> bool:
        try:
            client.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _get_connection(self):
        with self._cond:
            while True:
                if self._closed:
                    raise ClickHousePoolError("Pool is closed.")

                if self._pool:
                    client = self._pool.pop()
                    if not self._is_connection_alive(client):
                        logger.warning("Dropping stale connection.")
                        self._in_use_connections -= 1
                        continue
                    self._in_use_connections += 1
                    return client

                if self._in_use_connections < self._max:
                    client = self._create_connection()
                    self._in_use_connections += 1
                    return client

                self._cond.wait(timeout=10)

    def _release_connection(self, client, close: bool = False):
        with self._cond:
            self._in_use_connections -= 1
            
            if close or self._closed:
                try:
                    client.disconnect()
                except Exception:
                    pass
            elif len(self._pool) < self._min:
                self._pool.append(client)
            else:
                try:
                    client.disconnect()
                except Exception:
                    pass
            self._cond.notify()

    def _in_use_count(self):
        # 修复：返回正确的连接数
        return self._in_use_connections

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        finally:
            if conn:
                self._release_connection(conn)

    def close(self):
        with self._cond:
            self._closed = True
            while self._pool:
                conn = self._pool.pop()
                try:
                    conn.disconnect()
                except Exception:
                    pass
            self._cond.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    # 新增：获取连接池状态
    def get_status(self):
        """获取连接池状态"""
        return {
            "pool_size": len(self._pool),
            "in_use": self._in_use_connections,
            "max_connections": self._max,
            "min_connections": self._min,
            "total_created": self._total_created,
            "closed": self._closed
        }