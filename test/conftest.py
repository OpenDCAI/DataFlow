import os
import pytest

@pytest.fixture(autouse=True, scope="session")  # 自动应用，无需标记
def set_test_dir():
    # 强制切换工作目录到 /tests
    tests_dir = os.path.join(os.path.dirname(__file__))
    os.chdir(tests_dir)

import threading, atexit

def show_threads():
    print("Alive threads at exit:", threading.enumerate())

atexit.register(show_threads)

# conftest.py
import threading
import time
import pytest
import requests
from werkzeug.serving import make_server
from dummy_server import create_app

class ServerThread(threading.Thread):
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.app = create_app()
        # ✅ 关键：threaded=True
        self.httpd = make_server(self.host, self.port, self.app, threaded=True)
        self.port = self.httpd.server_port

    def run(self):
        self.httpd.serve_forever()

    def shutdown(self):
        # ✅ 关键：给 shutdown 一个超时兜底（下面 fixture 里做）
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(scope="function")
def dummy_server_base_url():
    server = ServerThread(host="127.0.0.1", port=0)
    server.start()
    base_url = f"http://127.0.0.1:{server.port}"

    deadline = time.time() + 3
    while True:
        try:
            r = requests.get(f"{base_url}/health", timeout=(0.2, 0.2))
            if r.status_code == 200:
                break
        except Exception:
            pass
        if time.time() > deadline:
            server.httpd.server_close()
            raise RuntimeError("Dummy server did not become ready in time.")
        time.sleep(0.05)

    try:
        yield base_url
    finally:
        # ✅ 关键：shutdown 可能阻塞，所以放到独立线程并设超时兜底
        t = threading.Thread(target=server.shutdown, daemon=True)
        t.start()
        t.join(timeout=2)

        # 不管怎样都强制 close socket
        try:
            server.httpd.server_close()
        except Exception:
            pass

        server.join(timeout=2)
        assert not server.is_alive(), "Dummy server thread did not stop cleanly."