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
