"""Independent manager AST qualification compatibility tests."""
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.cpu

SETUP = r"""
import importlib
import sys
import tempfile
from pathlib import Path
import dataflow
r = importlib.import_module("dataflow.utils.registry")
tmp = tempfile.TemporaryDirectory()
root = Path(tmp.name) / "dataflow"
dataflow.__path__.append(str(root))
def make(name, files):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    for filename, contents in files.items():
        (directory / filename).write_text(contents, encoding="utf-8")
    orig = r.os.listdir
    r.os.listdir = lambda path: orig(directory) if Path(path).name == name else orig(path)
    try:
        instance = r.Registry(name)
    finally:
        r.os.listdir = orig
    r._manager_ast_registry = instance
    return instance
"""
def fresh(script):
    proc = subprocess.run([sys.executable, "-c", SETUP+"\n"+textwrap.dedent(script)],
                          text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout+"\n"+proc.stderr

def test_bare_register_decorator_preserves_earlier_builtin_precedence():
    fresh(r"""
        reg = make("p11_manager_bare_decorator", {
            "a.py": (
                "from dataflow.utils import registry as r\n"
                "@r._manager_ast_registry.register\n"
                "class Shared: pass\n"
            ),
            "z.py": (
                "from dataflow.utils import registry as r\n"
                "@r._manager_ast_registry.register()\n"
                "class Shared: pass\n"
            ),
        })
        importlib.import_module("dataflow.p11_manager_bare_decorator.z")
        resolved = reg.get("Shared")
        assert resolved.__module__ == "dataflow.p11_manager_bare_decorator.a", resolved.__module__
    """)

def test_statically_unreachable_registration_does_not_import_unrelated_module():
    fresh(r"""
        reg = make("p11_manager_dead_code", {
            "a.py": (
                "from dataflow.utils import registry as r\n"
                "if False:\n"
                "    @r._manager_ast_registry.register()\n"
                "    class Shared: pass\n"
                "raise RuntimeError('non-candidate import executed')\n"
            ),
            "z.py": (
                "from dataflow.utils import registry as r\n"
                "@r._manager_ast_registry.register()\n"
                "class Shared: pass\n"
            ),
        })
        importlib.import_module("dataflow.p11_manager_dead_code.z")
        resolved = reg.get("Shared")
        assert resolved.__module__ == "dataflow.p11_manager_dead_code.z"
        assert "dataflow.p11_manager_dead_code.a" not in sys.modules
    """)


def test_reachable_else_and_bare_decorator_are_discovered():
    fresh(r"""
        reg = make("p11_ast_else", {
            "a.py": (
                "from dataflow.utils import registry as r\n"
                "if False:\n"
                "    raise RuntimeError('impossible branch')\n"
                "else:\n"
                "    @r._manager_ast_registry.register\n"
                "    class Shared: pass\n"
            ),
            "z.py": (
                "from dataflow.utils import registry as r\n"
                "@r._manager_ast_registry.register()\n"
                "class Shared: pass\n"
            ),
        })
        importlib.import_module("dataflow.p11_ast_else.z")
        assert reg.get("Shared").__module__ == "dataflow.p11_ast_else.a"
    """)


def test_statically_false_not_condition_and_deferred_function_do_not_import():
    fresh(r"""
        reg = make("p11_ast_deferred", {
            "a.py": (
                "from dataflow.utils import registry as r\n"
                "if not True:\n"
                "    @r._manager_ast_registry.register()\n"
                "    class Shared: pass\n"
                "def never_called():\n"
                "    @r._manager_ast_registry.register()\n"
                "    class Shared: pass\n"
                "raise RuntimeError('unrelated import')\n"
            ),
            "z.py": (
                "from dataflow.utils import registry as r\n"
                "@r._manager_ast_registry.register()\n"
                "class Shared: pass\n"
            ),
        })
        importlib.import_module("dataflow.p11_ast_deferred.z")
        assert reg.get("Shared").__module__ == "dataflow.p11_ast_deferred.z"
        assert "dataflow.p11_ast_deferred.a" not in sys.modules
    """)


def test_unrelated_bare_decorator_does_not_trigger_import():
    fresh(r"""
        reg = make("p11_ast_unrelated_bare", {
            "a.py": (
                "from dataflow.utils.registry import OPERATOR_REGISTRY\n"
                "@OPERATOR_REGISTRY.register\n"
                "class Shared: pass\n"
                "raise RuntimeError('different registry')\n"
            ),
            "z.py": (
                "from dataflow.utils import registry as r\n"
                "@r._manager_ast_registry.register()\n"
                "class Shared: pass\n"
            ),
        })
        importlib.import_module("dataflow.p11_ast_unrelated_bare.z")
        assert reg.get("Shared").__module__ == "dataflow.p11_ast_unrelated_bare.z"
        assert "dataflow.p11_ast_unrelated_bare.a" not in sys.modules
    """)
