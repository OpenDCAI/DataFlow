"""Runtime regressions for atomic discovery and stable duplicate-name resolution."""

import subprocess
import sys
import textwrap

import pytest


pytestmark = pytest.mark.cpu

_SETUP = r"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import dataflow
registry_module = importlib.import_module("dataflow.utils.registry")

temporary = tempfile.TemporaryDirectory()
dataflow.__path__.append(str(Path(temporary.name) / "dataflow"))


def fixture_registry(package_name, files, reverse=False):
    directory = Path(temporary.name) / "dataflow" / package_name
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    for filename, contents in files.items():
        (directory / filename).write_text(contents, encoding="utf-8")

    # Include a deliberately reversed filesystem order.
    listdir = registry_module.os.listdir
    def ordered_listdir(path):
        if Path(path).name == package_name:
            entries = listdir(directory)
            return list(reversed(entries)) if reverse else entries
        return listdir(path)
    registry_module.os.listdir = ordered_listdir
    try:
        registry = registry_module.Registry(package_name)
    finally:
        registry_module.os.listdir = listdir
    registry_module._p11_test_registry = registry
    return registry, directory
"""


def _run_fresh(script):
    result = subprocess.run(
        [sys.executable, "-c", _SETUP + "\n" + textwrap.dedent(script)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"Fresh-process registry regression failed ({result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_failed_discovery_restores_modules_registry_and_retry():
    _run_fresh(
        r"""
        helper = (
            "from dataflow.utils import registry as r\n"
            "@r._p11_test_registry.register()\n"
            "class Survivor: pass\n"
        )
        after = (
            "from dataflow.utils import registry as r\n"
            "@r._p11_test_registry.register()\n"
            "class After: pass\n"
        )
        registry, directory = fixture_registry(
            "p11_recovery",
            {
                "a.py": "from . import helper\nraise RuntimeError('discovery boom')\n",
                "helper.py": helper,
                "z.py": after,
            },
            reverse=True,
        )
        try:
            registry._get_all()
        except RuntimeError as error:
            assert "discovery boom" in str(error)
        else:
            raise AssertionError("Expected original import exception")

        assert registry.get_obj_map() == {}
        assert all(loader is None for loader in registry.loader_map.values())
        assert "dataflow.p11_recovery.helper" not in sys.modules
        assert "dataflow.p11_recovery.z" not in sys.modules

        (directory / "a.py").write_text("from . import helper\n", encoding="utf-8")
        importlib.invalidate_caches()
        registry._get_all()
        actual = {key: value.__name__ for key, value in registry.get_obj_map().items()}
        assert actual == {"After": "After", "Survivor": "Survivor"}

        fresh, _ = fixture_registry(
            "p11_recovery_fresh",
            {"a.py": "from . import helper\n", "helper.py": helper, "z.py": after},
        )
        fresh._get_all()
        expected = {key: value.__name__ for key, value in fresh.get_obj_map().items()}
        assert actual == expected
        """
    )


def test_lazy_lookup_failure_is_recoverable():
    _run_fresh(
        r"""
        registry, directory = fixture_registry(
            "p11_lookup",
            {
                "broken.py": (
                    "from dataflow.utils import registry as r\n"
                    "@r._p11_test_registry.register()\n"
                    "class Partial: pass\n"
                    "raise ValueError('lookup boom')\n"
                ),
            },
        )
        try:
            registry.get("Partial")
        except ValueError as error:
            assert str(error) == "lookup boom"
        else:
            raise AssertionError("Expected original lookup exception")
        assert not registry.get_obj_map()
        assert all(loader is None for loader in registry.loader_map.values())

        (directory / "broken.py").write_text(
            "from dataflow.utils import registry as r\n"
            "@r._p11_test_registry.register()\n"
            "class Partial: pass\n",
            encoding="utf-8",
        )
        importlib.invalidate_caches()
        assert registry.get("Partial").__module__ == "dataflow.p11_lookup.broken"
        """
    )


def test_duplicate_winner_and_iteration_are_import_order_independent():
    _run_fresh(
        r"""
        source = (
            "from dataflow.utils import registry as r\n"
            "@r._p11_test_registry.register()\n"
            "class Shared: pass\n"
        )
        files = {"z.py": source, "a.py": source}
        left, _ = fixture_registry("p11_duplicate_left", files, reverse=True)
        importlib.import_module("dataflow.p11_duplicate_left.z")
        left._get_all()
        assert left.get("Shared").__module__ == "dataflow.p11_duplicate_left.a"
        assert list(left.keys()) == sorted(left.keys())

        right, _ = fixture_registry("p11_duplicate_right", files)
        importlib.import_module("dataflow.p11_duplicate_right.a")
        right._get_all()
        assert right.get("Shared").__module__ == "dataflow.p11_duplicate_right.a"
        assert list(right.keys()) == sorted(right.keys())

        external, _ = fixture_registry("p11_duplicate_plugin", files, reverse=True)
        custom = type("Shared", (), {"__module__": "custom_user_extension"})
        external.register(custom)
        external._get_all()
        assert external.get("Shared") is custom
        """
    )


def test_lazy_file_import_failure_does_not_poison_sys_modules():
    _run_fresh(
        r"""
        path = Path(temporary.name) / "dataflow" / "p11_lazy" / "nested" / "entry.py"
        path.parent.mkdir(parents=True)
        path.write_text("raise RuntimeError('lazy boom')\n", encoding="utf-8")
        loader = registry_module.LazyLoader(
            "dataflow.p11_lazy", str(path.parent),
            {"Recovered": (str(path), "Recovered")},
        )
        loader._base_folder = Path(temporary.name)
        try:
            loader.Recovered
        except RuntimeError as error:
            assert str(error) == "lazy boom"
        else:
            raise AssertionError("Expected original lazy import exception")
        assert "dataflow.p11_lazy.nested.entry" not in sys.modules
        path.write_text("class Recovered: pass\n", encoding="utf-8")
        importlib.invalidate_caches()
        assert loader.Recovered.__name__ == "Recovered"
        """
    )


def test_lookup_duplicate_is_deterministic_before_explicit_full_discovery():
    _run_fresh(
        r"""
        source = (
            "from dataflow.utils import registry as r\n"
            "@r._p11_test_registry.register()\n"
            "class Shared: pass\n"
        )
        registry, _ = fixture_registry(
            "p11_early_collision", {"z.py": source, "a.py": source}
        )
        importlib.import_module("dataflow.p11_early_collision.z")
        assert registry.get("Shared").__module__ == "dataflow.p11_early_collision.a"
        """
    )


def test_direct_lazy_failure_rolls_back_partial_registry_registration():
    _run_fresh(
        r"""
        path = Path(temporary.name) / "dataflow" / "p11_lazy_registration" / "entry.py"
        path.parent.mkdir(parents=True)
        path.write_text(
            "from dataflow.utils import registry as r\n"
            "@r.OPERATOR_REGISTRY.register()\n"
            "class PartialByFailedLazy: pass\n"
            "raise RuntimeError('direct lazy failure')\n",
            encoding="utf-8",
        )
        loader = registry_module.LazyLoader(
            "dataflow.p11_lazy_registration",
            str(path.parent),
            {"Recovered": (str(path), "Recovered")},
        )
        loader._base_folder = Path(temporary.name)
        before = dict(registry_module.OPERATOR_REGISTRY.get_obj_map())
        try:
            loader.Recovered
        except RuntimeError as error:
            assert str(error) == "direct lazy failure"
        else:
            raise AssertionError("Expected failed direct lazy import")
        assert registry_module.OPERATOR_REGISTRY.get_obj_map() == before
        """
    )


def test_early_collision_resolution_keeps_unrelated_modules_lazy():
    _run_fresh(
        r"""
        source = (
            "from dataflow.utils import registry as r\n"
            "@r._p11_test_registry.register()\n"
            "class Shared: pass\n"
        )
        registry, _ = fixture_registry(
            "p11_selective",
            {
                "z.py": source,
                "a.py": source,
                "b.py": "raise RuntimeError('unrelated source must stay lazy')\n",
            },
        )
        importlib.import_module("dataflow.p11_selective.z")
        assert registry.get("Shared").__module__ == "dataflow.p11_selective.a"
        assert "dataflow.p11_selective.b" not in sys.modules
        assert registry.loader_map["b.py"] is None
        """
    )


def test_failure_preserves_preloaded_modules_and_explicit_registration():
    _run_fresh(
        r"""
        registry, directory = fixture_registry(
            "p11_preserved",
            {
                "safe.py": "class Safe: pass\n",
                "broken.py": (
                    "from dataflow.utils import registry as r\n"
                    "@r._p11_test_registry.register()\n"
                    "class Leaked: pass\n"
                    "raise RuntimeError('expected failure')\n"
                ),
            },
        )
        safe_module = importlib.import_module("dataflow.p11_preserved.safe")
        external = type("UserDefined", (), {"__module__": "extension_module"})
        registry.register(external)
        try:
            registry._get_all()
        except RuntimeError as error:
            assert str(error) == "expected failure"
        else:
            raise AssertionError("Expected import failure")
        assert sys.modules["dataflow.p11_preserved.safe"] is safe_module
        assert registry.get_obj_map() == {"UserDefined": external}
        assert registry.loader_map["safe.py"] is None
        """
    )


def test_direct_lazy_retry_preserves_prior_objects_without_leaks():
    _run_fresh(
        r"""
        path = Path(temporary.name) / "dataflow" / "p11_direct_retry" / "entry.py"
        path.parent.mkdir(parents=True)
        external = type("UserDefined", (), {"__module__": "extension_module"})
        registry_module.OPERATOR_REGISTRY.register(external)
        baseline = dict(registry_module.OPERATOR_REGISTRY.get_obj_map())
        path.write_text(
            "from dataflow.utils.registry import OPERATOR_REGISTRY\n"
            "@OPERATOR_REGISTRY.register()\n"
            "class Recovered: pass\n"
            "raise ValueError('retry needed')\n",
            encoding="utf-8",
        )
        loader = registry_module.LazyLoader(
            "dataflow.p11_direct_retry", str(path.parent),
            {"Recovered": (str(path), "Recovered")},
        )
        loader._base_folder = Path(temporary.name)
        try:
            loader.Recovered
        except ValueError as error:
            assert str(error) == "retry needed"
        else:
            raise AssertionError("Expected import failure")
        assert registry_module.OPERATOR_REGISTRY.get_obj_map() == baseline
        assert "dataflow.p11_direct_retry.entry" not in sys.modules
        path.write_text(
            "from dataflow.utils.registry import OPERATOR_REGISTRY\n"
            "@OPERATOR_REGISTRY.register()\n"
            "class Recovered: pass\n",
            encoding="utf-8",
        )
        importlib.invalidate_caches()
        recovered = loader.Recovered
        assert loader.Recovered is recovered
        assert registry_module.OPERATOR_REGISTRY.get("Recovered") is recovered
        assert registry_module.OPERATOR_REGISTRY.get("UserDefined") is external
        """
    )
