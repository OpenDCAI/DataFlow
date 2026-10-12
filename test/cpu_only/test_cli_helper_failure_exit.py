from __future__ import annotations

import sys
from types import ModuleType

import pytest
from typer.testing import CliRunner

import dataflow.cli_funcs as cli_funcs_pkg
from dataflow.cli import app


pytestmark = pytest.mark.cpu

runner = CliRunner()


COMMANDS = [
    ("pdf-init", ["pdf2model", "init"], "dataflow.cli_funcs.cli_pdf", "cli_pdf2model_init"),
    ("pdf-train", ["pdf2model", "train"], "dataflow.cli_funcs.cli_pdf", "cli_pdf2model_train"),
    ("text-init", ["text2model", "init"], "dataflow.cli_funcs.cli_text", "cli_text2model_init"),
    ("text-train", ["text2model", "train"], "dataflow.cli_funcs.cli_text", "cli_text2model_train"),
]


def _install_helper(monkeypatch, module_name: str, helper_name: str, behavior: str) -> None:
    module = ModuleType(module_name)

    if module_name.endswith("cli_pdf"):
        module.DATAFLEX_BACKEND_SPECS = {}

    def helper(*args, **kwargs):
        if behavior == "exception":
            raise RuntimeError("boom")
        print(f"helper-{behavior}")
        return behavior == "true"

    setattr(module, helper_name, helper)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(cli_funcs_pkg, module_name.rsplit(".", 1)[-1], module, raising=False)


@pytest.mark.parametrize("_name,args,module_name,helper_name", COMMANDS)
def test_helper_false_exits_nonzero_without_duplicate_wrapper_error(
    monkeypatch, _name, args, module_name, helper_name
):
    _install_helper(monkeypatch, module_name, helper_name, "false")

    result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert result.output.count("helper-false") == 1
    assert " error:" not in result.output.lower()


@pytest.mark.parametrize("_name,args,module_name,helper_name", COMMANDS)
def test_helper_true_preserves_success(monkeypatch, _name, args, module_name, helper_name):
    _install_helper(monkeypatch, module_name, helper_name, "true")

    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert result.output.count("helper-true") == 1


@pytest.mark.parametrize("_name,args,module_name,helper_name", COMMANDS)
def test_helper_exception_remains_nonzero(monkeypatch, _name, args, module_name, helper_name):
    _install_helper(monkeypatch, module_name, helper_name, "exception")

    result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert result.output.lower().count("error: boom") == 1
