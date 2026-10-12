from __future__ import annotations

import inspect

import pytest
from typer.testing import CliRunner

from dataflow.cli import app
import dataflow.cli_funcs.cli_text as cli_text


pytestmark = pytest.mark.cpu
runner = CliRunner()


def test_text2model_train_explicit_input_dir_crosses_real_helper_boundary(monkeypatch, tmp_path):
    """Public CLI must cross the real cli_text2model_train call boundary."""
    body_reached = {"value": False}

    def controlled_verify_environment() -> bool:
        body_reached["value"] = True
        return False

    monkeypatch.setattr(cli_text, "verify_environment", controlled_verify_environment)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    result = runner.invoke(app, ["text2model", "train", str(input_dir)])

    assert "unexpected keyword argument 'input_dir'" not in result.output
    assert body_reached["value"] is True
    # The controlled deeper dependency returns False, so the public wrapper must
    # still propagate helper failure to a nonzero process status.
    assert result.exit_code != 0


def test_text2model_train_signature_preserves_default_and_old_positional_usage(monkeypatch):
    signature = inspect.signature(cli_text.cli_text2model_train)
    assert signature.parameters["input_dir"].default == "./"

    calls = {"count": 0}

    def controlled_verify_environment() -> bool:
        calls["count"] += 1
        return False

    monkeypatch.setattr(cli_text, "verify_environment", controlled_verify_environment)

    # Existing two-positional-argument callers remain valid.
    assert cli_text.cli_text2model_train("text", "./.cache/train_config.yaml") is False
    # Default input_dir remains backward compatible.
    assert cli_text.cli_text2model_train() is False
    assert calls["count"] == 2
