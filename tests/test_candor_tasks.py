from __future__ import annotations

from pathlib import Path

from candor_tasks import cli


def test_gammafresh_python_command_shape(monkeypatch):
    captured = {}

    def fake_run(cmd, cwd, check, capture_output, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd

        class _Completed:
            stdout = "/tmp/python\n"

        return _Completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    result = cli._gammafresh_python()

    assert result == "/tmp/python"
    assert captured["cmd"][:4] == ["uv", "run", "--project", str(cli.GAMMAFRESH_DIR)]


def test_paths_are_repo_relative():
    assert cli.ROOT == Path(__file__).resolve().parents[1]
    assert cli.GAMMAFRESH_DIR.name == "GammaFresh"
    assert cli.GAMMAFRESH_DIR.parent.name == "algorithms_impl"
