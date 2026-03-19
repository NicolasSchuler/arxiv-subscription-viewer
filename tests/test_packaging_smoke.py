"""Packaging smoke tests for wheel contents and entrypoints."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tomllib
import zipfile
from pathlib import Path


def test_console_script_targets_cli_main() -> None:
    """The published console script should point at the canonical CLI entrypoint."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["arxiv-viewer"] == "arxiv_browser.cli:main"


def test_wheel_installs_cli_and_module_entrypoints(tmp_path) -> None:
    """A built wheel should contain typing metadata and runnable entrypoints."""
    repo_root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    assert uv is not None

    dist_dir = tmp_path / "dist"
    subprocess.run(
        [uv, "build", "--wheel", "-o", str(dist_dir)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(dist_dir.glob("*.whl"))

    with zipfile.ZipFile(wheel) as archive:
        assert "arxiv_browser/py.typed" in archive.namelist()

    venv_dir = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    bin_dir = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
    python = bin_dir / ("python.exe" if sys.platform == "win32" else "python")
    console_script = bin_dir / ("arxiv-viewer.exe" if sys.platform == "win32" else "arxiv-viewer")

    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    env = dict(os.environ)
    current_purelib = sysconfig.get_paths()["purelib"]
    env["PYTHONPATH"] = (
        current_purelib
        if not env.get("PYTHONPATH")
        else f"{current_purelib}{os.pathsep}{env['PYTHONPATH']}"
    )

    console_result = subprocess.run(
        [str(console_script), "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    module_result = subprocess.run(
        [str(python), "-m", "arxiv_browser", "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "arxiv-viewer" in console_result.stdout
    assert "arxiv-viewer" in module_result.stdout
