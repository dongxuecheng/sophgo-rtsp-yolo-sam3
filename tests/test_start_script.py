import os
from pathlib import Path
import shutil
import subprocess

import pytest


def write_script(path, text):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    path.chmod(0o700)


@pytest.fixture
def shell_project(tmp_path):
    bash = r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    try:
        subprocess.run([bash, "--version"], capture_output=True, check=True)
    except OSError:
        pytest.skip("bash unavailable")
    root = tmp_path / "project with spaces"
    root.mkdir()
    script = Path(__file__).resolve().parents[1] / "start.sh"
    shutil.copyfile(script, root / "start.sh")
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    write_script(python, '''#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then exit "${TEST_DEPS_EXIT:-0}"; fi
printf 'ARG=%s\\n' "$@"
printf 'PYTHONPATH=%s\\n' "$PYTHONPATH"
printf 'PYTHONUNBUFFERED=%s\\n' "$PYTHONUNBUFFERED"
exit "${TEST_PROGRAM_EXIT:-0}"
''')
    (root / "config").mkdir()
    (root / "config/config.yaml").touch()
    (root / "config/custom config.yaml").touch()
    helpers = root / "test-bin"
    helpers.mkdir()
    for name, source in {
        "flock": 'exit "${TEST_LOCK_EXIT:-0}"',
        "stdbuf": 'shift 2\nexec "$@"',
    }.items():
        path = helpers / name
        write_script(path, "#!/usr/bin/env bash\n" + source + "\n")

    def run(*args, **overrides):
        env = dict(os.environ, **overrides)
        # Bash interprets PATH with ':' even when launched from Windows.
        env["PATH"] = str(helpers) + os.pathsep + env.get("PATH", "")
        return subprocess.run([bash, "start.sh", *args], cwd=root, env=env,
                              capture_output=True, text=True, timeout=10)

    return root, run


def test_script_help_works_before_venv_or_config_exists(shell_project):
    root, run = shell_project
    (root / ".venv/bin/python").unlink()
    result = run("--help")
    assert result.returncode == 0
    assert "Periodic example" in result.stdout
    assert not (root / "logs").exists()


def test_script_preserves_config_path_with_spaces_and_exit_status(shell_project):
    root, run = shell_project
    result = run("config/custom config.yaml", "--check-config", TEST_PROGRAM_EXIT="7")
    assert result.returncode == 7, result.stderr
    assert "ARG=config/custom config.yaml" in result.stdout
    assert "ARG=--check-config" in result.stdout
    assert "PYTHONUNBUFFERED=1" in result.stdout
    assert "sophon-opencv-latest/opencv-python" in result.stdout
    assert not (root / "logs").exists()


def test_script_rejects_missing_config_without_creating_one(shell_project):
    root, run = shell_project
    result = run("config/missing.yaml")
    assert result.returncode == 2
    assert "Config not found" in result.stderr
    assert not (root / "config/missing.yaml").exists()


def test_script_reports_missing_dependencies_without_installing(shell_project):
    _, run = shell_project
    result = run(TEST_DEPS_EXIT="1")
    assert result.returncode == 2
    assert "pip install -r requirements.txt" in result.stderr


def test_script_lock_conflict_prevents_launch_but_allows_config_check(shell_project):
    _, run = shell_project
    result = run(TEST_LOCK_EXIT="1")
    assert result.returncode == 1
    assert "already running" in result.stderr
    assert "ARG=" not in result.stdout
    result = run("--check-config", TEST_LOCK_EXIT="1")
    assert result.returncode == 0, result.stderr


def test_default_start_forwards_console_option(shell_project):
    _, run = shell_project
    result = run("--no-console")
    assert result.returncode == 0, result.stderr
    assert "ARG=config/config.yaml" in result.stdout
    assert "ARG=--no-console" in result.stdout


def test_systemd_unit_enables_notify_watchdog_and_restart():
    unit = (Path(__file__).resolve().parents[1] /
            "deploy/fire-smoke-monitor.service").read_text(encoding="utf-8")
    assert "Type=notify" in unit
    assert "NotifyAccess=main" in unit
    assert "WatchdogSec=60" in unit
    assert "Restart=always" in unit
