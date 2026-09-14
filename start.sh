#!/usr/bin/env bash
# Run on the BM1684X Linux box. No downloads, shell profile or config overwrites.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_root}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash start.sh [config-file] [--check-config] [--no-console]' \
        'Default config: config/config.yaml' \
        'Periodic example: bash start.sh config/config.periodic.yaml' \
        'Logs: logs/monitor-YYYY-MM-DD.log (unless runtime.log_dir overrides it)'
    exit 0
fi

config_path="config/config.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
    config_path="$1"
    shift
fi
python_bin="${project_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
    printf '%s\n' 'Missing .venv/bin/python. First run:' \
        '  python3 -m venv --system-site-packages .venv' \
        '  .venv/bin/python -m pip install -r requirements.txt' >&2
    exit 2
fi
if [[ ! -f "${config_path}" ]]; then
    printf 'Config not found: %s\n' "${config_path}" >&2
    printf '%s\n' 'Copy and edit config/config.periodic.example.yaml or config/config.yolo.example.yaml first.' >&2
    exit 2
fi

umask 077
export PYTHONUNBUFFERED=1
export PYTHONPATH="${project_root}/src:/opt/sophon/sophon-opencv-latest/opencv-python${PYTHONPATH:+:${PYTHONPATH}}"
if ! "${python_bin}" -c 'import numpy, yaml, httpx' >/dev/null 2>&1; then
    printf '%s\n' 'Missing dependencies. Run: .venv/bin/python -m pip install -r requirements.txt' >&2
    exit 2
fi

# A config check is read-only and can run while the service is active.
if [[ " $* " != *" --check-config "* ]]; then
    if ! command -v flock >/dev/null 2>&1; then
        printf '%s\n' 'Missing flock (util-linux); needed to prevent duplicate service instances.' >&2
        exit 2
    fi
    mkdir -p "${project_root}/logs"
    exec 9>"${project_root}/logs/.monitor.lock"
    if ! flock -n 9; then
        printf '%s\n' 'This project is already running. Stop the previous foreground/systemd instance first.' >&2
        exit 1
    fi
fi

# C/C++ stdout must be line-buffered before Python redirects it into the log pipe.
if command -v stdbuf >/dev/null 2>&1; then
    exec stdbuf -oL -eL "${python_bin}" -u -m fire_smoke_monitor --config "${config_path}" "$@"
fi
printf '%s\n' 'Warning: stdbuf unavailable; some native SDK diagnostics may be buffered.' >&2
exec "${python_bin}" -u -m fire_smoke_monitor --config "${config_path}" "$@"
