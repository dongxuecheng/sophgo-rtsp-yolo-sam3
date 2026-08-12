from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_official_module(module_name: str, source_file: Path) -> ModuleType:
    """Load one pinned sophon-demo sample without copying vendor code here."""

    source_file = source_file.resolve()
    source_dir = str(source_file.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = importlib.util.spec_from_file_location(module_name, str(source_file))
    if spec is None or spec.loader is None:
        raise ImportError("无法加载官方模块: {}".format(source_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

