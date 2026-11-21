"""Where UStack keeps its files.

Everything lives next to the sources by default (the UStack folder), so a lab member opening the
shared folder sees the same configuration, templates and devices. ``USTACK_HOME`` moves the
library elsewhere; ``USTACK_CONFIG`` points at another config folder.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path(__file__).resolve().parent


def config_dir() -> Path:
    return Path(os.environ.get("USTACK_CONFIG", ROOT / "config"))


def library_dir() -> Path:
    path = Path(os.environ.get("USTACK_HOME", ROOT / "library"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_templates_dir() -> Path:
    path = library_dir() / "templates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def devices_dir() -> Path:
    path = library_dir() / "devices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def builtin_templates_dir() -> Path:
    return PACKAGE / "data" / "templates"


def safe_name(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in text.strip())
    return "_".join(keep.split()) or "untitled"
