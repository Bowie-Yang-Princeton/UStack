"""Saved devices in the library (``library/devices/<name>/spec.json``): rename and delete.

A device folder holds its spec, the images copied in (Flake images, Stage 2) and the exported
Stage 2 steps. The spec stores absolute paths to those files, so renaming moves the folder and
rewrites every path inside the spec that pointed into the old folder."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import paths


class DeviceError(Exception):
    pass


def _inside_library(folder: Path) -> bool:
    try:
        folder.resolve().relative_to(paths.devices_dir().resolve())
        return True
    except ValueError:
        return False


def rewrite_paths(obj, old: str, new: str):
    if isinstance(obj, str):
        return new + obj[len(old):] if obj.startswith(old) else obj
    if isinstance(obj, list):
        return [rewrite_paths(v, old, new) for v in obj]
    if isinstance(obj, dict):
        return {k: rewrite_paths(v, old, new) for k, v in obj.items()}
    return obj


def rename_device(spec_path: str | Path, new_name: str) -> Path:
    """Give the device a new name: its folder moves to ``devices/<safe new name>`` and paths in the
    spec that pointed into the old folder follow. Returns the new spec path."""
    spec_path = Path(spec_path)
    new_name = new_name.strip()
    if not new_name:
        raise DeviceError("The name is empty.")
    old_dir = spec_path.parent
    if not _inside_library(old_dir):
        raise DeviceError(f"{old_dir} is not in the devices library.")
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    new_dir = paths.devices_dir() / paths.safe_name(new_name)
    if new_dir.resolve() != old_dir.resolve():
        if new_dir.exists():
            raise DeviceError(f"A device folder named '{new_dir.name}' already exists.")
        old_res = str(old_dir.resolve())
        shutil.move(str(old_dir), str(new_dir))
        data = rewrite_paths(data, old_res, str(new_dir.resolve()))
        if str(old_dir) != old_res:
            data = rewrite_paths(data, str(old_dir), str(new_dir.resolve()))
    data["name"] = new_name
    out = new_dir / "spec.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def delete_device(spec_path: str | Path) -> None:
    """Remove the device's folder (spec, copied images, exported steps). The original image files
    the user loaded from elsewhere are not touched."""
    folder = Path(spec_path).parent
    if not _inside_library(folder) or folder.resolve() == paths.devices_dir().resolve():
        raise DeviceError(f"{folder} is not a device folder in the library.")
    shutil.rmtree(folder)
