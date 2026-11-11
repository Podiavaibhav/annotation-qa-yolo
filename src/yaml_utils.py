"""Helpers for dataset yaml files.

Ultralytics resolves a relative `path:` against its own datasets folder, not against the yaml
file. A yaml that says `path: datasets/coco8` therefore sends it looking in the wrong place,
or, worse, at a different copy of the data. Every yaml we hand to Ultralytics gets an absolute
`path` first.
"""
import os
from pathlib import Path
from typing import Optional

import yaml


def _find_root(src_yaml: Path, cfg: dict) -> Path:
    """Work out which folder actually holds the dataset."""
    raw = cfg.get("path")
    if raw:
        p = Path(raw)
        if p.is_absolute():
            if p.exists():
                return p
        else:
            cand = src_yaml.parent / p
            train = cfg.get("train")
            if cand.exists() and (not isinstance(train, str) or (cand / train).exists()):
                return cand
    return src_yaml.parent


def set_yaml_root(yaml_path, root) -> None:
    """Rewrite a yaml in place so `path` is the absolute `root`."""
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(Path(root).resolve())
    cfg.pop("download", None)  # never let Ultralytics run a download script from a copied yaml
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def absolutize_yaml(src_yaml, out_yaml, dataset_root: Optional[str] = None) -> Path:
    """Write a copy of `src_yaml` to `out_yaml` with an absolute `path`."""
    src_yaml = Path(src_yaml).resolve()
    with open(src_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    root = Path(dataset_root) if dataset_root else _find_root(src_yaml, cfg)
    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    set_yaml_root(out_yaml, root)
    return out_yaml


def write_prepared_yaml(info: dict, name: str = "data.yaml") -> Path:
    """Turn the dict Ultralytics' check_det_dataset returns into a yaml inside the dataset folder.

    Split entries become paths relative to the dataset root and `path` is absolute, so the yaml
    works for the loader, the noise injector (which copies the folder) and Ultralytics itself.
    """
    root = Path(info["path"]).resolve()

    def rel(p):
        p = p[0] if isinstance(p, (list, tuple)) else p
        p = Path(p).resolve()
        try:
            return str(p.relative_to(root))
        except ValueError:
            return str(p)

    names = info["names"]
    cfg = {"path": str(root), "train": rel(info["train"]), "val": rel(info["val"]),
           "nc": len(names), "names": dict(names) if isinstance(names, dict) else list(names)}
    if info.get("test"):
        cfg["test"] = rel(info["test"])
    out = root / name
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out
