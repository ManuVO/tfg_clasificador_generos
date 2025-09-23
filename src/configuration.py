"""Utilities to resolve stage-specific configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import copy
import yaml

ConfigDict = Dict[str, object]


def _load_yaml(path: Path) -> ConfigDict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """Recursively merge ``override`` into ``base`` returning a new dictionary."""

    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def load_stage_config(
    stage: str,
    *,
    config_path: Optional[str] = None,
    dataset: Optional[str] = None,
    project_config_path: str = "config.yaml",
) -> Tuple[ConfigDict, Dict[str, object]]:
    """Load a stage configuration merging project defaults and dataset overrides."""

    metadata: Dict[str, object] = {
        "stage": stage,
        "sources": [],
    }

    if config_path:
        manual_path = Path(config_path).expanduser().resolve()
        config = _load_yaml(manual_path)
        metadata.update(
            {
                "resolved_with": "explicit_config",
                "explicit_config_path": str(manual_path),
                "dataset": dataset,
            }
        )
        metadata["sources"].append(str(manual_path))
        return config, metadata

    project_path = Path(project_config_path).expanduser().resolve()
    project_root = project_path.parent

    project_cfg = _load_yaml(project_path)
    metadata.update(
        {
            "resolved_with": "project_defaults",
            "project_config_path": str(project_path),
        }
    )

    stages_cfg = project_cfg.get("stages", {})
    stage_entry = stages_cfg.get(stage)
    if not stage_entry:
        raise ValueError(
            f"Stage '{stage}' is not declared in {project_config_path}."
        )

    base_config: ConfigDict = {}
    base_path = stage_entry.get("base") or stage_entry.get("base_config")
    if base_path:
        base_resolved = _resolve_path(base_path, project_root)
        base_config = _load_yaml(base_resolved)
        metadata["base_config_path"] = str(base_resolved)
        metadata["sources"].append(str(base_resolved))

    dataset_name = dataset or project_cfg.get("defaults", {}).get("dataset")
    if not dataset_name:
        raise ValueError(
            "Dataset must be provided either explicitly or via project defaults."
        )

    datasets_cfg = project_cfg.get("datasets", {})
    dataset_entry = datasets_cfg.get(dataset_name)
    if not dataset_entry:
        raise ValueError(
            f"Dataset '{dataset_name}' is not declared in {project_config_path}."
        )

    metadata["dataset"] = dataset_name
    if display_name := dataset_entry.get("display_name"):
        metadata["dataset_display_name"] = display_name

    dataset_stages = dataset_entry.get("stages", {})
    dataset_stage_path = dataset_stages.get(stage)
    if dataset_stage_path is None and stage == "evaluation":
        dataset_stage_path = dataset_stages.get("training")

    merged_config = base_config
    if dataset_stage_path:
        dataset_resolved = _resolve_path(dataset_stage_path, project_root)
        dataset_config = _load_yaml(dataset_resolved)
        merged_config = _deep_merge(merged_config, dataset_config)
        metadata["dataset_config_path"] = str(dataset_resolved)
        metadata["sources"].append(str(dataset_resolved))
    else:
        metadata["dataset_config_path"] = None

    metadata["sources"] = list(dict.fromkeys(metadata["sources"]))
    return merged_config, metadata


__all__ = ["load_stage_config"]
