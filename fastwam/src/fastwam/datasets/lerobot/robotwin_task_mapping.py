from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_task_names(value: Any) -> list[str] | None:
    if value is None:
        return None

    normalized: list[str] = []
    for item in value:
        task_name = str(item).strip()
        if not task_name:
            continue
        normalized.append(task_name)

    if not normalized:
        return None

    deduped: list[str] = []
    seen = set()
    for task_name in normalized:
        if task_name in seen:
            continue
        seen.add(task_name)
        deduped.append(task_name)
    return deduped


def load_robotwin_task_mapping(mapping_file: Path) -> dict[str, list[int]]:
    if not mapping_file.is_file():
        raise FileNotFoundError(f"RobotWin task mapping file not found: {mapping_file}")

    first_nonempty_line = None
    with mapping_file.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                first_nonempty_line = stripped
                break

    if first_nonempty_line is None:
        raise ValueError(f"RobotWin task mapping file is empty: {mapping_file}")

    if first_nonempty_line.startswith("{") and mapping_file.suffix == ".json":
        payload = json.loads(mapping_file.read_text(encoding="utf-8"))
        task_to_episodes = _load_compact_robotwin_task_mapping(payload, mapping_file)
    else:
        task_to_episodes = _load_detailed_robotwin_task_mapping(mapping_file)

    for task_name in list(task_to_episodes.keys()):
        task_to_episodes[task_name] = sorted(set(task_to_episodes[task_name]))

    return task_to_episodes


def resolve_selected_episode_indices(
    *,
    mapping: dict[str, list[int]],
    selected_task_names: list[str],
) -> list[int]:
    missing_tasks = sorted(set(selected_task_names) - set(mapping.keys()))
    if missing_tasks:
        raise ValueError(
            f"Selected task names missing from mapping: {missing_tasks}"
        )

    return sorted(
        {
            episode_idx
            for task_name in selected_task_names
            for episode_idx in mapping[task_name]
        }
    )


def _load_detailed_robotwin_task_mapping(mapping_file: Path) -> dict[str, list[int]]:
    task_to_episodes: dict[str, list[int]] = {}
    with mapping_file.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_name = str(record["task_name"]).strip()
            episode_idx = int(record["processed_episode_index"])
            strict_confidence = bool(record.get("strict_confidence", True))
            if not strict_confidence:
                raise ValueError(
                    f"Mapping file contains non-strict record at line {line_idx}: {mapping_file}"
                )
            task_to_episodes.setdefault(task_name, []).append(episode_idx)
    return task_to_episodes


def _load_compact_robotwin_task_mapping(
    payload: dict[str, Any],
    mapping_file: Path,
) -> dict[str, list[int]]:
    file_format = payload.get("format")
    if file_format != "robotwin_task_ranges_v1":
        raise ValueError(
            f"Unsupported compact RobotWin task mapping format in {mapping_file}: {file_format}"
        )

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Invalid compact RobotWin mapping in {mapping_file}: missing `tasks` list")

    task_to_episodes: dict[str, list[int]] = {}
    for item in tasks:
        task_name = str(item["task_name"]).strip()
        episode_start = int(item["episode_start"])
        episode_end = int(item["episode_end"])
        episode_count = int(item["episode_count"])
        if episode_end < episode_start:
            raise ValueError(
                f"Invalid episode range for task {task_name} in {mapping_file}: "
                f"{episode_start}..{episode_end}"
            )
        episode_indices = list(range(episode_start, episode_end + 1))
        if len(episode_indices) != episode_count:
            raise ValueError(
                f"Episode count mismatch for task {task_name} in {mapping_file}: "
                f"range gives {len(episode_indices)} but episode_count={episode_count}"
            )
        task_to_episodes[task_name] = episode_indices

    return task_to_episodes
