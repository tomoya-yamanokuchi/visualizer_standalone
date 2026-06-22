from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional

import numpy as np
import yaml
from tqdm import tqdm

try:
    from .action_table import build_action_table
    from .fast_voxel_renderer import render_cutting_process_gif_fast
    from .io_utils import (
        ensure_dir,
        list_subdirs,
        load_image_as_numpy,
        load_pickle,
        numpy_to_pil,
        save_numpy_image,
    )
except ImportError:  # pragma: no cover - allows running this file directly.
    from action_table import build_action_table
    from fast_voxel_renderer import render_cutting_process_gif_fast
    from io_utils import (
        ensure_dir,
        list_subdirs,
        load_image_as_numpy,
        load_pickle,
        numpy_to_pil,
        save_numpy_image,
    )


DEFAULT_CONFIG = {
    "root_folder": "./eval_results",
    "tags": [],
    "save_prefix": "no_axis_w_cutting_plane3",
    "save_subdir": "3d_cutting_process_fast",
    "rollout_filename": "rollout_data.pickle",
    "oracle_obs_filename": "oracle_obs_cast_z_axis0.png",
    "dim_2d": 64,
    "dim_3d": 16,
    "bounds": [-0.05, 0.05, -0.05, 0.05, -0.05, 0.05],
    "model_type_indices": None,
    "episode_indices": None,
    "max_episodes": 1,
    "paper_frame_interleave": True,
    "renderer": {
        "save_eps": False,
        "save_png": False,
        "save_pdf": False,
        "show_edges": True,
        "draw_context_voxels": True,
        "window_size": 800,
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "visualization" in raw and isinstance(raw["visualization"], dict):
        raw = raw["visualization"]

    return _deep_update(DEFAULT_CONFIG, raw)


def _parse_optional_int_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in {"none", "null"}:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _select_by_indices(items: list[str], indices: Optional[Iterable[int]]) -> list[str]:
    if indices is None:
        return items
    return [items[int(i)] for i in indices]


def _resolve_episode_indices(
    episodes: list[str],
    episode_indices: Optional[Iterable[int]],
    max_episodes: Optional[int],
) -> list[int]:
    if episode_indices is not None:
        return [int(i) for i in episode_indices]

    if max_episodes is None:
        return list(range(len(episodes)))

    return list(range(min(int(max_episodes), len(episodes))))


def _build_root_folders(cfg: dict[str, Any]) -> list[Path]:
    root_folder = Path(str(cfg["root_folder"])).expanduser()
    tags = _as_list(cfg.get("tags", None))

    if not tags:
        return [root_folder]

    return [root_folder / str(tag) for tag in tags]


def _build_grid_config(cfg: dict[str, Any]) -> dict[str, Any]:
    bounds = tuple(float(x) for x in cfg["bounds"])
    if len(bounds) != 6:
        raise ValueError(f"bounds must contain 6 values, got {len(bounds)}: {bounds}")

    return {
        "bounds": bounds,
        "side_length": int(cfg["dim_3d"]),
    }


def _make_paper_interleaved_frames(
    oracle_2d_map: np.ndarray,
    cutting_process_2d_map: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cutting_process_2d_map_base = np.concatenate(
        [oracle_2d_map[None, :, :, :] * 0.0, cutting_process_2d_map],
        axis=0,
    )

    action_base = np.concatenate([np.asarray([0]), np.asarray(action)], axis=0)
    step_num, width, _, channel = cutting_process_2d_map_base.shape

    interleaved_maps = np.empty(
        (int(step_num * 2), width, width, channel),
        dtype=cutting_process_2d_map_base.dtype,
    )
    interleaved_maps[0::2] = cutting_process_2d_map_base
    interleaved_maps[1::2] = cutting_process_2d_map_base.copy()

    interleaved_actions = np.empty((int(step_num * 2),), dtype=action_base.dtype)
    interleaved_actions[0::2] = action_base
    interleaved_actions[1::2] = action_base.copy()
    interleaved_actions = np.roll(interleaved_actions, -1)

    interleaved_maps = np.concatenate([interleaved_maps, interleaved_maps[-1:]], axis=0)
    interleaved_actions = np.concatenate([interleaved_actions, interleaved_actions[-1:]], axis=0)

    return interleaved_maps, interleaved_actions


def _resize_cutting_process_maps(cutting_process_2d_map: np.ndarray, dim_2d: int) -> np.ndarray:
    resized = []
    for frame_idx in range(cutting_process_2d_map.shape[0]):
        pil_image = numpy_to_pil(cutting_process_2d_map[frame_idx])
        resized_image = pil_image.resize((dim_2d, dim_2d))
        resized.append(np.asarray(resized_image) / 255.0)
    return np.asarray(resized)


def _make_remaining_voxel_maps(
    oracle_2d_map: np.ndarray,
    cutting_process_2d_map: np.ndarray,
) -> np.ndarray:
    return np.where(
        (cutting_process_2d_map >= oracle_2d_map - 0.05)
        & (cutting_process_2d_map <= oracle_2d_map + 0.05),
        np.asarray([0.0, 0.0, 0.0]),
        oracle_2d_map,
    ) * 255.0


def _mask_over_cutting_voxels(
    oracle_2d_map: np.ndarray,
    cutting_process_2d_map_flip: np.ndarray,
) -> np.ndarray:
    masked = cutting_process_2d_map_flip.copy()

    for frame_idx in range(masked.shape[0]):
        over_cutting_voxels = (
            np.all(oracle_2d_map == np.asarray([0.2, 0.8, 0.8]), axis=-1)
            & np.all(masked[frame_idx] / 255.0 == [0, 0, 0], axis=-1)
        )

        frame = (masked[frame_idx] / 255.0).copy()
        frame[over_cutting_voxels] = np.asarray([148 / 255, 0.0, 211 / 255])
        masked[frame_idx] = frame * 255.0

    return masked


def _window_size_from_config(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return (800, 800)


def render_episode_fast(
    *,
    data_folder: Path,
    cfg: dict[str, Any],
    grid_config: dict[str, Any],
    action_table: dict,
) -> None:
    dim_2d = int(cfg["dim_2d"])
    dim_3d = int(cfg["dim_3d"])

    save_prefix = str(cfg["save_prefix"])
    save_name = f"dim_{dim_3d}_{save_prefix}"

    save_folder = data_folder / str(cfg.get("save_subdir", "3d_cutting_process_fast"))
    ensure_dir(save_folder)

    rollout_path = data_folder / str(cfg["rollout_filename"])
    oracle_obs_path = data_folder / str(cfg["oracle_obs_filename"])

    print(f"load_data: {data_folder}")
    rollout_data = load_pickle(rollout_path)
    oracle_2d_map = load_image_as_numpy(oracle_obs_path, resize=(dim_2d, dim_2d))

    cutting_process_2d_map = np.asarray(rollout_data["observations"])
    action = np.asarray(rollout_data["actions"])

    if bool(cfg.get("paper_frame_interleave", True)):
        cutting_process_2d_map, action = _make_paper_interleaved_frames(
            oracle_2d_map=oracle_2d_map,
            cutting_process_2d_map=cutting_process_2d_map,
            action=action,
        )

    print(f"action_idx: {action}")

    cutting_process_2d_map = _resize_cutting_process_maps(
        cutting_process_2d_map=cutting_process_2d_map,
        dim_2d=dim_2d,
    )

    cutting_process_2d_map_flip = _make_remaining_voxel_maps(
        oracle_2d_map=oracle_2d_map,
        cutting_process_2d_map=cutting_process_2d_map,
    )
    save_numpy_image(cutting_process_2d_map_flip[-1] / 255.0, data_folder / "last_remain_voxels.png")

    cutting_process_2d_map_flip = _mask_over_cutting_voxels(
        oracle_2d_map=oracle_2d_map,
        cutting_process_2d_map_flip=cutting_process_2d_map_flip,
    )
    save_numpy_image(
        cutting_process_2d_map_flip[-1] / 255.0,
        data_folder / "last_remain_voxels_w_ocv_masked.png",
    )

    renderer_cfg = cfg.get("renderer", {}) or {}
    start_time = perf_counter()

    render_cutting_process_gif_fast(
        save_path=str(save_folder),
        grid_config=grid_config,
        action=action,
        action_table=action_table,
        sample_images=cutting_process_2d_map_flip,
        save_tag=save_name,
        save_eps=bool(renderer_cfg.get("save_eps", False)),
        save_png=bool(renderer_cfg.get("save_png", False)),
        save_pdf=bool(renderer_cfg.get("save_pdf", False)),
        show_edges=bool(renderer_cfg.get("show_edges", True)),
        draw_context_voxels=bool(renderer_cfg.get("draw_context_voxels", True)),
        window_size=_window_size_from_config(renderer_cfg.get("window_size", 800)),
    )

    elapsed = perf_counter() - start_time
    print(f"[fast-render] {data_folder} finished in {elapsed:.2f} sec")


def render_root_folder_fast(
    *,
    root_folder: Path,
    cfg: dict[str, Any],
    grid_config: dict[str, Any],
    action_table: dict,
) -> None:
    if not root_folder.exists():
        raise FileNotFoundError(f"root_folder does not exist: {root_folder}")

    model_type_folders = _select_by_indices(
        items=list_subdirs(root_folder),
        indices=cfg.get("model_type_indices", None),
    )

    for model_type_folder in model_type_folders:
        episodes_folder = root_folder / model_type_folder
        episodes = list_subdirs(episodes_folder)
        target_episode_indices = _resolve_episode_indices(
            episodes=episodes,
            episode_indices=cfg.get("episode_indices", None),
            max_episodes=cfg.get("max_episodes", None),
        )

        for episode_idx in tqdm(
            target_episode_indices,
            desc=f"fast episodes/{model_type_folder}",
            unit="episode",
            dynamic_ncols=True,
        ):
            data_folder = episodes_folder / episodes[episode_idx]
            render_episode_fast(
                data_folder=data_folder,
                cfg=cfg,
                grid_config=grid_config,
                action_table=action_table,
            )


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = _deep_update({}, cfg)

    if args.root_folder is not None:
        cfg["root_folder"] = args.root_folder
    if args.tags is not None:
        cfg["tags"] = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    if args.max_episodes is not None:
        cfg["max_episodes"] = args.max_episodes
    if args.episode_indices is not None:
        cfg["episode_indices"] = _parse_optional_int_list(args.episode_indices)
    if args.model_type_indices is not None:
        cfg["model_type_indices"] = _parse_optional_int_list(args.model_type_indices)
    if args.dim_2d is not None:
        cfg["dim_2d"] = args.dim_2d
    if args.dim_3d is not None:
        cfg["dim_3d"] = args.dim_3d
    if args.paper_frame_interleave is not None:
        cfg["paper_frame_interleave"] = args.paper_frame_interleave
    if args.save_subdir is not None:
        cfg["save_subdir"] = args.save_subdir

    renderer_cfg = cfg.setdefault("renderer", {})
    if args.save_eps is not None:
        renderer_cfg["save_eps"] = args.save_eps
    if args.save_png is not None:
        renderer_cfg["save_png"] = args.save_png
    if args.save_pdf is not None:
        renderer_cfg["save_pdf"] = args.save_pdf
    if args.show_edges is not None:
        renderer_cfg["show_edges"] = args.show_edges
    if args.draw_context_voxels is not None:
        renderer_cfg["draw_context_voxels"] = args.draw_context_voxels
    if args.window_size is not None:
        renderer_cfg["window_size"] = args.window_size

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental faster cutting-process renderer with cached voxel geometry.",
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--root-folder", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None, help="Comma-separated tag names.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode-indices", type=str, default=None, help="Comma-separated indices, e.g. 0,2,5")
    parser.add_argument("--model-type-indices", type=str, default=None, help="Comma-separated indices, e.g. 0,1")
    parser.add_argument("--dim-2d", type=int, default=None)
    parser.add_argument("--dim-3d", type=int, default=None)
    parser.add_argument("--save-subdir", type=str, default=None)
    parser.add_argument("--window-size", type=int, default=None, help="Square render window size, e.g. 800 or 512.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved targets without rendering.")

    eps_group = parser.add_mutually_exclusive_group()
    eps_group.add_argument("--save-eps", dest="save_eps", action="store_true")
    eps_group.add_argument("--no-save-eps", dest="save_eps", action="store_false")
    parser.set_defaults(save_eps=None)

    png_group = parser.add_mutually_exclusive_group()
    png_group.add_argument("--save-png", dest="save_png", action="store_true")
    png_group.add_argument("--no-save-png", dest="save_png", action="store_false")
    parser.set_defaults(save_png=None)

    pdf_group = parser.add_mutually_exclusive_group()
    pdf_group.add_argument("--save-pdf", dest="save_pdf", action="store_true")
    pdf_group.add_argument("--no-save-pdf", dest="save_pdf", action="store_false")
    parser.set_defaults(save_pdf=None)

    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument("--show-edges", dest="show_edges", action="store_true")
    edge_group.add_argument("--no-edges", dest="show_edges", action="store_false")
    parser.set_defaults(show_edges=None)

    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument("--draw-context-voxels", dest="draw_context_voxels", action="store_true")
    context_group.add_argument("--hide-context-voxels", dest="draw_context_voxels", action="store_false")
    parser.set_defaults(draw_context_voxels=None)

    interleave_group = parser.add_mutually_exclusive_group()
    interleave_group.add_argument("--paper-frame-interleave", dest="paper_frame_interleave", action="store_true")
    interleave_group.add_argument("--no-paper-frame-interleave", dest="paper_frame_interleave", action="store_false")
    parser.set_defaults(paper_frame_interleave=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(_load_config(args.config), args)

    # The fast renderer is serial by design. Ignore any legacy Ray setting in config.yaml.
    cfg.setdefault("renderer", {})["use_ray"] = False

    grid_config = _build_grid_config(cfg)
    action_table = build_action_table(side_length=grid_config["side_length"])

    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    root_folders = _build_root_folders(cfg)
    if args.dry_run:
        print("[dry-run] target root folders:")
        for root_folder in root_folders:
            print(f"  - {root_folder}")
        return

    for root_folder in root_folders:
        print(f"root_folder: {root_folder}")
        render_root_folder_fast(
            root_folder=root_folder,
            cfg=cfg,
            grid_config=grid_config,
            action_table=action_table,
        )


if __name__ == "__main__":
    main()
