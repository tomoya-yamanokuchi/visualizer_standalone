from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from .cutting_process_render_worker import one_step_voxel_render_for_cutting_process_local
except ImportError:  # pragma: no cover - allows running this file directly.
    from cutting_process_render_worker import one_step_voxel_render_for_cutting_process_local


def _normalize_max_in_flight(max_in_flight: Optional[int], length: int) -> int:
    if length <= 0:
        raise ValueError("sample_images is empty; cannot render cutting process")

    if max_in_flight is None:
        return length

    return max(1, min(int(max_in_flight), length))


def _save_gif(
    frames: list[Image.Image],
    output_path: Path,
    *,
    duration_ms: int = 500,
) -> None:
    if not frames:
        raise ValueError("frames is empty; cannot save GIF")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )


def _render_cutting_process_serial(
    *,
    save_path: Path,
    grid_config: dict[str, Any],
    action: np.ndarray,
    action_table: dict[int, dict[str, int | str]],
    sample_images: np.ndarray,
    save_eps: bool,
) -> list[Image.Image]:
    frames: list[Image.Image] = []

    for k in tqdm(range(sample_images.shape[0]), desc="render frames", leave=False):
        frame = one_step_voxel_render_for_cutting_process_local(
            k=k,
            grid_config=grid_config,
            sample_images=sample_images,
            action=action,
            action_table=action_table,
            save_path=save_path,
            save_eps=save_eps,
        )
        frames.append(frame)

    return frames


def _render_cutting_process_ray(
    *,
    save_path: Path,
    grid_config: dict[str, Any],
    action: np.ndarray,
    action_table: dict[int, dict[str, int | str]],
    sample_images: np.ndarray,
    max_in_flight: int,
    save_eps: bool,
) -> list[Image.Image]:
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("use_ray=True requires the optional dependency 'ray'.") from exc

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    remote_render_one_frame = ray.remote(one_step_voxel_render_for_cutting_process_local)

    sample_images_ref = ray.put(sample_images)
    grid_config_ref = ray.put(grid_config)
    action_ref = ray.put(action)
    action_table_ref = ray.put(action_table)
    save_path_ref = ray.put(save_path)

    length = sample_images.shape[0]
    frames: list[Image.Image] = []

    for start in tqdm(range(0, length, max_in_flight), desc="render frame batches", leave=False):
        end = min(start + max_in_flight, length)
        result_refs = [
            remote_render_one_frame.remote(
                k=k,
                grid_config=grid_config_ref,
                sample_images=sample_images_ref,
                action=action_ref,
                action_table=action_table_ref,
                save_path=save_path_ref,
                save_eps=save_eps,
            )
            for k in range(start, end)
        ]
        frames.extend(ray.get(result_refs))

    return frames


def render_cutting_process_gif(
    *,
    save_path: str | Path,
    grid_config: dict[str, Any],
    action: np.ndarray,
    action_table: dict[int, dict[str, int | str]],
    sample_images: np.ndarray,
    save_tag: str,
    use_ray: bool = False,
    max_in_flight: Optional[int] = None,
    save_eps: bool = False,
    gif_duration_ms: int = 500,
) -> Path:
    """Render a cutting process as a GIF.

    This function is the standalone replacement for
    ``pv_voxel_render_parallel.render_cutting_process_v3`` from the original
    repository. It keeps Ray optional so that the visualizer can run in a small
    local environment with only PyVista/VTK installed.

    Args:
        save_path: Directory used for optional per-frame files.
        grid_config: Voxel grid config with ``bounds`` and ``side_length``.
        action: Action index for each rendered frame.
        action_table: Mapping from action index to ``{"axis": ..., "loc": ...}``.
        sample_images: Frame array shaped ``(T, H, W, 3)``. Values may be either
            in 0-255 or already normalized; the worker follows the original
            behavior and divides by 255.
        save_tag: Suffix used in the output GIF filename.
        use_ray: If true, render frame batches with Ray.
        max_in_flight: Maximum number of Ray tasks submitted at once.
        save_eps: Save per-frame EPS files in addition to the GIF. This is often
            fragile in headless Docker environments, so the default is false.
        gif_duration_ms: GIF frame duration in milliseconds.

    Returns:
        Path to the generated GIF.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    sample_images = np.asarray(sample_images)
    action = np.asarray(action)

    length = int(sample_images.shape[0])
    max_in_flight_value = _normalize_max_in_flight(max_in_flight, length)

    if action.shape[0] < length:
        raise ValueError(
            f"action length must be >= number of frames: action={action.shape[0]}, frames={length}"
        )

    if use_ray:
        frames = _render_cutting_process_ray(
            save_path=save_path,
            grid_config=grid_config,
            action=action,
            action_table=action_table,
            sample_images=sample_images,
            max_in_flight=max_in_flight_value,
            save_eps=save_eps,
        )
    else:
        frames = _render_cutting_process_serial(
            save_path=save_path,
            grid_config=grid_config,
            action=action,
            action_table=action_table,
            sample_images=sample_images,
            save_eps=save_eps,
        )

    output_path = save_path.parent / f"cutting_process_{save_tag}.gif"
    _save_gif(frames, output_path, duration_ms=gif_duration_ms)
    print(f"save_gif: {output_path}")
    return output_path
