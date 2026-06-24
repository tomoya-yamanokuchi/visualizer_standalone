from __future__ import annotations

import re
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


def _pil_to_rgb_array(frame: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB uint8 array for MP4 encoding."""
    if frame.mode == "RGB":
        return np.asarray(frame)

    # Composite transparency over black if an RGBA frame is ever returned.
    rgba = frame.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    composed = Image.alpha_composite(background, rgba)
    return np.asarray(composed.convert("RGB"))


def _save_mp4(
    frames: list[Image.Image],
    output_path: Path,
    *,
    fps: int = 30,
    duration_ms: int = 500,
    quality: int = 8,
) -> None:
    """Save PIL frames as a PowerPoint-compatible MP4 using imageio/ffmpeg.

    The renderer produces a small number of semantic frames. To preserve the same
    timing as GIF output, each frame is repeated for duration_ms at the requested
    fps.

    PowerPoint for Mac can be picky about MP4 metadata and pixel format. The
    writer settings below mirror the manual ffmpeg compatibility pass that was
    previously used after rendering:
      - H.264 via libx264
      - yuv420p pixel format
      - avc1 codec tag
      - main profile / level 4.0
      - faststart moov atom placement
      - no audio stream
    """
    if not frames:
        raise ValueError("frames is empty; cannot save MP4")

    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise RuntimeError(
            "MP4 export requires imageio with ffmpeg support. Install with: "
            "pip install 'imageio[ffmpeg]'"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = int(fps)
    repeat_count = max(1, int(round((float(duration_ms) / 1000.0) * fps)))

    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=int(quality),
        macro_block_size=16,
        pixelformat="yuv420p",
        output_params=[
            "-profile:v",
            "main",
            "-level",
            "4.0",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
            "-an",
        ],
    ) as writer:
        for frame in frames:
            frame_array = _pil_to_rgb_array(frame)
            for _ in range(repeat_count):
                writer.append_data(frame_array)


def _png_frame_index(path: Path) -> int:
    match = re.search(r"screenshot_(\d+)\.png$", path.name)
    if not match:
        raise ValueError(f"unexpected PNG frame name: {path.name}")
    return int(match.group(1))


def _load_png_frames(image_folder: Path) -> list[Image.Image]:
    frame_paths = sorted(image_folder.glob("screenshot_*.png"), key=_png_frame_index)
    if not frame_paths:
        raise FileNotFoundError(f"no screenshot_*.png frames found in: {image_folder}")

    frames: list[Image.Image] = []
    for frame_path in tqdm(frame_paths, desc=f"load PNG frames {image_folder.name}", unit="frame"):
        with Image.open(frame_path) as img:
            frames.append(img.convert("RGB").copy())

    return frames


def render_cutting_process_mp4_from_png(
    *,
    image_folder: str | Path,
    save_tag: str,
    gif_duration_ms: int = 500,
    mp4_fps: int = 30,
    mp4_quality: int = 8,
) -> Path:
    """Generate a PowerPoint-compatible MP4 from pre-rendered screenshot PNGs."""
    image_folder = Path(image_folder)
    frames = _load_png_frames(image_folder)

    output_path = image_folder.parent / f"cutting_process_{save_tag}.mp4"
    print(f"saving MP4 from PNG frames: {output_path}")
    _save_mp4(
        frames,
        output_path,
        fps=mp4_fps,
        duration_ms=gif_duration_ms,
        quality=mp4_quality,
    )
    print(f"save_mp4_from_png: {output_path}")
    return output_path


def _render_cutting_process_serial(
    *,
    save_path: Path,
    grid_config: dict[str, Any],
    action: np.ndarray,
    action_table: dict[int, dict[str, int | str]],
    sample_images: np.ndarray,
    save_eps: bool,
    save_png: bool,
    save_pdf: bool,
    fast_bounds_anchor: bool,
    progress_desc: str,
) -> list[Image.Image]:
    frames: list[Image.Image] = []

    frame_iter = tqdm(
        range(sample_images.shape[0]),
        desc=progress_desc,
        unit="frame",
        leave=True,
        dynamic_ncols=True,
    )

    for k in frame_iter:
        frame_iter.set_postfix_str(f"frame={k}")
        frame = one_step_voxel_render_for_cutting_process_local(
            k=k,
            grid_config=grid_config,
            sample_images=sample_images,
            action=action,
            action_table=action_table,
            save_path=save_path,
            save_eps=save_eps,
            save_png=save_png,
            save_pdf=save_pdf,
            fast_bounds_anchor=fast_bounds_anchor,
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
    save_png: bool,
    save_pdf: bool,
    fast_bounds_anchor: bool,
    progress_desc: str,
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

    progress = tqdm(
        total=length,
        desc=progress_desc,
        unit="frame",
        leave=True,
        dynamic_ncols=True,
    )

    try:
        for start in range(0, length, max_in_flight):
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
                    save_png=save_png,
                    save_pdf=save_pdf,
                    fast_bounds_anchor=fast_bounds_anchor,
                )
                for k in range(start, end)
            ]
            frames.extend(ray.get(result_refs))
            progress.update(end - start)
            progress.set_postfix_str(f"frames={end}/{length}")
    finally:
        progress.close()

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
    save_png: bool = False,
    save_pdf: bool = False,
    save_gif: bool = True,
    save_mp4: bool = False,
    gif_duration_ms: int = 500,
    mp4_fps: int = 30,
    mp4_quality: int = 8,
    fast_bounds_anchor: bool = False,
) -> dict[str, Path]:
    """Render a cutting process and save requested video formats.

    The historical function name is kept for backward compatibility. It can now
    save GIF, MP4, or both from the same rendered frames.
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

    progress_desc = f"render frames {save_tag}"

    if use_ray:
        frames = _render_cutting_process_ray(
            save_path=save_path,
            grid_config=grid_config,
            action=action,
            action_table=action_table,
            sample_images=sample_images,
            max_in_flight=max_in_flight_value,
            save_eps=save_eps,
            save_png=save_png,
            save_pdf=save_pdf,
            fast_bounds_anchor=fast_bounds_anchor,
            progress_desc=progress_desc,
        )
    else:
        frames = _render_cutting_process_serial(
            save_path=save_path,
            grid_config=grid_config,
            action=action,
            action_table=action_table,
            sample_images=sample_images,
            save_eps=save_eps,
            save_png=save_png,
            save_pdf=save_pdf,
            fast_bounds_anchor=fast_bounds_anchor,
            progress_desc=progress_desc,
        )

    output_paths: dict[str, Path] = {}

    if save_gif:
        output_path = save_path.parent / f"cutting_process_{save_tag}.gif"
        print(f"saving GIF: {output_path}")
        _save_gif(frames, output_path, duration_ms=gif_duration_ms)
        print(f"save_gif: {output_path}")
        output_paths["gif"] = output_path

    if save_mp4:
        output_path = save_path.parent / f"cutting_process_{save_tag}.mp4"
        print(f"saving MP4: {output_path}")
        _save_mp4(
            frames,
            output_path,
            fps=mp4_fps,
            duration_ms=gif_duration_ms,
            quality=mp4_quality,
        )
        print(f"save_mp4: {output_path}")
        output_paths["mp4"] = output_path

    if not output_paths:
        print("No GIF/MP4 output requested; rendered frames were only saved via save_png/save_eps/save_pdf options.")

    return output_paths
