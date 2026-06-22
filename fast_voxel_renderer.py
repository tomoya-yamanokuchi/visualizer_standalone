from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pyvista as pv
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    from .voxel_handlers_min import PvBoxArray
except ImportError:  # pragma: no cover - allows running this file directly.
    from voxel_handlers_min import PvBoxArray


pv.global_theme.allow_empty_mesh = True


def get_transformation_matrix(translation: Any, rotation: Any) -> np.ndarray:
    rot = Rotation.from_euler("xyz", rotation, degrees=True)
    rot_matrix = rot.as_matrix()

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=float)[:3]
    return matrix


def get_rotated_mesh(mesh: pv.DataSet, rotation: Any) -> pv.DataSet:
    origin = np.asarray(mesh.center, dtype=float)
    centered = mesh.translate(-origin, inplace=False)
    homo_matrix = get_transformation_matrix([0.0, 0.0, 0.0], rotation=rotation)
    rotated = centered.transform(homo_matrix, inplace=False)
    return rotated.translate(origin, inplace=False)


def make_cutting_plane(
    grid_config: dict,
    action_idx: int,
    action_table: dict,
) -> pv.DataSet:
    action_info = action_table[int(action_idx)]
    action_axis = str(action_info["axis"])
    loc_idx = int(action_info["loc"])

    bounds = tuple(float(v) for v in grid_config["bounds"])
    side_length = int(grid_config["side_length"])

    action_pos_candidate = np.linspace(bounds[0], bounds[1], side_length)
    action_pos = float(action_pos_candidate[loc_idx])

    if action_axis == "z":
        cutting_plane_translation = np.asarray([0.0, 0.0, action_pos])
        cutting_plane_rotation = np.asarray([0.0, 0.0, 0.0])
    elif action_axis == "y":
        cutting_plane_translation = np.asarray([0.0, action_pos, 0.0])
        cutting_plane_rotation = np.asarray([90.0, 0.0, 0.0])
    elif action_axis == "x":
        cutting_plane_translation = np.asarray([action_pos, 0.0, 0.0])
        cutting_plane_rotation = np.asarray([0.0, 90.0, 0.0])
    else:
        raise ValueError(f"Unsupported action axis: {action_axis}")

    cutting_plane_base = pv.Box(
        bounds=(
            bounds[0] - 0.01,
            bounds[1] + 0.01,
            bounds[2] - 0.01,
            bounds[3] + 0.01,
            -0.0001,
            0.0001,
        )
    )
    translated = cutting_plane_base.translate(cutting_plane_translation, inplace=False)
    return get_rotated_mesh(translated, cutting_plane_rotation)


def configure_camera(plotter: pv.Plotter, grid_config: dict) -> None:
    bounds = tuple(float(v) for v in grid_config["bounds"])
    cube = pv.Cube(center=(bounds[0], bounds[0], bounds[0]))

    plotter.set_focus(cube.center)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.1
    plotter.camera.position = (0.3, 0.55, 0.3)
    plotter.camera.up = (0.0, 0.0, 1.0)


class CachedFastVoxelRenderer:
    """Experimental faster renderer for one episode.

    The baseline renderer rebuilds the voxel geometry for every frame and adds
    one nearly invisible mesh for every voxel to stabilize bounds. This renderer
    caches the voxel geometry once per episode and uses a single invisible bounds
    mesh instead.

    It intentionally keeps one Plotter per frame for robustness. Reusing VTK
    actors would be faster, but it is more fragile and may change rendering
    behavior across PyVista/VTK versions.
    """

    def __init__(
        self,
        grid_config: dict,
        *,
        window_size: Tuple[int, int] = (800, 800),
        show_edges: bool = True,
        draw_context_voxels: bool = True,
        removed_threshold: float = 0.5,
    ) -> None:
        self.grid_config = grid_config
        self.bounds = tuple(float(v) for v in grid_config["bounds"])
        self.window_size = window_size
        self.show_edges = bool(show_edges)
        self.draw_context_voxels = bool(draw_context_voxels)
        self.removed_threshold = float(removed_threshold)

        self.box_array_handler = PvBoxArray(grid_config=grid_config)
        box_arrays_data = self.box_array_handler.get_box_array_data()

        # Cache voxel cube geometry as a sorted list for fast color lookup.
        self.indexed_boxes = [
            (int(idx), mesh)
            for idx, mesh in sorted(
                box_arrays_data.boxes.items(),
                key=lambda item: int(item[0]),
            )
        ]
        self.grid_centers = box_arrays_data.grid_centers
        self.bounds_mesh = pv.Box(bounds=self.bounds)
        self._cutting_plane_cache = {}

    def _get_cutting_plane(self, action_idx: int, action_table: dict) -> pv.DataSet:
        action_idx = int(action_idx)
        if action_idx not in self._cutting_plane_cache:
            self._cutting_plane_cache[action_idx] = make_cutting_plane(
                grid_config=self.grid_config,
                action_idx=action_idx,
                action_table=action_table,
            )
        return self._cutting_plane_cache[action_idx]

    def _add_visible_voxels(self, plotter: pv.Plotter, updated_colors: np.ndarray) -> int:
        visible_count = 0

        for voxel_idx, cell in self.indexed_boxes:
            color = updated_colors[voxel_idx]

            # Removed or already cut voxels. Skip completely in fast mode.
            if np.all(color >= np.asarray([0.0, 0.0, 0.0])) and np.all(
                color < np.asarray([self.removed_threshold] * 3)
            ):
                continue

            # White/gray context voxels.
            is_context = np.all(color >= np.asarray([0.5, 0.5, 0.5])) and np.all(
                color < np.asarray([1.3, 1.3, 1.3])
            )
            if is_context and not self.draw_context_voxels:
                continue

            if is_context:
                plotter.add_mesh(
                    cell,
                    style="wireframe",
                    opacity=0.001,
                    show_edges=self.show_edges,
                    edge_opacity=0.01,
                    color=[0.8, 0.8, 0.8],
                )
                plotter.add_mesh(
                    cell,
                    color=color,
                    opacity=0.1,
                    show_edges=self.show_edges,
                )
            else:
                plotter.add_mesh(
                    cell,
                    color=color,
                    opacity=0.9,
                    show_edges=self.show_edges,
                )

            visible_count += 1

        return visible_count

    def render_frame(
        self,
        *,
        k: int,
        sample_images: np.ndarray,
        action: np.ndarray,
        action_table: dict,
        save_path: Path,
        save_eps: bool = False,
        save_png: bool = False,
        save_pdf: bool = False,
    ) -> Image.Image:
        frame_index = int(k)
        action_idx = int(action[frame_index])
        cutting_plane = self._get_cutting_plane(action_idx, action_table)

        plotter = pv.Plotter(window_size=self.window_size, off_screen=True)

        try:
            step_image = sample_images[frame_index] / 255.0
            step_image = np.clip(step_image, 0.0, 1.0)
            updated_colors = self.box_array_handler.cast_2d_image_to_box_color(
                image=step_image,
                permute="z",
            )

            self._add_visible_voxels(plotter=plotter, updated_colors=updated_colors)

            # One invisible bounds mesh replaces thousands of invisible per-voxel meshes.
            plotter.add_mesh(
                self.bounds_mesh,
                color=[1.0, 1.0, 1.0],
                opacity=1e-10,
                show_edges=False,
            )

            if frame_index % 2 == 0:
                plotter.add_mesh(
                    cutting_plane,
                    color=(226 / 255.0, 220 / 255.0, 222 / 255.0),
                    opacity=0.8,
                    show_edges=False,
                    diffuse=1.0,
                )
            else:
                plotter.add_mesh(
                    cutting_plane,
                    color=(0.7, 0.7, 0.0),
                    opacity=0.0,
                    show_edges=False,
                )

            configure_camera(plotter, self.grid_config)

            if save_eps or save_pdf:
                save_path.mkdir(parents=True, exist_ok=True)
                if save_eps:
                    plotter.save_graphic(str(save_path / f"screenshot_{frame_index}.eps"))
                if save_pdf:
                    plotter.save_graphic(str(save_path / f"screenshot_{frame_index}.pdf"))

            image = plotter.screenshot()
            pil_image = Image.fromarray(np.asarray(image))

            if save_png:
                save_path.mkdir(parents=True, exist_ok=True)
                pil_image.save(save_path / f"screenshot_{frame_index}.png")

            return pil_image
        finally:
            plotter.close()


def save_gif(frames: list, output_path: Path, duration_ms: int = 500) -> None:
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


def render_cutting_process_gif_fast(
    *,
    save_path: str,
    grid_config: dict,
    action: np.ndarray,
    action_table: dict,
    sample_images: np.ndarray,
    save_tag: str,
    save_eps: bool = False,
    save_png: bool = False,
    save_pdf: bool = False,
    show_edges: bool = True,
    draw_context_voxels: bool = True,
    window_size: Tuple[int, int] = (800, 800),
    gif_duration_ms: int = 500,
) -> Path:
    """Render a cutting process GIF with cached voxel geometry.

    This is an experimental serial renderer. It intentionally does not use Ray;
    the goal is to speed up each frame and keep PyVista/VTK behavior stable.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    sample_images = np.asarray(sample_images)
    action = np.asarray(action)

    if action.shape[0] < sample_images.shape[0]:
        raise ValueError(
            f"action length must be >= number of frames: action={action.shape[0]}, frames={sample_images.shape[0]}"
        )

    renderer = CachedFastVoxelRenderer(
        grid_config=grid_config,
        window_size=window_size,
        show_edges=show_edges,
        draw_context_voxels=draw_context_voxels,
    )

    frames = []
    progress = tqdm(
        range(sample_images.shape[0]),
        desc=f"fast render frames {save_tag}",
        unit="frame",
        leave=True,
        dynamic_ncols=True,
    )

    for k in progress:
        progress.set_postfix_str(f"frame={k}")
        frames.append(
            renderer.render_frame(
                k=k,
                sample_images=sample_images,
                action=action,
                action_table=action_table,
                save_path=save_path,
                save_eps=save_eps,
                save_png=save_png,
                save_pdf=save_pdf,
            )
        )

    output_path = save_path.parent / f"cutting_process_fast_{save_tag}.gif"
    print(f"saving GIF: {output_path}")
    save_gif(frames, output_path, duration_ms=gif_duration_ms)
    print(f"save_gif: {output_path}")
    return output_path
