from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PIL import Image
from scipy.spatial.transform import Rotation

try:
    from .voxel_handlers_min import PvBoxArray
except ImportError:  # pragma: no cover - allows running this file directly.
    from voxel_handlers_min import PvBoxArray


pv.global_theme.allow_empty_mesh = True


def get_transformation_matrix(translation: Any, rotation: Any) -> np.ndarray:
    """Create a 4x4 homogeneous transform from xyz Euler angles in degrees."""
    rot = Rotation.from_euler("xyz", rotation, degrees=True)
    rot_matrix = rot.as_matrix()

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = np.asarray(translation, dtype=float)[:3]
    return matrix


def get_rotated_mesh(mesh: pv.DataSet, rotation: Any) -> pv.DataSet:
    """Rotate a mesh around its own center."""
    origin = np.asarray(mesh.center, dtype=float)
    centered = mesh.translate(-origin, inplace=False)
    homo_matrix = get_transformation_matrix([0.0, 0.0, 0.0], rotation=rotation)
    rotated = centered.transform(homo_matrix, inplace=False)
    return rotated.translate(origin, inplace=False)


def _make_cutting_plane(
    grid_config: dict[str, Any],
    action_idx: int,
    action_table: dict[int, dict[str, int | str]],
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


def _add_camera_bounds_anchor(plotter: pv.Plotter, bounds: tuple[float, ...]) -> None:
    """Add one invisible mesh actor to stabilize camera bounds."""
    plotter.add_mesh(
        pv.Box(bounds=bounds),
        color=[0.0, 0.0, 0.0],
        opacity=1e-10,
        show_edges=False,
    )


def _add_voxel_meshes(
    *,
    plotter: pv.Plotter,
    nearby_cells: dict[str, pv.DataSet],
    updated_colors: np.ndarray,
    fast_bounds_anchor: bool,
) -> None:
    """Add voxel cubes to a PyVista plotter using the original visual style.

    When fast_bounds_anchor=False, the historical behavior is preserved: every
    voxel cell gets one almost invisible mesh actor for camera-bound stability.
    This is safe but slow.

    When fast_bounds_anchor=True, visible voxels are unchanged, but the invisible
    per-cell anchor actors are skipped. Camera bounds should be stabilized by one
    global invisible bounding box instead.
    """
    black_lower = np.asarray([0.0, 0.0, 0.0])
    black_upper = np.asarray([0.5, 0.5, 0.5])
    gray_upper = np.asarray([1.3, 1.3, 1.3])

    for elements in nearby_cells:
        voxel_idx = int(elements)
        color = updated_colors[voxel_idx]
        cell = nearby_cells[elements]

        # Original behavior:
        # - Near-black voxels are treated as removed and are not visibly rendered.
        # - White/gray voxels are drawn as faint context.
        # - Colored voxels are drawn as solid remaining material.
        if np.all(color >= black_lower) and np.all(color < black_upper):
            pass
        else:
            if np.all(color >= black_upper) and np.all(color < gray_upper):
                plotter.add_mesh(
                    cell,
                    style="wireframe",
                    opacity=0.001,
                    show_edges=True,
                    edge_opacity=0.01,
                    color=[0.8, 0.8, 0.8],
                )
                plotter.add_mesh(
                    cell,
                    color=color,
                    opacity=0.1,
                    show_edges=True,
                )
            else:
                plotter.add_mesh(
                    cell,
                    color=color,
                    opacity=0.9,
                    show_edges=True,
                )

        if not fast_bounds_anchor:
            # Historical camera-bound anchor. Almost invisible, but expensive
            # because it adds one actor per voxel.
            plotter.add_mesh(
                cell,
                color=color,
                opacity=1e-10,
                show_edges=True,
            )


def _configure_camera(plotter: pv.Plotter, grid_config: dict[str, Any]) -> None:
    bounds = tuple(float(v) for v in grid_config["bounds"])
    center = (
        (bounds[0] + bounds[1]) / 2.0,
        (bounds[2] + bounds[3]) / 2.0,
        (bounds[4] + bounds[5]) / 2.0,
    )

    plotter.set_focus(center)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.1
    plotter.camera.position = (0.3, 0.55, 0.3)
    plotter.camera.up = (0.0, 0.0, 1.0)


def _save_vector_graphics(
    *,
    plotter: pv.Plotter,
    save_path: Path,
    frame_index: int,
    save_eps: bool,
    save_pdf: bool,
) -> None:
    """Save optional vector-like frame exports.

    PyVista uses VTK/GL2PS for these formats. PDF is usually easier to handle
    than EPS in PowerPoint, but both can still depend on the local VTK build.
    """
    if not (save_eps or save_pdf):
        return

    save_path.mkdir(parents=True, exist_ok=True)

    if save_eps:
        plotter.save_graphic(str(save_path / f"screenshot_{frame_index}.eps"))

    if save_pdf:
        plotter.save_graphic(str(save_path / f"screenshot_{frame_index}.pdf"))


def one_step_voxel_render_for_cutting_process_local(
    *,
    k: int,
    grid_config: dict[str, Any],
    sample_images: np.ndarray,
    action: np.ndarray,
    action_table: dict[int, dict[str, int | str]],
    save_path: str | Path,
    save_eps: bool = False,
    save_png: bool = False,
    save_pdf: bool = False,
    fast_bounds_anchor: bool = False,
) -> Image.Image:
    """Render one cutting-process frame.

    This is the standalone version of the original PyVista worker. It returns a
    PIL image so the caller can compose all frames into GIF/MP4. Optional EPS/PDF
    exports use ``plotter.save_graphic``. Optional PNG export uses the rendered
    screenshot image and is useful for debugging individual frames.
    """
    save_path = Path(save_path)
    bounds = tuple(float(v) for v in grid_config["bounds"])

    tmp_mesh = pv.Box(bounds=bounds)

    frame_index = int(k)
    action_idx = int(action[frame_index])
    cutting_plane = _make_cutting_plane(
        grid_config=grid_config,
        action_idx=action_idx,
        action_table=action_table,
    )

    box_array_handler = PvBoxArray(grid_config=grid_config)
    _ = box_array_handler.cast_mesh_to_box_array(mesh=tmp_mesh.copy())
    box_arrays_data = box_array_handler.get_box_array_data()
    nearby_cells = box_arrays_data.boxes

    plotter = pv.Plotter(window_size=(800, 800), off_screen=True)

    try:
        step_image = sample_images[frame_index] / 255.0
        step_image = np.clip(step_image, 0.0, 1.0)
        updated_colors = box_array_handler.cast_2d_image_to_box_color(
            image=step_image,
            permute="z",
        )

        if fast_bounds_anchor:
            _add_camera_bounds_anchor(plotter, bounds)

        _add_voxel_meshes(
            plotter=plotter,
            nearby_cells=nearby_cells,
            updated_colors=updated_colors,
            fast_bounds_anchor=fast_bounds_anchor,
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

        _configure_camera(plotter, grid_config)

        _save_vector_graphics(
            plotter=plotter,
            save_path=save_path,
            frame_index=frame_index,
            save_eps=save_eps,
            save_pdf=save_pdf,
        )

        image = plotter.screenshot()
        pil_image = Image.fromarray(np.asarray(image))

        if save_png:
            save_path.mkdir(parents=True, exist_ok=True)
            pil_image.save(save_path / f"screenshot_{frame_index}.png")

        return pil_image
    finally:
        plotter.close()
