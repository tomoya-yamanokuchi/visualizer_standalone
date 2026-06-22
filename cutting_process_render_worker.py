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


def _add_voxel_meshes(
    *,
    plotter: pv.Plotter,
    nearby_cells: dict[str, pv.DataSet],
    updated_colors: np.ndarray,
) -> None:
    """Add voxel cubes to a PyVista plotter using the original visual style."""
    for elements in nearby_cells:
        voxel_idx = int(elements)
        color = updated_colors[voxel_idx]
        cell = nearby_cells[elements]

        # Original behavior:
        # - Near-black voxels are treated as removed and are not visibly rendered.
        # - White/gray voxels are drawn as faint context.
        # - Colored voxels are drawn as solid remaining material.
        if np.all(color >= np.asarray([0.0, 0.0, 0.0])) and np.all(
            color < np.asarray([0.5, 0.5, 0.5])
        ):
            pass
        else:
            if np.all(color >= np.asarray([0.5, 0.5, 0.5])) and np.all(
                color < np.asarray([1.3, 1.3, 1.3])
            ):
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

        # Add an almost invisible mesh for each cell to keep the camera bounds stable.
        plotter.add_mesh(
            cell,
            color=color,
            opacity=1e-10,
            show_edges=True,
        )


def _configure_camera(plotter: pv.Plotter, grid_config: dict[str, Any]) -> None:
    bounds = tuple(float(v) for v in grid_config["bounds"])
    cube = pv.Cube(center=(bounds[0], bounds[0], bounds[0]))

    plotter.set_focus(cube.center)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.1
    plotter.camera.position = (0.3, 0.55, 0.3)
    plotter.camera.up = (0.0, 0.0, 1.0)


def one_step_voxel_render_for_cutting_process_local(
    *,
    k: int,
    grid_config: dict[str, Any],
    sample_images: np.ndarray,
    action: np.ndarray,
    action_table: dict[int, dict[str, int | str]],
    save_path: str | Path,
    save_eps: bool = False,
) -> Image.Image:
    """Render one cutting-process frame.

    This is the standalone version of the original PyVista worker. It returns a
    PIL image so the caller can compose all frames into a GIF. ``save_eps`` is
    optional because EPS export is fragile in Docker/headless environments.
    """
    save_path = Path(save_path)
    bounds = tuple(float(v) for v in grid_config["bounds"])

    tmp_mesh = pv.Box(bounds=bounds)

    action_idx = int(action[int(k)])
    cutting_plane = _make_cutting_plane(
        grid_config=grid_config,
        action_idx=action_idx,
        action_table=action_table,
    )

    box_array_handler = PvBoxArray(grid_config=grid_config)
    _ = box_array_handler.cast_mesh_to_box_array(mesh=tmp_mesh.copy())
    box_arrays_data = box_array_handler.get_box_array_data()
    nearby_cells = box_arrays_data.boxes
    centers = box_arrays_data.grid_centers

    plotter = pv.Plotter(window_size=(800, 800), off_screen=True)

    try:
        step_image = sample_images[int(k)] / 255.0
        step_image = np.clip(step_image, 0.0, 1.0)
        updated_colors = box_array_handler.cast_2d_image_to_box_color(
            image=step_image,
            permute="z",
        )

        _add_voxel_meshes(
            plotter=plotter,
            nearby_cells=nearby_cells,
            updated_colors=updated_colors,
        )

        plotter.add_points(
            centers,
            render_points_as_spheres=True,
            color=[0, 0, 0],
            opacity=1e-10,
        )

        if int(k) % 2 == 0:
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

        if save_eps:
            save_path.mkdir(parents=True, exist_ok=True)
            plotter.save_graphic(str(save_path / f"screenshot_{int(k)}.eps"))

        image = plotter.screenshot()
        return Image.fromarray(np.asarray(image))
    finally:
        plotter.close()
