from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pyvista as pv


PermuteAxis = Literal["x", "y", "z"]


@dataclass
class BoxArrayData:
    """Container for voxel box meshes, grid centers, and per-voxel colors."""

    boxes: dict[str, pv.DataSet]
    grid_centers: np.ndarray
    colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        num_cells = int(self.grid_centers.shape[0])
        grid_3dim_float = np.cbrt(num_cells)
        grid_3dim = int(round(grid_3dim_float))

        if grid_3dim**3 != num_cells:
            raise ValueError(
                f"number of grid centers must be a perfect cube, got {num_cells}"
            )

        # The original code flattens a cubic 3D grid into a square 2D map.
        grid_2dim_float = np.sqrt(num_cells)
        grid_2dim = int(round(grid_2dim_float))

        if grid_2dim**2 != num_cells:
            raise ValueError(
                f"number of grid centers must also be a perfect square, got {num_cells}"
            )

        self.grid_2dim_size = (grid_2dim, grid_2dim)
        self.grid_3dim_size = (grid_3dim, grid_3dim, grid_3dim)

    def set_colors(self, colors: np.ndarray) -> None:
        self.colors = np.asarray(colors)

    def get_grid_centers(self) -> np.ndarray:
        return self.grid_centers

    def get_boxes(self) -> dict[str, pv.DataSet]:
        return self.boxes

    def get_colors(self) -> np.ndarray:
        if self.colors is None:
            raise ValueError("colors have not been set")
        return self.colors


class PvBoxArray:
    """Minimal standalone version of ``pv_box_array`` from the main repository.

    The class builds a regular 3D voxel grid and provides conversion helpers
    between the repository's tiled 2D RGB representation and flat per-voxel RGB
    colors used for PyVista rendering.
    """

    def __init__(self, grid_config: dict[str, Any]) -> None:
        self.grid_bounds = tuple(float(v) for v in grid_config["bounds"])
        self.grid_side_len = int(grid_config["side_length"])

        if len(self.grid_bounds) != 6:
            raise ValueError(f"bounds must contain 6 values, got {self.grid_bounds}")
        if self.grid_side_len <= 0:
            raise ValueError(f"side_length must be positive, got {self.grid_side_len}")

        density = (self.grid_bounds[1] - self.grid_bounds[0]) / self.grid_side_len
        if density <= 0:
            raise ValueError(f"invalid grid density computed from bounds: {density}")

        mesh = pv.Box(bounds=self.grid_bounds, level=6)

        # ``check_surface=False`` is a bit more robust across PyVista versions and
        # matches the newer multi-type handler in the original repository.
        self.grid = pv.voxelize(mesh, density=density, check_surface=False)
        self.grid_centers = np.asarray(self.grid.cell_centers().points)
        self._box_colors = np.zeros((self.grid.GetNumberOfCells(), 3), dtype=float) + 1.0

        box_array = self._create_box_array()
        self.box_array = BoxArrayData(boxes=box_array, grid_centers=self.grid_centers)
        self.box_array.set_colors(colors=self._box_colors)

        self.batch_image_map: dict[int, tuple[int, int]] = {}
        batch_img_len = int(self.box_array.grid_2dim_size[0] / self.box_array.grid_3dim_size[0])
        k = 0
        for i in range(batch_img_len):
            for j in range(batch_img_len):
                self.batch_image_map[k] = (i, j)
                k += 1

    def _create_box_array(self) -> dict[str, pv.DataSet]:
        if self.grid_centers.shape[0] < 2:
            raise ValueError("at least two grid centers are required to build voxel boxes")

        side_length = float(np.abs(self.grid_centers[1][0] - self.grid_centers[0][0]) / 2.1)
        base_cube = pv.Box(bounds=(-1, 1, -1, 1, -1, 1))
        base_cube = base_cube.scale([side_length, side_length, side_length], inplace=False)

        nearby_cells: dict[str, pv.DataSet] = {}
        for i, center in enumerate(self.grid_centers):
            nearby_cells[str(i)] = base_cube.translate(center, inplace=False)

        return nearby_cells

    def cast_mesh_to_box_array(self, mesh: pv.DataSet) -> np.ndarray:
        """Mark grid cells that overlap a given mesh.

        This preserves the original behavior used before rendering: cells close
        to the full bounding box mesh are initialized to a yellow-ish color.
        The later image-to-color cast overwrites the final colors, but keeping
        this step maintains compatibility with the original pipeline.
        """
        if self.grid_centers.shape[0] < 2:
            raise ValueError("at least two grid centers are required to cast mesh")

        vicinity_box_length = float(np.abs(self.grid_centers[0] - self.grid_centers[1]).max())
        side_length = vicinity_box_length / 2.0

        ugrid = pv.voxelize(mesh, density=side_length, check_surface=False)
        ugrid_cell_center = np.asarray(ugrid.cell_centers().points)

        if ugrid_cell_center.size > 0:
            idxs = self.grid.find_closest_cell(ugrid_cell_center)
            self._box_colors[idxs] = [0.8, 0.8, 0.1]

        self.box_array.set_colors(colors=self._box_colors)
        return self._box_colors

    def get_box_color_to_2d_image(
        self,
        box_color: np.ndarray | None = None,
        permute: PermuteAxis = "x",
    ) -> np.ndarray:
        box_arrays_data = self.get_box_array_data()
        grid_2dim = box_arrays_data.grid_2dim_size[0]
        grid_3dim = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim / grid_3dim)

        if box_color is None:
            box_color = self._box_colors
        box_color = np.asarray(box_color)

        batch_2d_image = box_color.reshape(grid_3dim, grid_3dim, grid_3dim, 3)
        cast_image = np.empty((grid_2dim, grid_2dim, 3), dtype=box_color.dtype)

        if permute == "z":
            batch_2d_image_perm = batch_2d_image
        elif permute == "y":
            batch_2d_image_perm = batch_2d_image.transpose(1, 0, 2, 3)
        elif permute == "x":
            batch_2d_image_perm = batch_2d_image.transpose(2, 1, 0, 3)
        else:
            raise ValueError(f"Unsupported permute value: {permute}")

        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                cast_image[
                    j * grid_3dim : (j + 1) * grid_3dim,
                    i * grid_3dim : (i + 1) * grid_3dim,
                ] = batch_2d_image_perm[k]
                k += 1

        return cast_image

    def cast_2d_image_to_box_color(
        self,
        image: np.ndarray,
        permute: PermuteAxis = "z",
    ) -> np.ndarray:
        box_arrays_data = self.get_box_array_data()
        grid_2dim = box_arrays_data.grid_2dim_size[0]
        grid_3dim = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim / grid_3dim)

        image = np.asarray(image)
        expected_shape = (grid_2dim, grid_2dim, 3)
        if image.shape != expected_shape:
            raise ValueError(f"image shape must be {expected_shape}, got {image.shape}")

        batch_2d_image_base = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3), dtype=image.dtype)

        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                batch_2d_image_base[k] = image[
                    j * grid_3dim : (j + 1) * grid_3dim,
                    i * grid_3dim : (i + 1) * grid_3dim,
                ]
                k += 1

        if permute == "z":
            batch_2d_image = batch_2d_image_base
        elif permute == "y":
            batch_2d_image = batch_2d_image_base.transpose(1, 0, 2, 3)
        elif permute == "x":
            batch_2d_image = batch_2d_image_base.transpose(2, 1, 0, 3)
        else:
            raise ValueError(f"Unsupported permute value: {permute}")

        self._box_colors = batch_2d_image.reshape(-1, 3)
        self.box_array.set_colors(colors=self._box_colors)
        return self.get_box_array_data().get_colors()

    def get_2d_image_to_mini_batch_image(
        self,
        image: np.ndarray,
        permute: PermuteAxis = "z",
    ) -> np.ndarray:
        """Return the 3D mini-batch image representation without updating colors."""
        box_arrays_data = self.get_box_array_data()
        grid_2dim = box_arrays_data.grid_2dim_size[0]
        grid_3dim = box_arrays_data.grid_3dim_size[0]
        batch_img_len = int(grid_2dim / grid_3dim)

        image = np.asarray(image)
        batch_2d_image_base = np.zeros((grid_3dim, grid_3dim, grid_3dim, 3), dtype=image.dtype)

        k = 0
        for j in range(batch_img_len):
            for i in range(batch_img_len):
                batch_2d_image_base[k] = image[
                    j * grid_3dim : (j + 1) * grid_3dim,
                    i * grid_3dim : (i + 1) * grid_3dim,
                ]
                k += 1

        if permute == "z":
            return batch_2d_image_base
        if permute == "y":
            return batch_2d_image_base.transpose(1, 0, 2, 3)
        if permute == "x":
            return batch_2d_image_base.transpose(2, 1, 0, 3)
        raise ValueError(f"Unsupported permute value: {permute}")

    def get_box_array_data(self) -> BoxArrayData:
        return self.box_array


# Backward-compatible aliases for snippets copied from the original repository.
box_array_data = BoxArrayData
pv_box_array = PvBoxArray
