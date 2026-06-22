from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_subdirs(root_path: str | Path) -> list[str]:
    """Return subdirectory names sorted in natural-ish order.

    The original implementation used ``natsort.natsorted``. To keep this
    standalone utility lightweight, this function uses a small built-in natural
    sort key that handles common names such as ``episode_2`` < ``episode_10``.
    """
    root_path = Path(root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"not a directory: {root_path}")

    return sorted(
        [p.name for p in root_path.iterdir() if p.is_dir()],
        key=_natural_sort_key,
    )


def _natural_sort_key(value: str) -> list[int | str]:
    """Small natural-sort key without depending on natsort."""
    import re

    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def load_pickle(path: str | Path) -> Any:
    """Load a pickle file with the standard library pickle module."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pickle file does not exist: {path}")

    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: str | Path) -> Path:
    """Save a pickle file. Mainly useful for small standalone tests."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(data, f)
    return path


def load_image_as_numpy(
    path: str | Path,
    resize: Optional[tuple[int, int]] = None,
    channel_type: Optional[str] = None,
) -> np.ndarray:
    """Load an image as a float RGB/RGBA/etc. NumPy array in the range 0-1.

    Args:
        path: Image path.
        resize: Optional ``(width, height)`` passed to Pillow.
        channel_type: Optional Pillow conversion mode, e.g. ``"RGB"``.

    Returns:
        NumPy array normalized to 0-1, matching the behavior of the original
        ``pil_image_load_to_numpy`` helper.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image file does not exist: {path}")

    image = Image.open(path)
    try:
        if channel_type is not None:
            image = image.convert(channel_type)
        if resize is not None:
            image = image.resize(resize)
        return np.asarray(image) / 255.0
    finally:
        image.close()


def numpy_to_pil(data: np.ndarray) -> Image.Image:
    """Convert a normalized or uint8 NumPy image to a Pillow image."""
    array = np.asarray(data)

    if np.issubdtype(array.dtype, np.floating):
        # Preserve the original helper behavior: float input is expected to be
        # normalized. Clip to avoid wrap-around when values are slightly outside
        # the range because of interpolation or arithmetic.
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array)


def save_numpy_image(data: np.ndarray, path: str | Path) -> Path:
    """Save a NumPy image, accepting either 0-1 float or 0-255 uint8 data."""
    path = Path(path)
    ensure_dir(path.parent)
    image = numpy_to_pil(data)
    image.save(path)
    return path


def iter_episode_dirs(root_folder: str | Path) -> Iterable[Path]:
    """Yield episode directories below ``root/tag_or_model/object/episode`` trees.

    This helper is not used by the current entrypoint but is convenient for
    standalone scripts and quick checks.
    """
    root_folder = Path(root_folder)
    for path in root_folder.rglob("episode_*"):
        if path.is_dir():
            yield path
