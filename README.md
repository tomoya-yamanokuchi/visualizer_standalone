# visualizer_standalone

Standalone visualizer for rendering 3D cutting-process GIFs from saved rollout results.

This repository is intended to be separated from the original training/evaluation repository. It does **not** train diffusion models and does **not** load model weights. It only reads saved rollout data and renders the cutting process with PyVista/VTK.

## Purpose

The original project generates rollout results on a GPU server. This repository is for local visualization only:

```text
GPU server:
  train models
  run evaluation
  save rollout_data.pickle and oracle observation images

local machine:
  load saved rollout results
  render 3D voxel cutting process GIFs
  adjust camera/view/style for paper figures
```

This separation is useful because PyVista/VTK rendering often requires a working display or OpenGL context, while headless SSH/Docker GPU servers may not provide one.

## Expected input data

Each rendered episode should contain at least:

```text
episode_0/
  rollout_data.pickle
  oracle_obs_cast_z_axis0.png
```

The default directory layout is:

```text
root_folder/
  epsilon_greedy_00/
    Object_A/
      episode_0/
        rollout_data.pickle
        oracle_obs_cast_z_axis0.png
```

`rollout_data.pickle` is expected to contain:

```python
{
    "observations": ...,  # cutting process 2D maps
    "actions": ...,       # cutting action indices
}
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Minimal dependencies are PyVista/VTK, NumPy, Pillow, SciPy, PyYAML, and tqdm.

Ray is optional and disabled by default. Use serial rendering first.

## Quick start

Edit `config.yaml` so that `root_folder` points to your local evaluation result directory.

Then run a dry run first:

```bash
python3 render_cutting_process.py --config config.yaml --dry-run
```

If the target folder is correct, render one episode:

```bash
python3 render_cutting_process.py \
  --config config.yaml \
  --no-ray \
  --no-save-eps \
  --max-episodes 1
```

The output GIF is saved next to the per-episode render directory, for example:

```text
Object_A/episode_0/cutting_process_dim_16_no_axis_w_cutting_plane3.gif
```

or, depending on `save_subdir`, one level above the frame-output directory.

## Configuration

Main fields in `config.yaml`:

```yaml
root_folder: /path/to/eval/result/root

tags:
  - epsilon_greedy_00

save_prefix: no_axis_w_cutting_plane3
save_subdir: 3d_cutting_process

rollout_filename: rollout_data.pickle
oracle_obs_filename: oracle_obs_cast_z_axis0.png

dim_2d: 64
dim_3d: 16
bounds: [-0.05, 0.05, -0.05, 0.05, -0.05, 0.05]

model_type_indices: null
episode_indices: null
max_episodes: 1

paper_frame_interleave: true

renderer:
  use_ray: false
  max_in_flight: 1
  save_eps: false
```

### Selection options

Render only specific model-type folders:

```bash
python3 render_cutting_process.py --config config.yaml --model-type-indices 0,2
```

Render only specific episodes:

```bash
python3 render_cutting_process.py --config config.yaml --episode-indices 0,3,5
```

Override the root folder from CLI:

```bash
python3 render_cutting_process.py \
  --config config.yaml \
  --root-folder /path/to/eval/root \
  --tags epsilon_greedy_00
```

## Rendering progress

The renderer shows progress bars with `tqdm`, including frame-level progress:

```text
render frames dim_16_no_axis_w_cutting_plane3:  42%|....| 8/19 [00:34<00:47, 4.31s/frame, frame=7]
```

Rendering can be slow because each frame builds and renders many voxel meshes with PyVista/VTK.

## Docker / DISPLAY notes

PyVista uses VTK and OpenGL. Even with `off_screen=True`, some VTK builds require a valid X server / OpenGL context.

If running inside Docker on a local Linux desktop, pass DISPLAY and X11 socket:

```bash
xhost +local:root

docker run -it --rm \
  --gpus all \
  --net=host \
  --shm-size=10g \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \
  -v /path/to/visualizer_standalone:/workspace/visualizer_standalone \
  -v /path/to/eval/results:/workspace/eval_results \
  your_image_name
```

A minimal PyVista check:

```bash
python3 - <<'PY'
import pyvista as pv

p = pv.Plotter(off_screen=True)
p.add_mesh(pv.Cube())
p.screenshot('/tmp/pyvista_test.png')
p.close()
print('saved /tmp/pyvista_test.png')
PY
```

If this fails with a DISPLAY or OpenGL error, fix the rendering environment before running the visualizer.

## Common troubleshooting

### `vtkXOpenGLRenderWindow: bad X server connection`

The container cannot access a valid display/OpenGL context. Run on a local machine with DISPLAY passed to Docker, or configure an off-screen rendering backend such as Xvfb/EGL/OSMesa depending on your VTK build.

### `ModuleNotFoundError`

Run commands from the repository root:

```bash
cd visualizer_standalone
python3 render_cutting_process.py --config config.yaml --dry-run
```

or ensure the current directory contains the Python files.

### Python 3.8 compatibility

This project is intended to run on Python 3.8+ and avoids Python 3.9-only type alias syntax in runtime-critical files.

## Repository structure

```text
visualizer_standalone/
  render_cutting_process.py          # CLI entry point
  voxel_renderer.py                  # GIF rendering orchestration
  cutting_process_render_worker.py   # PyVista rendering for one frame
  voxel_handlers_min.py              # minimal 2D map <-> 3D voxel utilities
  action_table.py                    # standalone cutting action table builder
  io_utils.py                        # minimal file/image utilities
  config.yaml                        # example/default config
  requirements.txt                   # minimal dependencies
```

## Recommended workflow

1. Generate evaluation rollout data on the GPU server.
2. Sync only the evaluation result folder to the local machine.
3. Edit `config.yaml` to point to the synced data.
4. Run `--dry-run`.
5. Render with `--no-ray --no-save-eps` first.
6. Adjust camera/style locally for paper figures.
