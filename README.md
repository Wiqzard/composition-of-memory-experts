# PhD Playground (Paper Release)

This repository has been slimmed down to the training codepath used for the video-memory experiments in the paper.

## Quickstart

1. Create and activate a Python environment (3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python train.py experiment=m_video_maze
```

## What is included

The retained core training implementation is centered around:
- `src/models/dfot_video.py`
- `src/models/diffusion_video.py`
- `src/models/mea_video.py`

The configuration set is reduced to the `m_video_maze` experiment and the config files required to instantiate it.

## Data

The default `m_video_maze` config expects Memory Maze data in:

- `data/memory-maze-9x9` (or update `configs/data/memory_maze.yaml` accordingly).

## Notes

- The project uses Hydra for configuration composition.
- The root entrypoint (`train.py`) forwards to `src/train.py`.
