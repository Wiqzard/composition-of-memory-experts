# Composition of Memory Experts for Diffusion World Models

Published as a conference paper at **ICLR 2026**.

**Authors:** Sebastian Stapf, Pablo Acuaviva Huertos, Aram Davtyan, Paolo Favaro  
Computer Vision Group, Department of Computer Science, University of Bern

[Project Page](https://wiqzard.github.io/composition-of-memory-experts/) | [Paper (PDF)](docs/paper.pdf)

## Title Figure

![Title figure](docs/assets/title-image.png)

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

## What Is Included

The retained core training implementation is centered around:
- `src/models/dfot_video.py`
- `src/models/diffusion_video.py`
- `src/models/mea_video.py`

The configuration set is reduced to the `m_video_maze` experiment and the config files required to instantiate it.

## Data

The default `m_video_maze` config expects Memory Maze data in:
- `data/memory-maze-9x9` (or update `configs/data/memory_maze.yaml` accordingly).

## Citation

```bibtex
@inproceedings{stapf2026come,
  title     = {Composition of Memory Experts for Diffusion World Models},
  author    = {Stapf, Sebastian and Acuaviva Huertos, Pablo and Davtyan, Aram and Favaro, Paolo},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
