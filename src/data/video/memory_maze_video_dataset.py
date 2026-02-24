import io
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from internetarchive import download
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from src.data.video.base_video import BaseAdvancedVideoDataset, BaseSimpleVideoDataset


def process_abs_cond(external_cond: torch.Tensor) -> None:
    # Split positions and directions
    pos = external_cond[:, :2]  # (t, 2)
    dirs = external_cond[:, 2:]  # (t, 2)

    # 1) Translate so the first position is at the origin
    pos -= pos[0]  # in-place OK; comment out if you need the original intact

    # 2) Build a rotation matrix that sends the first direction to (1, 0)
    dx, dy = dirs[0]  # first unit vector
    rot_mat = torch.tensor(
        [[dx, dy], [-dy, dx]],
        dtype=external_cond.dtype,
        device=external_cond.device,
    )  # shape (2, 2)

    # 3) Rotate every vector
    pos = pos @ rot_mat.T  # (t, 2)
    dirs = dirs @ rot_mat.T  # (t, 2)

    # 4) Re-normalise directions (covers numerical noise)
    dirs = F.normalize(dirs, dim=-1)

    external_cond = torch.cat([pos, dirs], dim=-1)
    return external_cond


def _load_npz_metadata(path: Path, fps: float = 30.0):
    try:
        with np.load(path) as data:
            frames = data["image"]
            num_frames = frames.shape[0]
            pts = np.arange(num_frames)
        return {
            "path": path,
            "pts": torch.as_tensor(pts, dtype=torch.long),
            "fps": fps,
        }
    except Exception as e:
        return {"error": f"{path}: {e}"}


class MemoryMazeVideoDataset(BaseAdvancedVideoDataset):
    """
    DMLab dataset in the advanced format used for memory-based video models.
    """

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str = "",
        split: str = "training",
        current_epoch: Optional[int] = None,
        latent_enable: bool = False,
        latent_type: str = "pre_sample",
        external_cond_dim: int = 3,
        external_cond_stack: bool = True,
        max_frames: int = 300,
        n_frames: int = 17,
        frame_skip: int = 2,
        filter_min_len: Optional[int] = None,
        subdataset_size: Optional[int] = None,
        num_eval_videos: Optional[int] = None,
        absolute_action: bool = False,
        both_actions: bool = False,
        **kwargs,
    ):
        if split == "test":
            split = "validation"
        self.absolute_action = absolute_action
        self.both_actions = both_actions

        super().__init__(
            save_dir=save_dir,
            resolution=resolution,
            latent_downsampling_factor=latent_downsampling_factor,
            latent_suffix=latent_suffix,
            split=split,
            current_epoch=current_epoch,
            latent_enable=latent_enable,
            latent_type=latent_type,
            external_cond_dim=external_cond_dim,
            external_cond_stack=external_cond_stack,
            max_frames=max_frames,
            n_frames=n_frames,
            frame_skip=frame_skip,
            filter_min_len=filter_min_len,
            subdataset_size=subdataset_size,
            num_eval_videos=num_eval_videos,
            **kwargs,
        )

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        return 1001

    def build_transform(self):
        return transforms.Resize(
            self.resolution,
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )

    def download_dataset(self):
        pass

    def build_metadata(self, split, assumed_fps: float = 30.0, num_workers: int = 64) -> None:
        """
        Build metadata using multiprocessing. Each .npz file is assumed to contain 'video'.
        """
        data_dir = Path(self.save_dir) / split
        paths = sorted(data_dir.glob("**/*.npz"), key=lambda x: x.name)

        video_pts: List[torch.Tensor] = []
        video_fps: List[float] = []
        video_paths: List[Path] = []

        load_fn = partial(_load_npz_metadata, fps=assumed_fps)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(load_fn, paths),
                    total=len(paths),
                    desc=f"Building metadata for {split}",
                )
            )

        for result in results:
            if "error" in result:
                print(f"Failed to load {result['error']}")
                continue
            video_paths.append(result["path"])
            video_pts.append(result["pts"])
            video_fps.append(result["fps"])

        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def get_video_metadata(self, split: str) -> List[Dict[str, Any]]:
        data_dir = Path(self.save_dir) / split
        paths = sorted(list(data_dir.glob("**/*.npz")), key=lambda x: x.name)
        metadata = [{"video_paths": p} for p in paths]
        return metadata

    def load_video(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"]
        data = np.load(path)
        video = data["image"][start_frame:end_frame]  # (t, h, w, 3)
        video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2)  # (t, c, h, w)
        return video

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"]
        data = np.load(path)
        if not self.absolute_action or self.both_actions:
            rel_actions = data["action"][start_frame:end_frame]  # (t, 3)
            rel_actions = torch.tensor(rel_actions).float()
            if not self.both_actions:
                return rel_actions

        if self.absolute_action or self.both_actions:
            agent_pos = data["agent_pos"][start_frame:end_frame]  # (t, 2)
            agent_dir = data["agent_dir"][start_frame:end_frame]  # (t, 2)
            abs_actions = np.concatenate([agent_pos, agent_dir], axis=-1)  # (t, 2 + 2)
            abs_actions = torch.tensor(abs_actions).float()  # (t, 4)
            if not self.both_actions:
                return abs_actions

        if self.both_actions:
            return torch.cat([rel_actions, abs_actions], dim=-1)

    def _process_external_cond(self, external_cond: torch.Tensor) -> torch.Tensor:
        """
        Re-anchor a (t, 4) tensor of 2-D positions + unit-direction vectors.
        After the transform:
        • external_cond[0, :2] == (0, 0)
        • external_cond[0, 2:] == (1, 0)

        Args
        ----
        external_cond : (t, 4) tensor
            [:, :2] = positions (x, y)
            [:, 2:] = unit vectors (dx, dy)

        Returns
        -------
        Tensor with the same shape, translated and rotated.
        """
        if not self.absolute_action and not self.both_actions:
            return super()._process_external_cond(external_cond)

        if self.both_actions:
            rel_external_cond = external_cond[:, : self.external_cond_dim // self.frame_skip]
            abs_external_cond = external_cond[:, self.external_cond_dim // self.frame_skip :]
            # rel_external_cond, abs_external_cond = external_cond.split(self.external_cond_dim, dim=-1)
            rel_external_cond = super()._process_external_cond(rel_external_cond)
            # abs_external_cond = process_abs_cond(abs_external_cond)
            abs_external_cond = super()._process_external_cond(abs_external_cond)
            return torch.cat([rel_external_cond, abs_external_cond], dim=-1)

        abs_external_cond = process_abs_cond(external_cond)
        return super()._process_external_cond(external_cond)

    # def stack_external_cond(self, external_cond: torch.Tensor) -> torch.Tensor:
    # """
    # Post-process external condition to align with frame skipping.

    # By default:
    # - We shift the condition by (frame_skip - 1) frames,
    # - then flatten them in blocks of size `frame_skip`.
    # """
    # fs = self.frame_skip
    # if fs == 1:
    # return external_cond
    ## pad front so we have condition for each newly-sparse frame
    # external_cond = F.pad(external_cond, (0, 0, fs - 1, 0), value=0.0)
    ## rearrange from (T*fs, D) -> (T, fs*D)
    # return rearrange(external_cond, "(t fs) d -> t (fs d)", fs=fs)

    def video_metadata_to_latent_path(self, video_metadata: Dict[str, Any]) -> Path:
        """
        Map <save_dir>/<split>/.../foo.npz → <latent_dir>/<split>/.../foo.pt
        """
        npz_path: Path = video_metadata["video_paths"]
        rel = npz_path.relative_to(self.save_dir)
        return (self.latent_dir / rel).with_suffix(".pt")


class MemoryMazeSimpleVideoDataset(BaseSimpleVideoDataset):
    """
    Simple whole-clip loader for Memory-Maze NPZ files.

    - Expects files like .../<split>/**/*.npz with key 'image' shaped (T, H, W, 3)
    - Returns (T, C, H, W) float32 in [0, 1]
    - No external conditions (this is the "simple" variant)
    - Latent path mirrors NPZ path under latent_dir with .pt extension
    """

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str = "",
        split: str = "training",
        **kwargs: Any,
    ) -> None:
        # Treat "test" as "validation" to be consistent with your other datasets
        if split == "test":
            split = "validation"

        super().__init__(
            save_dir=save_dir,
            resolution=resolution,
            latent_downsampling_factor=latent_downsampling_factor,
            latent_suffix=latent_suffix,
            split=split,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # BaseVideoDataset hooks
    # ------------------------------------------------------------------

    def build_transform(self):
        # Resize only; rest of pipeline handles normalization/augmentations if needed
        return transforms.Resize(
            self.resolution,
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )

    def get_video_metadata(self, split: str) -> List[Dict[str, Any]]:
        """
        Scan <save_dir>/<split> for .npz files and return a list-of-dicts.
        """
        data_dir = Path(self.save_dir) / split  # self.save_dir provided by base
        paths = sorted(data_dir.glob("**/*.npz"), key=lambda p: p.name)
        return [{"video_paths": p} for p in paths]

    def video_metadata_to_latent_path(self, video_metadata: Dict[str, Any]) -> Path:
        """
        Map <save_dir>/<split>/.../foo.npz → <latent_dir>/<split>/.../foo.pt
        """
        npz_path: Path = video_metadata["video_paths"]
        rel = npz_path.relative_to(self.save_dir)
        return (self.latent_dir / rel).with_suffix(".pt")

    # ------------------------------------------------------------------
    # Actual clip loading
    # ------------------------------------------------------------------

    def load_video(self, video_metadata: Dict[str, Any], start_frame: int = 0) -> torch.Tensor:
        """
        Load the entire clip from 'image' key and return (T, C, H, W) in [0, 1].
        """
        path: Path = video_metadata["video_paths"]
        with np.load(path) as data:
            # (T, H, W, 3), uint8 expected
            img = data["image"]
        # Slice from start_frame to end (simple dataset loads full clip)
        img = img[start_frame:]  # (T, H, W, 3)
        vid = torch.from_numpy(img).float().div_(255.0).permute(0, 3, 1, 2)  # (T, C, H, W)
        return vid

    def download_dataset(self):
        # No-op (same as your other simple datasets)
        pass


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ds = MemoryMazeSimpleVideoDataset(
        save_dir="/data/cvg/sebastian/memory_maze/memory-maze-9x9",
        resolution=64,
        latent_downsampling_factor=[1, 1],
        split="training",
    )
    print(f"Found {len(ds)} videos")
    x, latent_path = ds[0]
    print("Video:", x.shape, "→ latent path:", latent_path)

if __name__ == "__main__":
    dataset = MemoryMazeVideoDataset(
        save_dir="/data/cvg/sebastian/memory_maze/memory-maze-9x9",
        resolution=64,
        max_frames=20,
        absolute_action=True,
        split="test",
        latent_downsampling_factor=[1, 1],
    )
    sample = dataset[0]
    print("Sample loaded.")

    path = "/data/cvg/sebastian/dmlab/validation/0/208.npz"
    path = "/data/cvg/sebastian/memory_maze/memory-maze-9x9/train/20220923T132217-1000.npz"
    import numpy as np

    sample = np.load(path)
    # image (1001, 64, 64, 3)
    # target_color (1001, 3)
    # agent_pos (1001, 2)
    # agent_dir (1001, 2)
    # targets_vec (1001, 3, 2)
    # targets_pos (1001, 3, 2)
    # target_vec (1001, 2)
    # target_pos (1001, 2)
    # maze_layout (1001, 9, 9)
    # action (1001, 6)
    # reward (1001,)
    # terminal (1001,)
    # reset (1001,)

    print(sample.keys())
