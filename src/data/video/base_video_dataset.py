import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BaseVideoDataset(Dataset, ABC):
    """
    Base class for video datasets. Videos may be of variable length.

    Folder structure of each dataset:
    - [save_dir] (specified in config, e.g., data/phys101)
        - /[split] (one per split)
            - /data_folder_name (e.g., videos)
            metadata.json
    """

    def __init__(
        self,
        save_dir,
        resolution,
        external_cond_dim,
        n_frames,
        frame_skip,
        validation_multiplier,
        split="training",
    ):
        super().__init__()
        self.split = split
        self.resolution = resolution
        self.external_cond_dim = external_cond_dim
        self.n_frames = (
            n_frames * frame_skip
            if split == "training"
            else n_frames * frame_skip * validation_multiplier
        )
        self.frame_skip = frame_skip
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.split_dir = self.save_dir / f"{split}"

        self.metadata_path = self.save_dir / "metadata.json"

        if not self.metadata_path.exists():
            # Build dataset
            print(f"Creating dataset in {self.save_dir}...")
            # self.download_dataset()
            json.dump(
                {
                    "training": self.get_data_lengths("training"),
                    "validation": self.get_data_lengths("validation"),
                },
                open(self.metadata_path, "w"),
            )

        self.metadata = json.load(open(self.metadata_path))
        self.data_paths = self.get_data_paths(self.split)
        self.clips_per_video = np.clip(
            np.array(self.metadata[split]) - self.n_frames + 1, a_min=1, a_max=None
        ).astype(np.int32)
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)
        self.transform = transforms.Resize((self.resolution, self.resolution), antialias=True)

    @abstractmethod
    def download_dataset(self) -> Sequence[int]:
        """
        Download dataset from the internet and build it in save_dir

        Returns a list of video lengths
        """
        raise NotImplementedError

    @abstractmethod
    def get_data_paths(self, split):
        """Return a list of data paths (e.g. xxx.mp4) for a given split"""
        raise NotImplementedError

    def get_data_lengths(self, split):
        """Return a list of num_frames for each data path (e.g. xxx.mp4) for a given split"""
        lengths = []
        for path in self.get_data_paths(split):
            length = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT)
            print(length)
            lengths.append(length)
        return lengths

    def split_idx(self, idx):
        video_idx = np.argmax(self.cum_clips_per_video > idx)
        frame_idx = idx - np.pad(self.cum_clips_per_video, (1, 0))[video_idx]
        return video_idx, frame_idx

    @staticmethod
    def load_video(path: Path):
        """
        Load video from a path
        """
        cap = cv2.VideoCapture(str(path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()
        frames = np.stack(frames, dtype=np.uint8)
        return np.transpose(frames, (0, 3, 1, 2))

    def __len__(self):
        return self.clips_per_video.sum()

    def __getitem__(self, idx):
        idx = idx
        video_idx, frame_idx = self.split_idx(idx)
        video_path = self.data_paths[video_idx]
        video = self.load_video(video_path)[frame_idx : frame_idx + self.n_frames]
        pad_len = self.n_frames - len(video)
        nonterminal = np.ones(self.n_frames)
        if len(video) < self.n_frames:
            video = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
            nonterminal[-pad_len:] = 0
        video = torch.from_numpy(video / 256.0).float()
        video = self.transform(video)
        return video[:: self.frame_skip], nonterminal[:: self.frame_skip]
