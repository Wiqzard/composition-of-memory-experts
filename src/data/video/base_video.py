import bisect
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
from torchvision.datasets.video_utils import _collate_fn, _VideoTimestampsDataset
from tqdm import tqdm

from src.data.video.utils import VideoTransform, read_video
from src.utils.distributed_utils import rank_zero_print
from src.utils.print_utils import cyan

SPLIT = Literal["training", "validation", "test"]


class BaseVideoDataset(torch.utils.data.Dataset, ABC):
    """
    Common base class for video dataset.
    Methods here are shared between simple and advanced video datasets.

    Folder structure of each dataset:
    - save_dir (specified as argument)
        - /{split}
            - data files (e.g. 000001.mp4, 000001.pt)
        - /metadata
            - {split}.pt
    - Also a latent_dir of the form: {save_dir}_latent_{latent_resolution} (same structure).
    """

    _ALL_SPLITS: Tuple[SPLIT, ...] = ("training", "validation", "test")
    metadata: List[Dict[str, Any]]

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str = "",
        split: SPLIT = "training",
        **kwargs: Any,
    ):
        """
        Args:
            save_dir: Root directory containing dataset files.
            resolution: Spatial resolution (e.g. 256).
            latent_downsampling_factor: [temporal_down, spatial_down], used to compute latent_resolution.
            latent_suffix: Extra string appended to name of latent directory.
            split: Which data split ("training", "validation", "test").
        """
        super().__init__()
        self.split = split
        self.save_dir = Path(save_dir)

        self.resolution = resolution

        if isinstance(resolution, (list, tuple, ListConfig)):
            self.latent_resolution = tuple(
                r // f for r, f in zip(resolution, latent_downsampling_factor)
            )
        else:
            # Assume scalar resolution and use the second factor for division
            self.latent_resolution = resolution // latent_downsampling_factor[1]

        # Build a path for latents, e.g., mydata_latent_32_suffix
        suffix_str = f"_{latent_suffix}" if latent_suffix else ""
        self.latent_dir = self.save_dir.with_name(
            f"{self.save_dir.name}_latent_{self.latent_resolution}{suffix_str}"
        )

        self.split_dir = self.save_dir / split
        self.metadata_dir = self.save_dir / "metadata"

        # Download dataset if it does not exist
        if self._should_download():
            self.download_dataset()
        # if not self.metadata_dir.exists() or not (self.metadata_dir / f"{self.split}.pt").exists():
        if not (self.metadata_dir / f"{self.split}.pt").exists():
            self.metadata_dir.mkdir(exist_ok=True, parents=True)
            self.build_metadata(self.split)
            # for s in self._ALL_SPLITS:
            #    self.build_metadata(s)
        # self.build_metadata("validation")

        self.metadata = self.load_metadata()
        self.augment_dataset()
        self.transform = self.build_transform()

    def _should_download(self) -> bool:
        """Check if the dataset should be downloaded (dummy example)."""
        return not (self.save_dir / self.split).exists()

    @abstractmethod
    def download_dataset(self) -> None:
        """Download dataset from the internet and build it in save_dir."""
        raise NotImplementedError

    def build_metadata(self, split: SPLIT) -> None:
        """
        Build metadata for the dataset and save it in metadata_dir.
        Default example uses torchvision's _VideoTimestampsDataset to read frame pts.
        """
        video_paths = sorted(list((self.save_dir / split).glob("**/*.mp4")), key=str)
        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(video_paths),
            batch_size=16,
            num_workers=64,
            collate_fn=_collate_fn,
        )
        video_pts: List[torch.Tensor] = []
        video_fps: List[float] = []

        with tqdm(total=len(dl), desc=f"Building metadata for {split}") as pbar:
            for batch in dl:
                pbar.update(1)
                batch_pts, batch_fps = list(zip(*batch))
                batch_pts = [torch.as_tensor(pts, dtype=torch.long) for pts in batch_pts]
                video_pts.extend(batch_pts)
                video_fps.extend(batch_fps)

        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def subsample(
        self,
        metadata: List[Dict[str, Any]],
        filter_fn: Callable[[Dict[str, Any]], bool],
        filter_msg: str,
    ) -> List[Dict[str, Any]]:
        """
        Subsample the dataset with the given filter function.
        """
        before_len = len(metadata)
        metadata = [video_metadata for video_metadata in metadata if filter_fn(video_metadata)]
        after_len = len(metadata)
        rank_zero_print(
            cyan(
                f"{self.split}: {after_len} / {before_len} videos remain after filtering out {filter_msg}"
            )
        )
        return metadata

    def augment_dataset(self) -> None:
        """
        Hooks to augment the dataset if needed (default: None).
        """
        augmentation = self._build_data_augmentation()
        if augmentation is not None:
            fn, msg = augmentation
            self.metadata = self._augment_dataset(self.metadata, fn, msg)

    def _build_data_augmentation(
        self,
    ) -> Optional[Tuple[Callable[[Dict[str, Any]], List[Dict[str, Any]]], str]]:
        """Return (augment_fn, msg) or None if no augmentation."""
        return None

    def _augment_dataset(
        self,
        metadata: List[Dict[str, Any]],
        augment_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        augment_msg: str,
    ) -> List[Dict[str, Any]]:
        """Apply an augment_fn to each item of metadata and flatten the results."""
        before_len = len(metadata)
        metadata = [aug_item for video_md in metadata for aug_item in augment_fn(video_md)]
        after_len = len(metadata)
        rank_zero_print(
            cyan(
                f"{self.split}: {before_len} -> {after_len} videos after augmenting with {augment_msg}"
            )
        )
        return metadata

    def load_metadata(self) -> List[Dict[str, Any]]:
        """Load pre-saved metadata file for the current split."""
        metadata = torch.load(self.metadata_dir / f"{self.split}.pt", weights_only=False)
        return [
            {key: metadata[key][i] for key in metadata.keys()}
            for i in range(len(metadata["video_paths"]))
        ]

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        """Return the length (#frames) of the video."""
        return len(video_metadata["video_pts"])

    def build_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build a transform (e.g. resize) that is applied to each video frame."""
        if isinstance(self.resolution, (list, tuple, ListConfig)):
            return VideoTransform((self.resolution[0], self.resolution[1]))
        else:
            return VideoTransform((self.resolution, self.resolution))

    def video_metadata_to_latent_path(self, video_metadata: Dict[str, Any]) -> Path:
        """
        Convert a video path to the corresponding latent .pt path.
        Example: data/abc/training/000001.mp4 -> data/abc_latent_32/training/000001.pt
        """
        return (
            self.latent_dir / video_metadata["video_paths"].relative_to(self.save_dir)
        ).with_suffix(".pt")

    def get_latent_paths(self, split: SPLIT) -> List[Path]:
        """Return list of latent .pt files in the latent_dir for the given split."""
        return sorted(list((self.latent_dir / split).glob("**/*.pt")), key=str)

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load video frames from start_frame (inclusive) to end_frame (exclusive).
        Returns a Tensor of shape (T, C, H, W) in [0,1].
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        video_path = video_metadata["video_paths"]
        video_pts = video_metadata["video_pts"]
        start_pts = video_pts[start_frame].item()
        end_pts = video_pts[end_frame - 1].item()
        video = read_video(video_path, start_pts, end_pts)
        # (T, H, W, C) -> (T, C, H, W)
        return video.permute(0, 3, 1, 2) / 255.0


class BaseSimpleVideoDataset(BaseVideoDataset):
    """
    Base class for simple video datasets:
      - loads entire videos (all frames) at the given resolution
      - also provides path where latents can be saved
    """

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str = "",
        split: SPLIT = "training",
        **kwargs: Any,
    ):
        """
        Inherits from BaseVideoDataset and additionally ensures latents are saved
        only for videos that haven't been preprocessed yet.
        """
        super().__init__(
            save_dir=save_dir,
            resolution=resolution,
            latent_downsampling_factor=latent_downsampling_factor,
            latent_suffix=latent_suffix,
            split=split,
            **kwargs,
        )
        # Ensure latent_dir is created
        self.latent_dir.mkdir(exist_ok=True, parents=True)

        # Filter out videos that already have latents
        self.metadata = self.exclude_videos_with_latents(self.metadata)

    def exclude_videos_with_latents(self, metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Exclude videos that already have a .pt latent file."""
        latent_paths = set(self.get_latent_paths(self.split))
        return self.subsample(
            metadata,
            lambda m: self.video_metadata_to_latent_path(m) not in latent_paths,
            "videos that have already been preprocessed to latents",
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Loads the entire video, plus the path where the latent should be saved.
        """
        video_metadata = self.metadata[idx]
        video = self.load_video(video_metadata, 0)  # load from frame 0 to end
        return (
            self.transform(video),  # (T, C, H, W)
            self.video_metadata_to_latent_path(video_metadata).as_posix(),
        )


class BaseAdvancedVideoDataset(BaseVideoDataset):
    """
    Base class for more advanced video datasets that:
      - can load variable-length videos,
      - extract sub-clips,
      - apply frame skip,
      - handle subdataset slicing for large datasets,
      - optionally use preprocessed latents, etc.
    """

    cumulative_sizes: List[int]
    idx_remap: List[int]

    def __init__(
        self,
        save_dir: str,
        resolution: int,
        latent_downsampling_factor: List[int],
        latent_suffix: str = "",
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
        # "latent" group
        latent_enable: bool = False,
        latent_type: str = "pre_sample",
        # other config values
        external_cond_dim: int = 4,
        external_cond_stack: bool = True,
        max_frames: int = 50,
        n_frames: int = 17,
        frame_skip: int = 2,
        filter_min_len: Optional[int] = None,
        subdataset_size: Optional[int] = None,
        num_eval_videos: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            save_dir=save_dir,
            resolution=resolution,
            latent_downsampling_factor=latent_downsampling_factor,
            latent_suffix=latent_suffix,
            split=split,
            **kwargs,
        )
        self.current_subepoch = current_epoch
        self.latent_enable = latent_enable
        self.latent_type = latent_type

        # If latents are preprocessed, we check that the folder exists
        self.use_preprocessed_latents = latent_enable and latent_type.startswith("pre_")
        if self.use_preprocessed_latents and not self.latent_dir.exists():
            raise ValueError(
                f"You requested preprocessed latents, but {self.latent_dir} does not exist."
            )

        # subdataset slicing
        self.subdataset_size = subdataset_size
        self.num_eval_videos = num_eval_videos

        # external condition dimension
        # possibly stacked if external_cond_stack is True
        self.external_cond_dim = external_cond_dim * (frame_skip if external_cond_stack else 1)
        self.frame_skip = frame_skip

        # n_frames logic: for training, use max_frames; else use n_frames
        if split == "training":
            self.n_frames = 1 + (max_frames - 1) * frame_skip
        else:
            self.n_frames = 1 + (n_frames - 1) * frame_skip

        # filter_min_len logic
        if split == "training" or filter_min_len is None:
            self.filter_min_len = self.n_frames
        else:
            self.filter_min_len = filter_min_len

        # Optionally exclude videos that lack latents
        if self.use_preprocessed_latents:
            self.metadata = self.exclude_videos_without_latents(self.metadata)

        # Exclude any too-short videos
        self.metadata = self.exclude_short_videos(self.metadata, self.filter_min_len)

        # Let subclasses hook in here before computing clips
        self.on_before_prepare_clips()

        # Prepare clips (computes cumulative_sizes, etc.)
        self.prepare_clips()

    def on_before_prepare_clips(self) -> None:
        """Hook to do anything right before self.prepare_clips()."""
        return

    @property
    def use_subdataset(self) -> bool:
        """Check if subdataset strategy is enabled (for training)."""
        return (
            self.split == "training"
            and self.subdataset_size is not None
            and self.current_subepoch is not None
        )

    @property
    def use_evaluation_subdataset(self) -> bool:
        """Check if we want a deterministic subdataset for evaluation."""
        return self.split != "training" and self.num_eval_videos is not None

    def prepare_clips(self) -> None:
        """
        Precompute how many sub-clips each video can yield, and build self.idx_remap.
        """
        num_clips = torch.as_tensor(
            [max(self.video_length(md) - self.n_frames + 1, 1) for md in self.metadata]
        )
        self.cumulative_sizes = num_clips.cumsum(0).tolist()
        self.idx_remap = self._build_idx_remap()

    def _build_idx_remap(self) -> List[int]:
        """
        Deterministically build the indexing scheme for subdatasets and evaluation.
        """
        if self.use_subdataset:
            # For large training sets, slice the dataset differently each epoch:
            def idx_to_epoch_and_idx(idx: int) -> Tuple[int, int]:
                effective_idx = idx + self.subdataset_size * self.current_subepoch
                return divmod(effective_idx, self.cumulative_sizes[-1])

            start_epoch, start_idx_in_epoch = idx_to_epoch_and_idx(0)
            end_epoch, end_idx_in_epoch = idx_to_epoch_and_idx(self.subdataset_size - 1)
            assert 0 <= end_epoch - start_epoch <= 1, "Subdataset size > dataset size?"

            epoch_to_shuffled: Dict[int, List[int]] = {}
            for epoch in range(start_epoch, end_epoch + 1):
                all_indices = list(range(self.cumulative_sizes[-1]))
                random.seed(epoch)
                random.shuffle(all_indices)
                epoch_to_shuffled[epoch] = all_indices

            if start_epoch == end_epoch:
                idx_remap = epoch_to_shuffled[start_epoch][
                    start_idx_in_epoch : end_idx_in_epoch + 1
                ]
            else:
                idx_remap = (
                    epoch_to_shuffled[start_epoch][start_idx_in_epoch:]
                    + epoch_to_shuffled[end_epoch][: end_idx_in_epoch + 1]
                )
            assert len(idx_remap) == self.subdataset_size
            return idx_remap

        elif self.use_evaluation_subdataset:
            # Deterministically choose exactly one clip per video, then shuffle
            if self.num_eval_videos and self.num_eval_videos > len(self.cumulative_sizes):
                rank_zero_print(
                    cyan(
                        f"Fewer total videos ({len(self.cumulative_sizes)}) than requested "
                        f"eval videos ({self.num_eval_videos})."
                    )
                )
            random.seed(0)
            # pick exactly one clip from each video
            idx_remap = []
            for start, end in zip([0] + self.cumulative_sizes[:-1], self.cumulative_sizes):
                idx_remap.append(random.randrange(start, end))
            random.shuffle(idx_remap)
            return idx_remap[: self.num_eval_videos]

        else:
            # Full dataset with a fixed random shuffle seed = 0
            idx_remap = list(range(self.__len__()))
            random.seed(0)
            random.shuffle(idx_remap)
            return idx_remap

    def exclude_short_videos(
        self, metadata: List[Dict[str, Any]], min_frames: int
    ) -> List[Dict[str, Any]]:
        """
        Exclude videos that are shorter than min_frames frames.
        """
        return self.subsample(
            metadata,
            lambda m: self.video_length(m) >= min_frames,
            filter_msg=f"videos shorter than {min_frames} frames",
        )

    def exclude_videos_without_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """If latents must exist, drop any video that lacks them."""
        latent_paths = set(self.get_latent_paths(self.split))
        return self.subsample(
            metadata,
            lambda m: self.video_metadata_to_latent_path(m) in latent_paths,
            "videos without latents",
        )

    def get_clip_location(self, idx: int) -> Tuple[int, int]:
        """
        Map a dataset index -> (video_idx, clip_idx).
        - video_idx is which video in self.metadata
        - clip_idx is how many frames in to start the sub-clip
        """
        idx = self.idx_remap[idx]
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    def load_latent(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        """Load preprocessed latents from a .pt file."""
        latent_path = self.video_metadata_to_latent_path(video_metadata)
        return torch.load(latent_path, weights_only=False)[start_frame:end_frame]

    @abstractmethod
    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        """Load external condition for frames [start_frame, end_frame)."""
        raise NotImplementedError

    def load_video_and_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video frames plus the external condition.
        """
        video = self.load_video(video_metadata, start_frame, end_frame)
        cond = self.load_cond(video_metadata, start_frame, end_frame)
        return video, cond

    def __len__(self) -> int:
        if self.use_subdataset:
            return self.subdataset_size
        elif self.use_evaluation_subdataset and self.num_eval_videos is not None:
            return min(self.num_eval_videos, len(self.cumulative_sizes))
        else:
            return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a dictionary containing:
            - "videos": (T, C, H, W) or None
            - "latents": (T, #channels, H, W) or None
            - "conds": (T, external_cond_dim) or None
            - "nonterminal": (T,)  boolean mask
        """
        video_idx, clip_idx = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_len = self.video_length(video_metadata)

        start_frame = clip_idx
        end_frame = min(clip_idx + self.n_frames, video_len)

        # Decide whether to load latents / video / cond
        latent = None
        video = None
        cond = None

        if self.use_preprocessed_latents:
            latent = self.load_latent(video_metadata, start_frame, end_frame)

        # If training with latents only, skip the video data unless external cond is needed
        if self.use_preprocessed_latents and self.split == "training":
            if self.external_cond_dim > 0:
                cond = self.load_cond(video_metadata, start_frame, end_frame)
        else:
            # load video (and cond if relevant)
            if self.external_cond_dim > 0:
                video, cond = self.load_video_and_cond(video_metadata, start_frame, end_frame)
            else:
                video = self.load_video(video_metadata, start_frame, end_frame)

        # Check all loaded T match
        loaded_lengths = [len(x) for x in (video, cond, latent) if x is not None]
        if len(loaded_lengths) > 0:
            assert (
                len(set(loaded_lengths)) == 1
            ), f"video, cond, and latent must have the same sequence length but got {loaded_lengths}"

        # Zero-pad if end of video is short
        T = loaded_lengths[0] if loaded_lengths else 0
        pad_len = self.n_frames - T
        nonterminal = torch.ones(self.n_frames, dtype=torch.bool)

        if pad_len > 0 and T > 0:
            if video is not None:
                video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len))  # (T, C, H, W)
            if latent is not None:
                latent = F.pad(latent, (0, 0, 0, 0, 0, 0, 0, pad_len))
            if cond is not None:
                cond = F.pad(cond, (0, 0, 0, pad_len))
            nonterminal[-pad_len:] = 0

        # Apply frame skip
        if self.frame_skip > 1:
            if video is not None:
                video = video[:: self.frame_skip]
            if latent is not None:
                latent = latent[:: self.frame_skip]
            nonterminal = nonterminal[:: self.frame_skip]
            if cond is not None:
                cond = self._process_external_cond(cond)

        # Final dictionary
        output = {}
        if video is not None:
            output["videos"] = self.transform(video)  # e.g. resize
        if latent is not None:
            output["latents"] = latent
        if cond is not None:
            output["conds"] = cond
        output["nonterminal"] = nonterminal

        return output

    def _process_external_cond(self, external_cond: torch.Tensor) -> torch.Tensor:
        """
        Post-process external condition to align with frame skipping.

        By default:
          - We shift the condition by (frame_skip - 1) frames,
          - then flatten them in blocks of size `frame_skip`.
        """
        fs = self.frame_skip
        if fs == 1:
            return external_cond
        # pad front so we have condition for each newly-sparse frame
        external_cond = F.pad(external_cond, (0, 0, fs - 1, 0), value=0.0)
        # rearrange from (T*fs, D) -> (T, fs*D)
        return rearrange(external_cond, "(t fs) d -> t (fs d)", fs=fs)
