from typing import Any, Callable, Dict, Optional, Type

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DataModuleFactory(LightningDataModule):
    """
    A generic LightningDataModule that can instantiate one or more dataset splits
    from a provided dataset class and shared keyword arguments.

    This class expects that your dataset class accepts a 'split' argument
    (e.g., 'training', 'validation', 'test'), along with any other necessary kwargs.
    """

    def __init__(
        self,
        dataset_cls: Callable[..., Dataset],
        batch_size: int = 32,
        num_workers: int = 16,
        pin_memory: bool = False,
        shuffle: bool = True,
        overwrite_split: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            dataset_cls (Type[Dataset]): The class (not instance) of the dataset to use.
            dataset_kwargs (Dict[str, Any]): A dictionary of keyword arguments to pass
                                             into the dataset constructor (apart from 'split').
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of workers for dataloaders.
            pin_memory (bool): Whether to pin GPU memory in dataloaders.
            **kwargs: Additional keyword arguments for further extension if needed.
        """
        super().__init__()
        self.dataset_cls = dataset_cls
        # self.dataset_kwargs = dataset_kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.overwrite_split = overwrite_split

        # Placeholders for the splits
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called by Lightning at the start of fit/validate/test/predict.
        Depending on 'stage', sets up the appropriate datasets.
        """
        if stage == "fit" or stage is None:
            # Instantiate the 'training' and 'validation' datasets
            self.dataset_train = self.dataset_cls(
                split=("training" if self.overwrite_split is None else self.overwrite_split)
            )  # , **self.dataset_kwargs)

        if stage == "fit" or stage == "validate":
            self.dataset_val = self.dataset_cls(
                split=("validation" if self.overwrite_split is None else self.overwrite_split)
            )  # , **self.dataset_kwargs)

        if stage == "test":
            # Instantiate the 'test' dataset
            self.dataset_test = self.dataset_cls(
                split="test" if self.overwrite_split is None else self.overwrite_split
            )  # , **self.dataset_kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.dataset_train is None:
            raise ValueError("Training dataset is not initialized. Call `.setup('fit')` first.")
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            worker_init_fn=lambda worker_id: (
                self.dataset_train.worker_init_fn(worker_id)
                if hasattr(self.dataset_train, "worker_init_fn")
                else None
            ),
        )

    def val_dataloader(self) -> DataLoader:
        if self.dataset_val is None:
            raise ValueError("Validation dataset is not initialized. Call `.setup('fit')` first.")
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.dataset_test is None:
            raise ValueError("Test dataset is not initialized. Call `.setup('test')` first.")
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
