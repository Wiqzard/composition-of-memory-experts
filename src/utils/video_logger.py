from lightning.pytorch.callbacks import Callback


class VideoLogger(Callback):
    def __init__(self, log_dir: str, log_freq: int = 100) -> None:
        """Initialize a `VideoLogger`.

        :param log_dir: The directory to save the logs.
        :param log_freq: The frequency to log the video.
        """
        super().__init__()
        self.log_dir = log_dir
        self.log_freq = log_freq

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_freq == 0:
            video = pl_module.generate_video()
            video.save(self.log_dir / f"epoch_{trainer.current_epoch}.gif")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_freq == 0:
            video = pl_module.generate_video()
            video.save(self.log_dir / f"epoch_{trainer.current_epoch}.gif")

    def on_test_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_freq == 0:
            video = pl_module.generate_video()
            video.save(self.log_dir / f"epoch_{trainer.current_epoch}.gif")
