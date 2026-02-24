from torch import Tensor
from src.models.metrics.video.types import VideoMetricModelType
from .cosine_similarity_dimension import CosineSimilarityDimension


class SubjectConsistency(CosineSimilarityDimension):
    """
    Subject consistency dimension.
    """

    def extract_features(self, videos: Tensor) -> Tensor:
        """
        Extract DINO features from the video.
        """
        return self.registry(VideoMetricModelType.DINO, videos)
