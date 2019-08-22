from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from object_detector.data.structs import DataLoaderResult, LoaderConfig, AnnotationInformation, ImageInformation


class AbstractLoader(ABC):
    def __init__(self, loader_config: LoaderConfig):
        self._config = loader_config

    @abstractmethod
    def _load_single(self, idx: int) -> Tuple[ImageInformation, List[AnnotationInformation]]:
        pass

    @abstractmethod
    def images_count(self) -> int:
        pass

    def load(self, indices: List[int]) -> DataLoaderResult:
        images, all_annotations = [], []
        for idx, item_idx in enumerate(indices):
            image, annotations = self._load_single(item_idx)
            annotations = self._filter_by_size(annotations)
            image.annotations = np.arange(len(all_annotations), len(annotations))
            images.append(image)
            all_annotations.extend(annotations)
        return DataLoaderResult(images=images, annotations=all_annotations)

    def _filter_by_size(self, annotations: List[AnnotationInformation]) -> List[AnnotationInformation]:
        return annotations
