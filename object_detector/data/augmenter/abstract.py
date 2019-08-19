from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from object_detector.data.structs import AugmenterConfig, AnnotationInformation


class AbstractAugmenter(ABC):
    def __init__(self, augmenter_config: AugmenterConfig, order: int):
        self._config: AugmenterConfig = augmenter_config
        self._order: int = order

    def get_order(self) -> int:
        return self._order

    @abstractmethod
    def __call__(self,
                 image: np.ndarray,
                 annotations: List[AnnotationInformation]) -> Tuple[np.ndarray, AnnotationInformation]:
        pass
