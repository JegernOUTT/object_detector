from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from object_detector.data.structs import TransformConfig, AnnotationInformation


class AbstractTransformer(ABC):
    def __init__(self, transform_config: TransformConfig, order: int):
        self._config: TransformConfig = transform_config
        self._order: int = order

    def get_order(self) -> int:
        return self._order

    @abstractmethod
    def __call__(self,
                 image: np.ndarray,
                 annotations: List[AnnotationInformation]) -> Tuple[np.ndarray, AnnotationInformation]:
        pass
