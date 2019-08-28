from abc import ABC, abstractmethod
from typing import List

import numpy as np

from object_detector.data.structs import HeadFormatterConfig, AnnotationInformation


class AbstractHeadFormatter(ABC):
    def __init__(self, head_formatter_config: HeadFormatterConfig):
        self._config = head_formatter_config

    @abstractmethod
    def __call__(self, image: np.ndarray, annotations: List[AnnotationInformation]) -> dict:
        pass
