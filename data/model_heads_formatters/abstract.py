from abc import ABC, abstractmethod

import numpy as np

from data.structs import AnnotationInformation, HeadFormatterConfig


class AbstractHeadFormatter(ABC):
    def __init__(self, head_formatter_config: HeadFormatterConfig):
        self._config: HeadFormatterConfig = head_formatter_config

    @abstractmethod
    def __call__(self, image: np.ndarray, annotations: AnnotationInformation) -> dict:
        pass
