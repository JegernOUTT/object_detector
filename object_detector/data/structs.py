from abc import ABC, abstractproperty
from enum import Enum

from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Optional, Tuple, Type

from object_detector.data.loader.abstract import AbstractLoader
from object_detector.tools.bbox.bbox import Bbox
from object_detector.tools.keypoint.keypoints import Keypoints

import numpy as np

from object_detector.tools.mask.mask import Mask
from object_detector.tools.polygon.polygon import Polygon


@dataclass
class LoaderConfig(ABC):
    class LoadType(Enum):
        Skip = 0
        AsIgnore = 1

    categories: List[str]
    bbox_minmax_sizes: Tuple[float, float, float, float] = (0., 0., 1., 1.)
    filtered_bbox_skip_type: LoadType = LoadType.AsIgnore

    @abstractproperty
    def _loader_type(self) -> Type[AbstractLoader]:
        pass


@dataclass
class DatasetConfig:
    pass


@dataclass
class TransformConfig:
    pass


@dataclass
class HeadFormatterConfig:
    pass


@dataclass
class ImageInformation:
    filename: Path()
    annotations: np.ndarray = np.array([])  # numpy array of int indices, filling in AbstractLoader
    meta: Any = None


@dataclass
class AnnotationInformation:
    annotation: Optional[Bbox, Mask, List[Keypoints], Polygon]
    confidence: float
    category_id: int
    ignore: bool
    meta: Any = None


@dataclass
class DataLoaderResult:
    images: List[ImageInformation]
    annotations: List[AnnotationInformation]
