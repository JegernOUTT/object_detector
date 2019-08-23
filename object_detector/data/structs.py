from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Any, Tuple, Union

import numpy as np

from object_detector.tools.bbox.bbox import Bbox
from object_detector.tools.keypoint.keypoints import Keypoints
from object_detector.tools.mask.mask import Mask
from object_detector.tools.polygon.polygon import Polygon
from object_detector.tools.registry import RegistryConfig


@dataclass
class LoaderConfig(ABC, RegistryConfig):
    class LoadType(Enum):
        Skip = 0
        AsIgnore = 1

    categories: List[str]
    bbox_minmax_sizes: Tuple[float, float, float, float] = (0., 0., 1., 1.)
    filtered_bbox_skip_type: LoadType = LoadType.AsIgnore


@dataclass
class DatasetConfig(ABC, RegistryConfig):
    pass


@dataclass
class TransformConfig(ABC, RegistryConfig):
    pass


@dataclass
class HeadFormatterConfig(ABC, RegistryConfig):
    categories_count: int


@dataclass
class ImageInformation:
    filename: Path()
    annotations: np.ndarray = np.array([])  # numpy array of int indices, filling in AbstractLoader
    meta: Any = None


@dataclass
class AnnotationInformation:
    annotation: Union[Bbox, Mask, List[Keypoints], Polygon]
    confidence: float
    category_id: int
    ignore: bool
    meta: Any = None


@dataclass
class DataLoaderResult:
    images: List[ImageInformation]
    annotations: List[AnnotationInformation]
