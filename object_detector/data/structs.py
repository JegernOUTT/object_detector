from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Optional

from object_detector.tools.bbox.bbox import Bbox
from object_detector.tools.keypoint.keypoint import Keypoint

import numpy as np

from object_detector.tools.mask.mask import Mask


@dataclass
class LoaderConfig:
    pass


@dataclass
class AugmenterConfig:
    pass


@dataclass
class HeadFormatterConfig:
    pass


@dataclass
class ImageInformation:
    filename: Path()
    annotations: np.ndarray  # numpy array of int indices
    meta: Any


@dataclass
class AnnotationInformation:
    annotation: Optional[Bbox, Mask, Keypoint]
    meta: Any


@dataclass
class DataLoaderResult:
    image_base_path: Path
    images: List[ImageInformation]
    annotations: List[AnnotationInformation]
