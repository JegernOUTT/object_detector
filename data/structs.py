from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Optional

from tools.bbox.bbox import Bbox
from tools.keypoint.keypoint import Keypoint
from tools.mask.mask import Mask
from tools.structs import Size2D

import numpy as np


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
