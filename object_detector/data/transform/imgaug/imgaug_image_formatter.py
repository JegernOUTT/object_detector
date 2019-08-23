from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Type

import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa

from object_detector.data.structs import AnnotationInformation
from object_detector.data.transform.imgaug.imgaug_transformer import ImgaugTransformer, ImgaugTransformerConfig
from object_detector.data.transform.imgaug.smart_crop import get_smart_crop_augmenter
from object_detector.tools.structs import Size2D


class ImageFormatOptions(Enum):
    KeepAspectRatioAndRandomPad = 1
    KeepAspectRatioAndRandomCrop = 2
    Resize = 3
    KeepAspectRatioAndCenterPad = 4
    KeepAspectRatioAndCenterCrop = 5


@dataclass
class ImgaugImageFormatterConfig(ImgaugTransformerConfig):
    new_image_size: Size2D
    format_options: ImageFormatOptions = ImageFormatOptions.KeepAspectRatioAndCenterPad
    use_smart_crop: bool = False
    min_crop_ratio: float = 9 / 16
    max_crop_ratio: float = 16 / 9
    min_crop_pix_size: int = 50
    max_crop_pix_size: int = 5000
    aug_pipeline: iaa.Augmenter = iaa.Noop()

    def owner_type(self) -> Type['ImgaugImageFormatter']:
        return ImgaugImageFormatter


class ImgaugImageFormatter(ImgaugTransformer):
    def __init__(self,
                 transform_config: ImgaugImageFormatterConfig,
                 order: int):
        super().__init__(transform_config, order)

    def _smart_crop(self, image: np.ndarray, annotations: List[AnnotationInformation]) \
            -> Tuple[np.ndarray, List[AnnotationInformation]]:
        if self._config.use_smart_crop:
            smart_crop_augmenter = get_smart_crop_augmenter(
                image_size=Size2D(width=image.shape[1], height=image.shape[0]),
                annotations=annotations,
                min_crop_ratio=self._config.min_crop_ratio,
                max_crop_ratio=self._config.max_crop_ratio,
                min_crop_pix_size=self._config.min_crop_pix_size,
                max_crop_pix_size=self._config.max_crop_pix_size)
        else:
            smart_crop_augmenter = iaa.Noop()

        self._set_aug_pipeline(smart_crop_augmenter)
        return super().__call__(image=image, annotations=annotations)

    def __call__(self, image: np.ndarray, annotations: List[AnnotationInformation]) \
            -> Tuple[np.ndarray, List[AnnotationInformation]]:
        image, annotations = self._smart_crop(image=image, annotations=annotations)

        h, w = image.shape[:2]
        new_w, new_h = self._config.new_image_size.width, self._config.new_image_size.height

        if (w / h) < (new_w / new_h):
            if self._config.format_options in [ImageFormatOptions.KeepAspectRatioAndRandomPad,
                                               ImageFormatOptions.KeepAspectRatioAndCenterPad]:
                resize_params = {"height": new_h, "width": 'keep-aspect-ratio'}
            else:
                resize_params = {"height": 'keep-aspect-ratio', "width": new_w}
        else:
            if self._config.format_options in [ImageFormatOptions.KeepAspectRatioAndRandomPad,
                                               ImageFormatOptions.KeepAspectRatioAndCenterPad]:
                resize_params = {"height": 'keep-aspect-ratio', "width": new_w}
            else:
                resize_params = {"height": new_h, "width": 'keep-aspect-ratio'}

        if self._config.format_options == ImageFormatOptions.KeepAspectRatioAndRandomPad:
            self._set_aug_pipeline(
                iaa.Sequential([
                    iaa.Resize(resize_params, interpolation=ia.ALL),
                    iaa.PadToFixedSize(width=new_w, height=new_h, pad_mode='constant', pad_cval=127),
                ]))
        elif self._config.format_options == ImageFormatOptions.KeepAspectRatioAndRandomCrop:
            self._set_aug_pipeline(
                iaa.Sequential([
                    iaa.Resize(resize_params, interpolation=ia.ALL),
                    iaa.CropToFixedSize(width=new_w, height=new_h)
                ]))
        elif self._config.format_options == ImageFormatOptions.Resize:
            self._set_aug_pipeline(
                iaa.Sequential([
                    iaa.Resize({"height": new_h, "width": new_w}, interpolation=ia.ALL)
                ]))
        elif self._config.format_options == ImageFormatOptions.KeepAspectRatioAndCenterPad:
            self._set_aug_pipeline(
                iaa.Sequential([
                    iaa.Resize(resize_params, interpolation=ia.ALL),
                    iaa.PadToFixedSize(width=new_w, height=new_h, pad_mode='constant',
                                       position=(0.5, 0.5), pad_cval=127),
                ]))
        elif self._config.format_options == ImageFormatOptions.KeepAspectRatioAndCenterCrop:
            self._set_aug_pipeline(
                iaa.Sequential([
                    iaa.Resize(resize_params, interpolation=ia.ALL),
                    iaa.CropToFixedSize(width=new_w, height=new_h, position=(0.5, 0.5))
                ]))
        else:
            assert False

        return super().__call__(image=image, annotations=annotations)
