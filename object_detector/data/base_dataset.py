from typing import List, Union

import torch

from object_detector.data.loader.abstract import AbstractLoader
from object_detector.data.model_heads_formatters.abstract import AbstractHeadFormatter
from object_detector.data.transform.abstract import AbstractTransformer
from object_detector.tools.helpers import list_if_not_list
from object_detector.tools.image.image import read_image, tensor_from_rgb_image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 loader: AbstractLoader,
                 transformers: Union[List[AbstractTransformer], AbstractTransformer],
                 head_formatters: Union[List[AbstractHeadFormatter], AbstractHeadFormatter],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loader: AbstractLoader = loader
        self._transformers: List[AbstractTransformer] = sorted(list_if_not_list(transformers),
                                                               key=lambda x: x.get_order())
        self._head_formatters: List[AbstractHeadFormatter] = list_if_not_list(head_formatters)
        self._idx_ranges_to_cache: dict = {}

    def __getitem__(self, idx: int) -> dict:
        load_result = self._loader.load([idx])
        image_info, annotations = load_result.images[0], load_result.annotations
        image = read_image(image_info.filename)

        for transformer in self._transformers:
            image, annotations = transformer(image=image, annotations=annotations)

        model_data = {}
        for head_formatter in self._head_formatters:
            model_data.update(head_formatter(image=image, annotations=annotations))

        model_data['image'] = tensor_from_rgb_image(image)

        return model_data

    def __len__(self) -> int:
        return self._loaders.images_count()
