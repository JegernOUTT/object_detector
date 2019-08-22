from typing import List, Tuple, Union

import torch
import tqdm as tqdm

from object_detector.data.transform.abstract import AbstractTransformer
from object_detector.data.loader.abstract import AbstractLoader
from object_detector.data.model_heads_formatters.abstract import AbstractHeadFormatter
from object_detector.data.structs import DataLoaderResult, ImageInformation, AnnotationInformation
from object_detector.tools.helpers import list_if_not_list
from object_detector.tools.image.image import read_image, tensor_from_rgb_image


class DataPipeline(torch.utils.data.Dataset):
    def __init__(self,
                 loaders: Union[List[AbstractLoader], AbstractLoader],
                 transformers: Union[List[AbstractTransformer], AbstractTransformer],
                 head_formatters: Union[List[AbstractHeadFormatter], AbstractHeadFormatter]):
        self._loaders: List[AbstractLoader] = list_if_not_list(loaders)
        self._transformers: List[AbstractTransformer] = sorted(list_if_not_list(transformers),
                                                               key=lambda x: x.get_order())
        self._head_formatters: List[AbstractHeadFormatter] = list_if_not_list(head_formatters)

        self._data_cache: List[DataLoaderResult] = []
        self._idx_ranges_to_cache: dict = {}

    def load(self):
        max_idx = 0
        for loader in tqdm.tqdm(self._loaders, 'Loading data'):
            self._data_cache.append(loader.load())
            self._idx_ranges_to_cache[(max_idx, len(self._data_cache[-1].images) - 1)] = self._data_cache[-1]
            max_idx += len(self._data_cache[-1].images)

    def _get_item_by_idx(self, idx: int) -> Tuple[ImageInformation, List[AnnotationInformation]]:
        for (low, high), cache in self._idx_ranges_to_cache.items():
            if low <= idx <= high:
                cache_idx = idx - low
                image = cache.images[cache_idx]
                image.filename = cache.image_base_path / image.filename
                return image, list(map(cache.annotations.__getitem__, image.annotations))
        assert False, f'{idx} element not found in loaded datasets'

    def __getitem__(self, idx: int) -> dict:
        image_info, annotations = self._get_item_by_idx(idx)
        image = read_image(image_info)

        for transformer in self._transformers:
            image, annotations = transformer(image=image, annotations=annotations)

        model_data = {}
        for head_formatter in self._head_formatters:
            model_data.update(head_formatter(image=image, annotations=annotations))

        model_data['image'] = tensor_from_rgb_image(image)

        return model_data

    def __len__(self) -> int:
        return sum([len(res.images) for res in self._data_cache])
