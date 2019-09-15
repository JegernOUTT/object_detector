from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from object_detector.modelling.abstract import AbstractModel, CheckpointsParams, BaseModelParams, BaseModelConfig


@dataclass
class CenternetDetectionHeadConfig(BaseModelConfig):
    conv_in_channels: int
    categories_count: int
    conv_out_channels: int = 128
    xy_key_name = 'centernet_xy'
    xy_offset_key_name = 'centernet_xy_offset'
    wh_key_name = 'centernet_wh'

    def create_module(self) -> 'CenternetDetectionHead':
        return CenternetDetectionHead(
            conv_in_channels=self.conv_in_channels,
            conv_out_channels=self.conv_out_channels,
            categories_count=self.categories_count,
            base_params=BaseModelParams(name=self.name),
            config=self)

    def _get_outputs_by_strides(self, strides: Tuple) -> List[str]:
        return [self.xy_key_name, self.xy_offset_key_name, self.wh_key_name]

    def _get_all_outputs(self) -> List[str]:
        return [self.xy_key_name, self.xy_offset_key_name, self.wh_key_name]


class CenternetDetectionHead(AbstractModel):
    def __init__(self,
                 conv_in_channels: int,
                 conv_out_channels: int,
                 categories_count: int,
                 base_params: BaseModelParams,
                 config: CenternetDetectionHeadConfig):
        super().__init__(base_params)
        self._config = config

        def __create_simple_seq(out_ch_count):
            return torch.nn.Sequential(
                torch.nn.Conv2d(conv_in_channels,
                                conv_out_channels,
                                kernel_size=3,
                                padding=1,
                                bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(conv_out_channels,
                                out_ch_count,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True))

        self._xy = __create_simple_seq(categories_count)
        self._xy_offset = __create_simple_seq(2)
        self._wh = __create_simple_seq(2)

    def init_weights(self, params: CheckpointsParams):
        pass

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert len(input) == 1
        x, = tuple(input.values())
        xy_out = self._xy(x)
        xy_offset_out = self._xy_offset(x)
        wh_out = self._wh(x)
        return {self._config.xy_key_name: xy_out,
                self._config.xy_offset_key_name: xy_offset_out,
                self._config.wh_key_name: wh_out}
