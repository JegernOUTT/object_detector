import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import torch

from tools.checkpoint import load_checkpoint


class InitializeType(Enum):
    pass


@dataclass
class BaseModelParams(ABC):
    name: str
    input_layers_names: Union[str, List[str]]
    output_layers_names: Union[str, List[str]]


@dataclass
class CommonModelParams(ABC):
    norm_type: torch.nn.Module
    activation_type: torch.nn.Module


@dataclass
class CheckpointsParams(ABC):
    checkpoint_path: Union[str, None]
    conv_init_type: InitializeType
    bn_init_type: InitializeType


@dataclass
class LayersParams(ABC):
    layers_count: int
    freeze_layers: List[bool]


@dataclass
class AbstractModelParams(BaseModelParams, CommonModelParams, CheckpointsParams, LayersParams):
    pass


class AbstractModel(ABC, torch.nn.Module):
    def __init__(self, params: BaseModelParams):
        super().__init__()
        self._name = params.name
        self._output_names = []
        self._inputs_names = []

    def get_name(self) -> str:
        return self._name

    def set_inputs(self, inputs_names: List[str]):
        self._inputs_names = inputs_names

    def set_outputs(self, output_names: List[str]):
        self._output_names = output_names

    def load_checkpoint(self, filename):
        logger = logging.getLogger()
        load_checkpoint(self, filename, strict=False, logger=logger)

    @abstractmethod
    def init_weights(self, params: CheckpointsParams):
        pass
