import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Dict, Tuple

import torch

from object_detector.tools.checkpoint import load_checkpoint


class InitializeType(Enum):
    pass


@dataclass
class BaseModelParams(ABC):
    name: str


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


@dataclass
class BaseModelConfig:
    name: str

    def get_outputs(self, strides: Union[None, Tuple] = None) -> List[str]:
        if strides is not None:
            return self._get_outputs_by_strides(strides=strides)
        else:
            return self._get_all_outputs()

    @abstractmethod
    def create_module(self) -> 'AbstractModel':
        pass

    @abstractmethod
    def _get_outputs_by_strides(self, strides: Tuple) -> List[str]:
        pass

    @abstractmethod
    def _get_all_outputs(self) -> List[str]:
        pass


class AbstractModel(ABC, torch.nn.Module):
    def __init__(self, params: BaseModelParams):
        super().__init__()
        self._name = params.name
        self._possible_inputs: Dict[str, int] = {}
        self._possible_outputs: Dict[str, int] = {}
        self._output_names: List[str] = []
        self._inputs_names: List[str] = []

        self._current_outputs: Dict[str, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return self._name

    def _add_possible_input(self, name: str, stride: int):
        self._possible_inputs[name] = stride

    def _add_possible_output(self, name: str, stride: int):
        self._possible_outputs[name] = stride

    def _infer_and_save_if_needed(
            self, layer_name: str, input: torch.Tensor) -> torch.Tensor:
        x = getattr(self, layer_name)(input)
        if layer_name in self._output_names:
            self._current_outputs[layer_name] = x
        return x

    def _pop_saved_output(self) -> Dict[str, torch.Tensor]:
        outputs = {f'{self._name}/{name}': tensor
                   for name, tensor in self._current_outputs.items()}
        self._current_outputs.clear()
        return outputs

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
