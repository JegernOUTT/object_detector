from typing import List, Union

import torch

from object_detector.modelling.abstract import AbstractModel, CheckpointsParams
from object_detector.modelling.model_pipeline import Stage, Model2Layer, InputType, OutputType


class Detector(torch.nn.Module, AbstractModel):
    def __init__(self, modules: List[AbstractModel], stages: List[Stage]):
        super().__init__()
        self._modules = torch.nn.ModuleDict({
            module.name: module
            for module in modules
        })
        self._stages = stages

    def init_weights(self, params: CheckpointsParams):
        pass

    def _get_inputs_from_buffer(self,
                                stage_inputs: Union[List[Model2Layer], List[InputType]],
                                current_buffer: dict) \
            -> Union[List[torch.Tensor], torch.Tensor]:
        inputs = []
        for stage_input in stage_inputs:
            if type(stage_input) == InputType:
                if stage_input == InputType.Image:
                    inputs.append(current_buffer[InputType.Image])
            else:
                inputs.append(current_buffer[(stage_input.model, stage_input.layer)])
        return inputs

    def _clean_expired_buffer(self, current_buffer: dict, current_stage: int) -> dict:
        if len(self._stages) - 1 <= current_stage:
            return {}

        required_names = set()
        for stage in self._stages[current_stage + 1:]:
            for in_ in stage.in_:
                required_names.add((in_.model, in_.name))

        return {k: v for k, v in current_buffer.items() if k in required_names}

    def forward(self, image: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        outputs = []
        required_output_buffer = {
            InputType.Image: image
        }
        for idx, stage in enumerate(self._stages):
            module = self._modules[stage.model]
            outputs = module(self._get_inputs_from_buffer(stage.in_, required_output_buffer))
            for out, layer_name in zip(outputs, stage.out):
                required_output_buffer[(module.name, layer_name)] = out
                if layer_name == OutputType.LastOutput:
                    outputs.append(out)
            required_output_buffer = self._clean_expired_buffer(required_output_buffer, idx)
        return outputs
