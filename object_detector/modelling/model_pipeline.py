from dataclasses import dataclass
from enum import Enum
from typing import Union, List

from object_detector.modelling.abstract import BaseModelParams
from object_detector.tools.helpers import list_if_not_list


class InputType(Enum):
    Image = 0


class OutputType(Enum):
    All = 0
    LastOutput = 1


@dataclass
class Model2Layer:
    model: str
    layer: str


@dataclass
class Stage:
    module: str
    in_: Union[List[Model2Layer], List[InputType], Model2Layer, InputType]
    out: Union[List[str], List[OutputType], str, OutputType]

    def __post_init__(self):
        self.in_ = list_if_not_list(self.in_)
        self.out = list_if_not_list(self.out)


class ModelPipeline:
    def __init__(self, model_settings: List[BaseModelParams]):
        self._pipeline = []
        self._model_settings = model_settings
        self._model_names = {
            model.name: {
                'possible_in': set(list_if_not_list(model.input_layers_names) + [InputType.Image]),
                'possible_out': set(list_if_not_list(model.output_layers_names) + [OutputType.All])
            }
            for model in self._model_settings}

        self._stages: List[Stage] = []

        self._used_models = set()
        self._current_available_layers = set()

    def get_model_settings(self) -> List[BaseModelParams]:
        return self._model_settings

    def get_stages(self) -> List[Stage]:
        return self._stages

    def add_stage(self, stage: Stage) -> 'ModelPipeline':
        assert stage.module not in self._used_models, \
            f'{stage.module} was used on early stages. Use each model single time'

        assert all(l.layer in self._model_names[l.model]['possible_out'] for l in stage.in_
                   if type(l) != InputType), \
            f'Unrecognized input layer name found for model {stage.module}, check input layers: {stage.in_}'

        assert all(l in self._model_names[stage.module]['possible_out'] for l in stage.out
                   if type(l) != OutputType), \
            f'Unrecognized output layer name found for model {stage.module}, check output layers: {stage.out}'

        self._used_models.add(stage.module)
        self._current_available_layers.update(stage.out)

        if OutputType.All in stage.out:
            stage.out = self._model_names[stage.module]['possible_out']

        # Save all outputs in this Mode
        if OutputType.LastOutput in stage.out:
            stage.out = [OutputType.LastOutput]

        self._stages.append(stage)
        return self
