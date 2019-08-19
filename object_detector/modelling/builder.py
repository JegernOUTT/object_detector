from typing import List

from object_detector.modelling.abstract import AbstractModelParams, AbstractModel
from object_detector.modelling.detectors.detector import Detector
from object_detector.modelling.model_pipeline import ModelPipeline


class DetectorBuilder:
    def __init__(self,
                 model_params: List[AbstractModelParams],
                 model_pipeline: ModelPipeline):
        self._model_params: List[AbstractModelParams] = model_params
        self._model_pipeline: ModelPipeline = model_pipeline

    def _load_states(self):
        pass

    def build_detector(self) -> Detector:
        modules = []
        stages = self._model_pipeline.get_stages()
        stages_by_model_name = {stage.model: stage for stage in stages}

        for params in self._model_params:
            module: AbstractModel = ...  #  Create by name from registry
            if params.checkpoint_path is not None:
                module.load_checkpoint(params.checkpoint_path)
            else:
                module.init_weights(params)

            stage = stages_by_model_name[params.name]
            module.set_inputs(stage.in_)
            module.set_outputs(stage.out)
            modules.append(module)

        return Detector(modules=modules, stages=stages)
