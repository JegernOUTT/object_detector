from typing import List


class ModelPipeline:
    def __init__(self, stages):
        self._stages: List['BuilderStage'] = stages

        self._loss_stages = [stage for stage in self._stages if stage._loss_config is not None]
        self._final_stages = [stage for stage in self._stages if stage._final]

        self._lifetime_indices = {}
        for stage_idx, stage in reversed(list(enumerate(self._stages))):
            if stage in self._loss_stages or stage in self._final_stages:
                continue
            for name in stage._input_layer_names:
                if name in self._lifetime_indices:
                    continue
                self._lifetime_indices[name] = stage_idx + 1

        self._iter_idx = 0

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __len__(self):
        return len(self._stages)

    def __next__(self):
        stage_idx = self._iter_idx
        module_name = self._stages[self._iter_idx]._stage_config.name
        input_tensors_names = self._stages[self._iter_idx]._input_layer_names
        expired_tensor_names = [name for name in input_tensors_names
                                if self._lifetime_indices[name] <= stage_idx]
        self._iter_idx += 1
        return stage_idx, module_name, input_tensors_names, expired_tensor_names

    def get_final_tensor_names(self) -> List[str]:
        return [
            name
            for stage in self._final_stages
            for name in stage._output_layer_names
        ]

    def get_loss_tensor_names(self) -> List[str]:
        return [
            name
            for stage in self._loss_stages
            for name in stage._output_layer_names
        ]
