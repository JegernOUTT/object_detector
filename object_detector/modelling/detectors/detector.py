from typing import List, Dict, Any, Tuple

import torch

from object_detector.modelling.abstract import AbstractModel
from object_detector.modelling.detector_pipeline import ModelPipeline


class Detector(torch.nn.Module):
    def __init__(self,
                 modules: List[AbstractModel],
                 encoders: List,
                 decoders: List,
                 losses: List,
                 pipeline: ModelPipeline):
        super().__init__()

        self._pipeline = pipeline
        self._tensor_cache = {}

        self._modules = torch.nn.ModuleDict({
            module.name: module
            for module in modules
        })
        self._encoders: List = encoders
        self._decoders: List = decoders
        self._losses = torch.nn.ModuleDict({
            loss.name: loss
            for loss in losses
        })

    def forward_decode(self, gt):
        _, final_tensors = self._forward_modules(gt)
        return self._decode(final_tensors)

    def forward_loss(self, gt):
        loss_tensors, _ = self._forward_modules(gt)
        return self._forward_loss(self._encode(gt), loss_tensors)

    def _forward_modules(self, gt) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        def __load_save_infer(module_name, input_tensor_names, expired_tensor_names):
            input_tensors = self._load_tensor_from_cache(input_tensor_names)
            out_tensors = self._modules[module_name](input_tensors)
            self._remove_tensors_from_cache(expired_tensor_names)
            self._save_tensors_in_cache(out_tensors)

        pipeline_iter = iter(self._pipeline)
        _, module_name, input_tensors_names, expired_tensor_names = next(pipeline_iter)
        self._save_gt_in_cache(gt, input_tensors_names)
        __load_save_infer(module_name, input_tensors_names, expired_tensor_names)

        for stage_idx, module_name, input_tensors_names, expired_tensor_names in pipeline_iter:
            __load_save_infer(module_name, input_tensors_names, expired_tensor_names)

        final_tensors = self._load_tensor_from_cache(self._pipeline.get_final_tensor_names())
        loss_tensors = self._load_tensor_from_cache(self._pipeline.get_loss_tensor_names())
        self._clean_cache()
        return loss_tensors, final_tensors

    def _forward_loss(self, encoded_gt: Dict[str, torch.Tensor], predicted: Dict[str, torch.Tensor]) \
            -> Dict[str, torch.Tensor]:
        return {loss.name: loss(encoded_gt, predicted) for loss in self._losses}

    def _encode(self, gt) -> Dict[str, torch.Tensor]:
        encoded_data = {}
        for encoder in self._encoders:
            encoded_data.update(encoder(gt))
        return encoded_data

    def _decode(self, predicted: Dict[str, torch.Tensor]) -> Dict[Any, Any]:
        decoded_data = {}
        for decoder in self._decoders:
            decoded_data.update(decoder(predicted))
        return decoded_data

    def _save_gt_in_cache(self, gt: Dict, keys: Tuple[str] = ()):
        for key in keys:
            assert key in gt and type(gt[key]) == torch.Tensor
            self._tensor_cache[key] = gt[key]

    def _save_tensors_in_cache(self, tensors: Dict[str, torch.Tensor]):
        self._tensor_cache.update(tensors)

    def _load_tensor_from_cache(self, names: List[str]) -> Dict[str, torch.Tensor]:
        return {name: self._tensor_cache[name] for name in names}

    def _remove_tensors_from_cache(self, names: List[str]):
        for name in names:
            del self._tensor_cache[name]

    def _clean_cache(self):
        self._tensor_cache.clear()
