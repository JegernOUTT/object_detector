from typing import List, Any

from object_detector.modelling.detector_pipeline import ModelPipeline
from object_detector.modelling.detectors.detector import Detector


class BuilderStage:
    def __init__(self, builder: 'DetectorPipelineBuilder', stage_config: Any):
        self._builder = builder

        self._stage_config = stage_config
        self._input_layer_names = []
        self._output_layer_names = []
        self._encode_decode_config = None
        self._loss_config = None
        self._final = False

    def input(self, layer_names: List[str]):
        self._input_layer_names = layer_names
        return self

    def output(self, layer_names: List[str]):
        self._output_layer_names = layer_names
        return self

    def encode_decode(self, encode_decode_config: Any):
        self._encode_decode_config = encode_decode_config
        return self

    def loss(self, loss_config: Any):
        self._loss_config = loss_config
        return self

    def mark_final(self):
        self._final = True
        return self

    def end_stage(self):
        return self._builder


class DetectorPipelineBuilder:
    def __init__(self):
        self._stages: List[BuilderStage] = []

    def stage(self, stage_config):
        stage = BuilderStage(self, stage_config)
        self._stages.append(stage)
        return stage

    def create(self):
        pipeline = ModelPipeline(stages=self._stages)
        modules = [stage._stage_config.create_module() for stage in self._stages]
        encoders, decoders = list(zip(*[(stage._encode_decode_config.create_encoder(),
                                         stage._encode_decode_config.create_decoder())
                                        for stage in self._stages if stage._encode_decode_config is not None]))
        losses = [stage._loss_config.create_module() for stage in self._stages if stage._loss_config is not None]

        return Detector(modules=modules, encoders=encoders, decoders=decoders, losses=losses, pipeline=pipeline)


if __name__ == '__main__':
    from object_detector.modelling.models.dla.dla import DLAModelConfig
    from object_detector.encode_decode.impl.centernet import CenternetDetectionEncoderDecoderConfig
    from object_detector.modelling.head.centernet_detection import CenternetDetectionHeadConfig

    dla_config = DLAModelConfig(name='dla34', type='dla34')
    dla_upsampling_config = None
    centernet_detection_heads = CenternetDetectionHeadConfig(categories_count=1,
                                                             conv_in_channels=512,
                                                             name='centernet_detection_head')
    centernet_bbox_encode_decode_config = CenternetDetectionEncoderDecoderConfig()
    centernet_detection_loss_config = None

    '''
        .stage(dla_upsampling_config) \
            .input(dla_config.get_outputs(strides=(4, 8, 16, 32))) \
            .output(dla_upsampling_config.get_outputs()) \
        .end_stage() \
    '''
    detector = DetectorPipelineBuilder() \
            .stage(dla_config) \
                .input(['image']) \
                .output(dla_config.get_outputs(strides=(4, 8, 16, 32))) \
            .end_stage() \
            .stage(centernet_detection_heads) \
                .input(dla_config.get_outputs()) \
                .output(centernet_detection_heads.get_outputs()) \
                .encode_decode(centernet_bbox_encode_decode_config) \
                .loss(centernet_detection_loss_config) \
                .mark_final() \
            .end_stage() \
        .create()
