**This is the project for object detector experiments**

* Easy to create data loaders, advanced augmentation
* Easy to mix different models
* Easy to save experiment artifacts

Now it in **WIP** status

**Complex experiment creation example:**

Base settings:
```python
categories = ['person']
```

Dataset configuration part:
```python
data_loaders = [
    CocoLoaderConfig(categories=categories, annotations_path='', images_path='')
]
transformers = {
    0: ImgaugImageFormatterConfig(new_image_size=(100, 100)),
    1: ImgaugTransformerConfig(aug_pipeline=iaa.Noop())
}
```

Ground truth encoding part:
```python
centernet_bbox_encode_decode_config = CenternetDetectionEncodeDecodeConfig()
```

Model configuration part:
```python
# Create models
dla_config = DLAConfig(type='dla34', pretrained='')
dla_upsampling_config = FPNConfig(inputs_count=4)
centernet_detection_heads = CenternetDetectionHeadsConfig()

# Create losses
centernet_detection_loss_config = CenternetDetectionLossConfig()

# Create detectors
detector = DetectorPipelineBuilder() \
        .stage(dla_config) \
            .input({'image'}) \
            .output(dla_config.get_outputs(strides=(4, 8, 16, 32))) \
        .end_stage() \
        .stage(dla_upsampling_config) \
            .input(dla_config.get_outputs(strides=(4, 8, 16, 32))) \
            .output(dla_upsampling_config.get_outputs()) \
        .end_stage() \
        .stage(centernet_detection_heads) \
            .input(dla_upsampling_config.get_outputs()) \
            .output(centernet_detection_heads.get_outputs()) \
            .encode_decode(centernet_bbox_encode_decode_config) \
            .loss(centernet_detection_loss_config) \
            .mark_final() \
        .end_stage() \
    .create()
```