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

Model configuration part:
```python
# Create models
dla = DLA34Config(pretrained='').create_modulde()
base_fpn = FPNConfig(inputs_count=3).create_module()
roi_align = RoiAlignConfig().create_module()
masks_fpn = SmallFPNConfig(inputs_count=4).create_module()
centernet_detection_heads = CenternetDetectionHeadsConfig().create_module()
centernet_pose_heads = CenternetPoseHeadsConfig().create_module()
mask_head = BboxSegmentationMaskHeadConfig().create_module()
yolo3_detection_heads = Yolo3DetectionsHeadConfig(fpn_strides=[16, 32]).create_module()

# Create losses
centernet_detection_loss = CenternetDetectionLossConfig().create_module()
centernet_pose_loss = CenternetPoseLossConfig().create_module()
mask_loss = MaskLossConfig().create_module()
yolo3_detection_loss = Yolo3LossConfig().create_module()
giou_centernet_detection_loss = GIoULossConfig().create_module()
giou_yolo3_detection_loss = GIoULossConfig().create_module()

# Create predict heads
centernet_bbox_decoder = CenternetDetectionEncodeDecodeConfig.create_decoder()
centernet_pose_decoder = CenternetPoseEncodeDecodeConfig.create_decoder()
mask_decoder = MaskEncodeDecodeConfig.create_decoder()
yolo3_bbox_decoder = Yolo3EncodeDecodeConfig.create_decoder()

# Create detectors
train_model = ModelPipelineBuilder \
    .with_models([dla, base_fpn, roi_align, masks_fpn,
                  centernet_detection_heads, centernet_pose_heads,
                  mask_head]) \
        .add_stage(Stage(module=dla, in_=InputType.Image)) \
        .add_stage(Stage(module=base_fpn, in_=dla.get_outputs(strides=(8, 16, 32)))) \
        .add_stage(Stage(module=masks_fpn, in_=dla.get_outputs(strides=(4, 8, 16, 32)))) \
        .add_stage(Stage(module=roi_align, in_=masks_fpn.get_outputs(strides=(4, 8, 16, 32)))) \
        # Heads
        .add_stage(Stage(module=centernet_detection_heads, in_=base_fpn.get_outputs())) \
        .add_stage(Stage(module=centernet_pose_heads, in_=base_fpn.get_outputs())) \
        .add_stage(Stage(module=mask_head, in_=roi_align.get_outputs())) \
        .add_stage(Stage(module=yolo3_detection_heads, in_=base_fpn.get_outputs(strides=(16, 32)))) \
        # Common losses
        .add_stage(Stage(module=centernet_detection_loss, in_=centernet_detection_heads.get_outputs()))\
        .add_stage(Stage(module=centernet_pose_loss, in_=centernet_pose_heads.get_outputs()))\
        .add_stage(Stage(module=mask_loss, in_=mask_head.get_outputs())) \
        .add_stage(Stage(module=yolo3_detection_loss, in_=yolo3_detection_heads.get_outputs())) \
        # Extra losses
        .add_stage(Stage(module=centernet_bbox_decoder, in_=centernet_detection_heads.get_outputs())) \
        .add_stage(Stage(config=giou_centernet_detection_loss, in_=centernet_bbox_decoder.get_outputs())) \
        .add_stage(Stage(config=giou_yolo3_detection_loss, in_=yolo3_detection_heads.get_outputs())) \
    .create()

prediction_model = ModelPipelineBuilder\
    .with_models([dla, base_fpn, roi_align, masks_fpn,
                  centernet_detection_heads, centernet_pose_heads,
                  mask_head]) \
        .add_stage(Stage(module=dla, in_=InputType.Image)) \
        .add_stage(Stage(module=base_fpn, in_=dla.get_outputs(strides=(8, 16, 32)))) \
        .add_stage(Stage(module=masks_fpn, in_=dla.get_outputs(strides=(4, 8, 16, 32)))) \
        .add_stage(Stage(module=roi_align, in_=masks_fpn.get_outputs(strides=(4, 8, 16, 32)))) \
        # Heads
        .add_stage(Stage(module=centernet_detection_heads, in_=base_fpn.get_outputs())) \
        .add_stage(Stage(module=centernet_pose_heads, in_=base_fpn.get_outputs())) \
        .add_stage(Stage(module=mask_head, in_=roi_align.get_outputs())) \
        .add_stage(Stage(module=yolo3_detection_heads, in_=base_fpn.get_outputs(strides=(16, 32)))) \
        # Heads decoders
        .add_stage(Stage(module=centernet_bbox_decoder, in_=centernet_detection_heads.get_outputs()))
        .add_stage(Stage(module=centernet_pose_decoder, in_=centernet_pose_heads.get_outputs()))
        .add_stage(Stage(module=mask_decoder, in_=mask_head.get_outputs())) \
        .add_stage(Stage(module=yolo3_bbox_decoder, in_=yolo3_detection_heads.get_outputs())) \
    .create()
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
formatters = [
    CenterNetHeadFormatterConfig()
]
```

Ground truth encoding part:
```python

```
