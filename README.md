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
centernet_pose_encode_decode_config = CenternetPoseEncodeDecodeConfig()
mask_encoder_encode_decode_config = MaskEncodeDecodeConfig()
yolo3_bbox_encode_decode_config = Yolo3EncodeDecodeConfig()
```

Model configuration part:
```python
# Create models
dla = DLA34Config(pretrained='')
base_fpn = FPNConfig(inputs_count=3)
roi_align = RoiAlignConfig()
masks_fpn = SmallFPNConfig(inputs_count=4)
centernet_detection_heads = CenternetDetectionHeadsConfig()
centernet_pose_heads = CenternetPoseHeadsConfig()
mask_head = BboxSegmentationMaskHeadConfig()
yolo3_detection_heads = Yolo3DetectionsHeadConfig(fpn_strides=[16, 32])

# Create losses
centernet_detection_loss = CenternetDetectionLossConfig()
centernet_pose_loss = CenternetPoseLossConfig()
mask_loss = MaskLossConfig()
yolo3_detection_loss = Yolo3LossConfig()
giou_centernet_detection_loss = GIoULossConfig()
giou_yolo3_detection_loss = GIoULossConfig()

# Create detectors
detector = ModelPipelineBuilder \
        .add_stage(ModuleStage(dla, in_=InputType.Image)) \
        .add_stage(ModuleStage(base_fpn, in_=dla.get_outputs(strides=(8, 16, 32)))) \
        .add_stage(ModuleStage(masks_fpn, in_=dla.get_outputs(strides=(4, 8, 16, 32)))) \
        .add_stage(ModuleStage(roi_align, in_=masks_fpn.get_outputs(strides=(4, 8, 16, 32)))) \
        # Heads
        .add_stage(HeadStage(centernet_detection_heads, in_=base_fpn.get_outputs(), 
                             encoder_decoder=centernet_bbox_encode_decode_config, loss=centernet_detection_loss)) \
        .add_stage(HeadStage(centernet_pose_heads, in_=base_fpn.get_outputs(),
                             encoder_decoder=centernet_pose_encode_decode_config, loss=centernet_pose_loss)) \
        .add_stage(HeadStage(mask_head, in_=roi_align.get_outputs(),
                             encoder_decoder=mask_encoder_encode_decode_config, loss=mask_loss)) \
        .add_stage(HeadStage(yolo3_detection_heads, in_=base_fpn.get_outputs(strides=(16, 32)),
                             encoder_decoder=yolo3_bbox_encode_decode_config, loss=yolo3_detection_loss)) \
        # Extra losses
        .add_stage(LossStage(giou_centernet_detection_loss, in_=centernet_detection_heads.get_decode_outputs())) \
        .add_stage(LossStage(giou_yolo3_detection_loss, in_=yolo3_detection_heads.get_outputs())) \
    .create()
```