import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead


def get_model(num_classes, nms_thresh=0.3, weights=None):
    """
    Faster R-CNN (ResNet50 + FPN) for license plate detection
    with pretrained COCO weights and custom anchors.
    """
    # 1. Load pretrained model on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights
    )

    # 2. Replace the box predictor (for your number of classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 3. Define custom anchor generator
    # Wide aspect ratios for license plates
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    model.rpn.anchor_generator = anchor_generator

    # 4. Reinitialize RPN head to match number of anchors
    out_channels = model.backbone.out_channels  # usually 256 for FPN
    num_anchors = len(aspect_ratios[0]) * len(anchor_sizes[0])  # 5 aspect ratios × 1 size = 5
    model.rpn.head = RPNHead(out_channels, num_anchors)

    # 5. Adjust NMS and score thresholds
    model.roi_heads.nms_thresh = nms_thresh
    model.roi_heads.score_thresh = 0.05

    return model