import json
import torch as T
import torchvision
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.models.detection.ssdlite import (
    SSDLiteClassificationHead,
    SSDLiteRegressionHead
)

def make_ssdlite_model(labels):
    num_classes = len(labels)
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weight="COCO_V1", trainable_backbone_layers = 0)
    out_channels = retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    classification_head = SSDLiteClassificationHead(
        in_channels=out_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=T.nn.BatchNorm2d
    )
    
    regression_head = SSDLiteRegressionHead(in_channels=out_channels, num_anchors=num_anchors, norm_layer=T.nn.BatchNorm2d)
    model.head.classification_head = classification_head
    model.head.regression_head = regression_head
    
    return model

def load_labels_from_json(json_file_path):
    """
    Loads labels from a JSON file into a list (array).

    Args:
        json_file_path (str): The path to the classes.json file.

    Returns:
        list: A list of labels, or None if the file could not be loaded.
    """
    try:
        with open(json_file_path, "r") as f:
            labels = json.load(f)
        return labels
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
        return None