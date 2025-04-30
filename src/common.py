# pylint: disable=C0114, C0413, E0401, W0621, E1101
import os
import json
import shutil
import random
import cv2
import torch as T
import albumentations as A
import torchvision
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.models.detection.ssdlite import (
    SSDLiteClassificationHead,
    SSDLiteRegressionHead
)

from transforms.add_random_background import BACKGROUND_DIR, AddRandomBackground
from transforms.merge_images import merge_images_horizontally, read_annotation

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
    
def remove_directory_contents(directory_path):
    """Removes all files and subdirectories within the specified directory.

    Args:
        directory_path (str): The path to the directory.
    """
    try:
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Removed file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Removed directory: {item_path}")
        print(f"Successfully removed all contents from: {directory_path}")
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory_path}")
    except OSError as e:
        print(f"Error removing contents of {directory_path}: {e}")

def get_image_for_transform(annotation, source_dir, class_list):
    fn, ext = os.path.splitext(os.path.basename(annotation))
    img_name = fn + ".jpg"
    image = cv2.imread(os.path.join(source_dir, img_name), cv2.IMREAD_COLOR_RGB)
    bboxes, category_ids = read_annotation(os.path.join(source_dir, annotation))

    return image, bboxes, category_ids, [class_list[i] for i in category_ids]

def save_transform(data, category_ids, target_dir, fn):
    lines = []
    for cat_id, bbox in zip(category_ids, data["bboxes"]):
        lines.append(f"{cat_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

    new_annotation_name = f"{fn}.txt"
    new_image_name = f"{fn}.jpg"
    write_lines_to_file(os.path.join(target_dir, new_annotation_name), lines)
    cv2.imwrite(
        os.path.join(target_dir, new_image_name),
        cv2.cvtColor(data["image"], cv2.COLOR_BGR2RGB),
    )
    
def write_lines_to_file(filepath, lines):
    """Creates a file (or overwrites if it exists) and writes a list of lines to it.

    Args:
        filepath (str): The path to the file to create or write to.
        lines (list of str): A list of strings, where each string will be written as a new line.
    """
    try:
        with open(filepath, "w") as f:  # 'w' mode for writing (overwrites if exists)
            for line in lines:
                f.write(line + "\n")  # Add a newline character after each line
        print(f"Successfully wrote {len(lines)} lines to '{filepath}'")
    except Exception as e:
        print(f"An error occurred while writing to '{filepath}': {e}")
        
def generate_samples(source_dir, target_dir, json_file_path, width, height, n=1):
    remove_directory_contents(target_dir)
    print("Cleaned target directory")   

    class_list = load_labels_from_json(json_file_path)
    annotations = [
        f for f in os.listdir(source_dir) if f.endswith(".txt")
    ]
    bckg_paths = [
        os.path.join(BACKGROUND_DIR, f)
        for f in os.listdir(BACKGROUND_DIR)
        if f.endswith(".jpg")
    ]
    background_images = [
        cv2.resize(cv2.imread(path), (width, height)) for path in bckg_paths
    ]
    
    augmentation_transform = A.Compose(
        [
            A.Affine(
                scale=(0.6, 1.2),  # Zoom in/out by 80-120%
                rotate=(-360, 360),  # Rotate by -15 to +15 degrees
                translate_percent=(0.05, 0.05),  # Optional: translate by 0-10%
                shear=(-2, 2),  # Optional: shear by -10 to +10 degrees
                p=0.5,
            ),
            A.CoarseDropout(num_holes_range=(1, 2)),
            AddRandomBackground(background_images, p=0.5),
            A.Resize(height, width, p=1),
        ],
        bbox_params=A.BboxParams("yolo", ["class_labels"]),
    )

    resize_transform = A.Compose(
        [A.Resize(height, width)], bbox_params=A.BboxParams("yolo", ["class_labels"])
    )
    
    for annotation in annotations:
        fn, ext = os.path.splitext(os.path.basename(annotation))
        image, bboxes, category_ids, class_labels = get_image_for_transform(
            annotation, source_dir, class_list
        )

        resized_image = resize_transform(
            image=image, bboxes=bboxes, class_labels=class_labels
        )
        save_transform(resized_image, category_ids, target_dir, fn)

        for i in range(n):
            annotation_to_merge = random.choice(
                [f for f in annotations if f != annotation]
            )
            fn_m, ext_m = os.path.splitext(os.path.basename(annotation_to_merge))
            merged_image = merge_images_horizontally(
                os.path.join(source_dir, annotation),
                os.path.join(source_dir, annotation_to_merge),
                class_list,
            )
            augmented_merged_image = augmentation_transform(
                image=merged_image["image"],
                bboxes=merged_image["bboxes"],
                class_labels=merged_image["class_labels"],
            )

            save_transform(
                augmented_merged_image,
                merged_image["category_ids"],
                target_dir,
                f"{fn}-{fn_m}-{i}",
            )

            augmented_image = augmentation_transform(
                image=resized_image["image"],
                bboxes=resized_image["bboxes"],
                class_labels=resized_image["class_labels"],
            )
            save_transform(augmented_image, category_ids, target_dir, f"{fn}-{i}")
    
