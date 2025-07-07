# pylint: disable=C0114, C0413, E0401, W0621, E1101
import os
import json
import shutil
import random
import cv2
import torch as T
import albumentations as A
import torchvision
import numpy as np
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.models.detection.ssdlite import SSDLiteHead
from functools import partial

from transforms.add_random_background import BACKGROUND_DIR, AddRandomBackground
from transforms.merge_images import merge_images_horizontally, read_annotation


def make_ssdlite_model(labels, trainable_backbone_layers=0):
    num_classes = len(labels)
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weight="COCO_V1", trainable_backbone_layers=trainable_backbone_layers
    )
    in_channels = retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(T.nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head = SSDLiteHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=norm_layer,
    )
    



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
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
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
    except Exception as e:
        print(f"An error occurred while writing to '{filepath}': {e}")


def generate_negative_samples(
    num_images: int,
    width: int,
    height: int,
    output_dir: str,
    methods: list = ["solid", "horizontal_gradient", "vertical_gradient", "noise"],
    file_prefix: str = "background",
):
    """
    Generates and saves background images as JPGs with corresponding empty
    TXT label files for use as negative samples.

    Args:
        num_images (int): The number of negative samples to generate.
        width (int): The width of the images.
        height (int): The height of the images.
        output_dir (str): The directory to save the files in.
        methods (list): A list of background generation methods to use.
        file_prefix (str): The prefix for the saved filenames.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {num_images} negative samples in '{output_dir}'...")

    for i in range(num_images):
        # Choose a random generation method
        method = random.choice(methods)

        # Create an empty image array (RGB)
        bg_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Generate the background based on the chosen method
        if method == "solid":
            random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            bg_img[:] = random_color

        elif method == "horizontal_gradient":
            start_color = np.random.randint(0, 256, size=3)
            end_color = np.random.randint(0, 256, size=3)
            for y in range(height):
                alpha = y / (height - 1)
                color = ((1 - alpha) * start_color + alpha * end_color).astype(np.uint8)
                bg_img[y, :] = color

        elif method == "vertical_gradient":
            start_color = np.random.randint(0, 256, size=3)
            end_color = np.random.randint(0, 256, size=3)
            for x in range(width):
                alpha = x / (width - 1)
                color = ((1 - alpha) * start_color + alpha * end_color).astype(np.uint8)
                bg_img[:, x] = color

        elif method == "noise":
            bg_img = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

        # --- Save the image and create the empty label file ---
        # Define the base filename (e.g., background_0001)
        base_filename = f"{file_prefix}_{i+1:04d}"

        # 1. Save the JPG image
        image_filename = f"{base_filename}.jpg"
        image_filepath = os.path.join(output_dir, image_filename)
        image_to_save = cv2.cvtColor(
            bg_img, cv2.COLOR_RGB2BGR
        )  # Convert to BGR for OpenCV
        cv2.imwrite(image_filepath, image_to_save)

        # 2. Create the empty TXT file
        txt_filename = f"{base_filename}.txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        with open(txt_filepath, "w") as f:
            pass  # Creates an empty file

    print("✅ Generation complete!")


def generate_samples(source_dir, target_dir, json_file_path, width, height, n=1):
    class_list = load_labels_from_json(json_file_path)
    annotations = [f for f in os.listdir(target_dir) if f.endswith(".txt")]

    augmentation_transform = A.Compose(
        [
            # Geometric transformations
            A.OneOf(
                [
                    A.Resize(height, width, p=0.2),
                    A.RandomResizedCrop(
                        size=(height, width),
                        scale=(0.7, 1.0),
                        ratio=(0.8, 1.2),
                        interpolation=cv2.INTER_LINEAR,
                        p=0.8,
                    ),
                ], p=1
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),  # Rare for groceries
            A.Rotate(
                limit=15,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.6,
            ),
            # Perspective and distortion
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
            A.ElasticTransform(
                alpha=50,
                sigma=10,
                interpolation=cv2.INTER_LINEAR,
                p=0.2,
            ),
            # Lighting and color - conservative for color preservation
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=10,  # Small hue shifts
                sat_shift_limit=15,  # Small saturation shifts
                val_shift_limit=15,  # Small value shifts
                p=0.4,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # Lighting simulation
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1,2),
                shadow_dimension=5,
                p=0.2,
            ),
            A.RandomSunFlare(
                num_flare_circles_range=(1,2),
                p=0.1,
            ),
        ],
        bbox_params=A.BboxParams("yolo", ["class_labels"]),
    )

    resize_transform = A.Compose(
        [A.Resize(height, width)], bbox_params=A.BboxParams("yolo", ["class_labels"])
    )

    for annotation in annotations:
        fn, ext = os.path.splitext(os.path.basename(annotation))
        image, bboxes, category_ids, class_labels = get_image_for_transform(
            annotation, target_dir, class_list
        )

        resized_image = resize_transform(
            image=image, bboxes=bboxes, class_labels=class_labels
        )
        save_transform(resized_image, category_ids, target_dir, fn)

        for i in range(n):
            augmented_image = augmentation_transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            save_transform(augmented_image, category_ids, target_dir, f"{fn}-{i}")


class UintSsdLite(T.nn.Module):
    def __init__(self, ssd_lite_model):
        super().__init__()
        self.ssd_lite_model = ssd_lite_model

    def forward(self, img_uint8: T.tensor):
        image_float = img_uint8.to(T.float32)
        image_float = image_float / 255.0

        return self.ssd_lite_model.forward(image_float)
