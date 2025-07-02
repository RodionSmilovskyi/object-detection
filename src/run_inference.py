#!/usr/bin/env python
# pylint: skip-file
import os
import sys
import torch as T
import onnxruntime
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torch.utils.tensorboard import SummaryWriter
from datasets.yolo_dataset import YoloDataset, detection_collate_fn
from detection.engine import evaluate, train_one_epoch
from common import (
    UintSsdLite,
    generate_samples,
    load_labels_from_json,
    make_ssdlite_model,
    remove_directory_contents,
)

WORKDIR = os.path.dirname(os.path.abspath(__file__))

def infer(params):
    writer = SummaryWriter(os.path.join(params["tensorboard_dir"], params["prefix"]))
    labels = load_labels_from_json(os.path.join(params["config_dir"], "classes.json"))
    model_name = f"model_uint8_{params["final_height"]}_{params["final_width"]}.onnx"
    transforms = v2.Compose(
        [
            v2.Resize(size=[params["final_height"], params["final_width"]]),
        ]
    )
    test_dataset = YoloDataset(params["validation_dir"], transforms, T.device("cpu"))
    ort_session = onnxruntime.InferenceSession(
        os.path.join(params["model_dir"], model_name),
        providers=["CPUExecutionProvider"],
    )

    inference_dir = os.path.join(params["output_dir"], "data", "uint8")
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
    else:
        remove_directory_contents(inference_dir)

    for image_idx, (image, _) in enumerate(test_dataset):
        fn, ext = os.path.splitext(os.path.basename(test_dataset.annotations[image_idx]))
        inputs = {"images": image.unsqueeze(0).numpy()}
        output = ort_session.run(None, inputs)
        indices = output[1] > 0.6
        pred_labels = [
            f"{labels[label]}: {score:.3f}"
            for label, score in zip(output[2][indices], output[1][indices])
        ]
        output_image = draw_bounding_boxes(
            image,
            T.tensor(output[0][indices], dtype=T.float32, device=T.device("cpu")),
            pred_labels,
            colors="red",
            width=2,
        )
        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0))
        plt.savefig(os.path.join(inference_dir, f"{image_idx}.jpg"))
        writer.add_image(f"validation/uint_{fn}.jpg", output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()

    print(f"Prefix {args.prefix}")
    print(f"Training seed {args.seed}")
    print(f"Final image width {args.width}")
    print(f"Final image height {args.height}")

    
    np.random.seed(args.seed)
    random.seed(args.seed)
    T.manual_seed(args.seed)
    T.use_deterministic_algorithms(True)
    
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")

    if not os.path.exists(os.environ["SM_OUTPUT_DIR"]):
        os.makedirs(os.environ["SM_OUTPUT_DIR"])

    if not os.path.exists(os.environ["SM_MODEL_DIR"]):
        os.makedirs(os.environ["SM_MODEL_DIR"])

    checkpoint_dir = (
        os.environ["CHECKPOINT_DIR"]
        if "CHECKPOINT_DIR" in os.environ
        else "/opt/ml/checkpoints"
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    tensorboard_dir = (
        os.environ["TENSORBOARD_DIR"]
        if "TENSORBOARD_DIR" in os.environ
        else "/opt/ml/output/tensorboard"
    )

    tmp_dir = os.environ["TMP_DIR"] if "TMP_DIR" in os.environ else "/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    infer(
        {
            "prefix": args.prefix,
            "train_dir": os.path.join(os.environ["SM_CHANNEL_TRAIN"]),
            "validation_dir": os.path.join(os.environ["SM_CHANNEL_VALIDATION"]),
            "config_dir": os.path.join(os.environ["SM_CHANNEL_CONFIG"]),
            "output_dir": os.environ["SM_OUTPUT_DIR"],
            "model_dir": os.environ["SM_MODEL_DIR"],
            "checkpoint_dir": checkpoint_dir,
            "tensorboard_dir": tensorboard_dir,
            "final_height": args.height,
            "final_width": args.width,
            "tmp_dir": tmp_dir,
            "device": device,
        }
    )

    print("SUCCESS")
    sys.exit(0)