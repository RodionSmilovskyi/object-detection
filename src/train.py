#!/usr/bin/env python
# pylint: skip-file
import os
import sys
import torch as T
import onnxruntime
import argparse
import random
import shutil
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
    generate_negative_samples,
    generate_samples,
    load_labels_from_json,
    make_ssdlite_model,
    remove_directory_contents,
)
from detection_training_config import DetectionTrainingConfig

WORKDIR = os.path.dirname(os.path.abspath(__file__))


def train(params):
    writer = SummaryWriter(os.path.join(params["tensorboard_dir"], params["prefix"]))
    labels = load_labels_from_json(os.path.join(params["config_dir"], "classes.json"))
    model = make_ssdlite_model(labels, params["trainable_backbone_layers"])

    print("Obtained model")

    writer.add_text("description", params["comment"])

    train_transform = v2.Compose(
        [
            v2.ToDtype(T.float, scale=True),
            # v2.RandomResizedCrop(
            #     size=(params["final_height"], params["final_width"]),
            #     scale=(0.8, 1.0),  # Keep most of the image
            #     ratio=(0.8, 1.2)
            # ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(
                degrees=10,  # Small rotation to preserve object shape
                interpolation=v2.InterpolationMode.BILINEAR,
            ),
            # # Color transforms - subtle to preserve color information
            # v2.ColorJitter(
            #     brightness=0.15,    # Subtle brightness changes
            #     contrast=0.15,      # Subtle contrast changes
            #     saturation=0.10,    # Minimal saturation change
            #     hue=0.05           # Very minimal hue shift
            # ),
            v2.SanitizeBoundingBoxes(),
            v2.ToPureTensor(),
        ]
    )

    test_transforms = v2.Compose(
        [
            v2.Resize(size=[params["final_height"], params["final_width"]]),
            v2.ToDtype(T.float, scale=True),
            v2.ToPureTensor(),
        ]
    )

    model.train()
    model.to(params["device"])

    training_config = DetectionTrainingConfig(
        "ssd", params["rounds"] * params["epochs"], params["batch_size"], params["lr"]
    )
    optimizer = training_config.setup_optimizer(model)
    scheduler_config = training_config.get_scheduler_config(optimizer, 1)

    train_dir = params["train_dir"]
    test_dir = params["validation_dir"]

    global_epoch = 0
    best_mAP = 0.0
    best_mAR = 0.0

    for round in range(params["rounds"]):
        remove_directory_contents(params["tmp_dir"])
        print("Cleaned temp directory")

        shutil.copytree(train_dir, params["tmp_dir"], dirs_exist_ok=True)
        print("Copied train samples to target directory")

        # generate_negative_samples(10, params["final_width"], params["final_height"], params["tmp_dir"])
        generate_samples(
            train_dir,
            params["tmp_dir"],
            os.path.join(params["config_dir"], "classes.json"),
            params["final_height"],
            params["final_width"],
            params["samples_per_image"],
        )

        train_dataset = YoloDataset(
            params["tmp_dir"], train_transform, params["device"]
        )
        data_loader_train = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            collate_fn=detection_collate_fn,
            shuffle=True,
            drop_last=True,
        )

        test_dataset = YoloDataset(test_dir, test_transforms, params["device"])
        data_loader_test = DataLoader(
            test_dataset,
            batch_size=2,
            collate_fn=detection_collate_fn,
            shuffle=True,
            drop_last=True,
        )

        for epoch in range(params["epochs"]):
            ml = train_one_epoch(
                model,
                optimizer,
                data_loader_train,
                params["device"],
                epoch,
                print_freq=10,
                scheduler=scheduler_config["scheduler"],
            )

            report = evaluate(model, data_loader_test, device=params["device"])
            writer.add_scalar(
                "train/box_regression", ml.meters["bbox_regression"].avg, global_epoch
            )
            writer.add_scalar(
                "train/classification", ml.meters["classification"].avg, global_epoch
            )
            writer.add_scalar("train/loss", ml.meters["loss"].avg, global_epoch)
            writer.add_scalar(
                "eval/avg_precision", report.coco_eval["bbox"].stats[0], global_epoch
            )
            writer.add_scalar(
                "eval/ap50", report.coco_eval["bbox"].stats[1], global_epoch
            )
            writer.add_scalar(
                "eval/avg_recall", report.coco_eval["bbox"].stats[8], global_epoch
            )

            current_mAP = report.coco_eval["bbox"].stats[0]
            current_mAR = report.coco_eval["bbox"].stats[8]

            if current_mAP > best_mAP:
                best_mAP = current_mAP
                T.save(
                    model.state_dict(),
                    os.path.join(params["model_dir"], "best_model.pth"),
                )
                print(
                    f"*** New best model saved with mAP: {best_mAP:.4f}, epoch {epoch}, global epoch {global_epoch}***"
                )

            if current_mAR > best_mAR:
                best_mAR = current_mAR

            global_epoch = global_epoch + 1
            pass

    writer.add_hparams(
        {
            "prefix": params["prefix"],
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "rounds": params["rounds"],
            "trainable_backbone_layers": params["trainable_backbone_layers"],
            "lr": params["lr"],
            "samples_per_image": params["samples_per_image"],
            "comment": params["comment"],
        },
        {"hparam/mAP": best_mAP, "hparam/mAR": best_mAR},
    )

    checkpoint = T.load(os.path.join(params["model_dir"], "best_model.pth"))
    model.load_state_dict(checkpoint)
    model.to(T.device("cpu"))

    dummy_float_image = T.zeros(
        (1, 3, params["final_height"], params["final_width"]),
        device=T.device("cpu"),
        dtype=T.float32,
    )

    model_name = f"model_{params["final_height"]}_{params["final_width"]}.onnx"
    T.onnx.export(
        model,
        (dummy_float_image,),
        os.path.join(params["model_dir"], model_name),
        opset_version=18,
    )

    uint_model = UintSsdLite(model)
    unit_model_name = (
        f"model_uint8_{params["final_height"]}_{params["final_width"]}.onnx"
    )
    dummy_uint_image = T.zeros(
        (1, 3, params["final_height"], params["final_width"]),
        device=T.device("cpu"),
        dtype=T.uint8,
    )
    T.onnx.export(
        uint_model,
        (dummy_uint_image,),
        os.path.join(params["model_dir"], unit_model_name),
        opset_version=18,
        input_names=["images"],
    )

    check_inference(params, writer)
    check_unit_inference(params, writer)


def check_inference(params, writer: SummaryWriter):
    labels = load_labels_from_json(os.path.join(params["config_dir"], "classes.json"))
    model_name = f"model_{params["final_height"]}_{params["final_width"]}.onnx"
    transforms = v2.Compose(
        [
            v2.Resize(size=[params["final_height"], params["final_width"]]),
            v2.ToDtype(T.float, scale=True),
        ]
    )
    test_dataset = YoloDataset(params["validation_dir"], transforms, T.device("cpu"))
    ort_session = onnxruntime.InferenceSession(
        os.path.join(params["model_dir"], model_name),
        providers=["CPUExecutionProvider"],
    )

    inference_dir = os.path.join(params["output_dir"], "data", "float32")

    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
    else:
        remove_directory_contents(inference_dir)

    for image_idx, (image, _) in enumerate(test_dataset):
        fn, ext = os.path.splitext(
            os.path.basename(test_dataset.annotations[image_idx])
        )
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
        writer.add_image(f"validation/float_{fn}.jpg", output_image)


def check_unit_inference(params, writer: SummaryWriter):
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
        fn, ext = os.path.splitext(
            os.path.basename(test_dataset.annotations[image_idx])
        )
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--samples-per-image", type=int, default=1)
    parser.add_argument("--trainable-backbone-layers", type=int, default=0)
    parser.add_argument("--comment", type=str, default="")
    args = parser.parse_args()

    print(f"Prefix {args.prefix}")
    print(f"Training rounds {args.rounds}")
    print(f"Training epochs {args.epochs}")
    print(f"Training seed {args.seed}")
    print(f"Learning rate {args.lr}")
    print(f"Batch size {args.batch_size}")
    print(f"Final image width {args.width}")
    print(f"Final image height {args.height}")
    print(f"Samples per image {args.samples_per_image}")
    print(f"Trainable backbone layers {args.trainable_backbone_layers}")
    print(f"Comment {args.comment}")

    np.random.seed(args.seed)
    random.seed(args.seed)
    T.manual_seed(args.seed)
    # T.use_deterministic_algorithms(True)

    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    if T.cuda.is_available():
        T.cuda.manual_seed(args.seed)
        T.cuda.manual_seed_all(args.seed)
        # T.backends.cudnn.deterministic = True
        # T.backends.cudnn.benchmark = False
        print(f"Number of available GPUs: {T.cuda.device_count()}")
    else:
        print("No CUDA-enabled GPU is available.")

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

    train(
        {
            "prefix": args.prefix,
            "train_dir": os.path.join(os.environ["SM_CHANNEL_TRAIN"]),
            "validation_dir": os.path.join(os.environ["SM_CHANNEL_VALIDATION"]),
            "config_dir": os.path.join(os.environ["SM_CHANNEL_CONFIG"]),
            "output_dir": os.environ["SM_OUTPUT_DIR"],
            "model_dir": os.environ["SM_MODEL_DIR"],
            "checkpoint_dir": checkpoint_dir,
            "tensorboard_dir": tensorboard_dir,
            "tmp_dir": tmp_dir,
            "batch_size": args.batch_size,
            "final_height": args.height,
            "final_width": args.width,
            "epochs": args.epochs,
            "rounds": args.rounds,
            "trainable_backbone_layers": args.trainable_backbone_layers,
            "lr": args.lr,
            "samples_per_image": args.samples_per_image,
            "comment": args.comment,
            "device": device,
        }
    )

    print("SUCCESS")
    sys.exit(0)
