#!/usr/bin/env python
# pylint: skip-file
import os
import sys
import torch as T
import argparse
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from datasets.yolo_dataset import YoloDataset, detection_collate_fn
from detection.engine import evaluate, train_one_epoch
from common import generate_samples, load_labels_from_json, make_ssdlite_model

WORKDIR = os.path.dirname(os.path.abspath(__file__))


def train(params):
    labels = load_labels_from_json(os.path.join(params["config_dir"], "classes.json"))
    model = make_ssdlite_model(labels)

    print("Obtained model")

    train_transform = v2.Compose(
        [
            v2.RandomRotation(degrees=15),
            v2.SanitizeBoundingBoxes(),
            v2.Resize(size=[params["final_height"], params["final_width"]]),
            v2.ToDtype(T.float, scale=True),
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = T.optim.SGD(trainable_params, lr=params["lr"])
    scheduler = T.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["rounds"] * params["epochs"], eta_min=0.01
    )
    train_dir = params["train_dir"]
    test_dir = params["validation_dir"]

    for round in range(params["rounds"]):
        generate_samples(train_dir, params["tmp_dir"], os.path.join(params["config_dir"], "classes.json"), params["final_height"], params["final_width"], params["samples_per_image"])
        
        train_dataset = YoloDataset(
            params["tmp_dir"], labels, train_transform, params["device"]
        )
        data_loader_train = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            collate_fn=detection_collate_fn,
            shuffle=True,
            drop_last=True,
        )
        
        test_dataset = YoloDataset(test_dir, labels, test_transforms, params["device"])
        data_loader_test = DataLoader(
            test_dataset,
            batch_size=2,
            collate_fn=detection_collate_fn,
            shuffle=True,
            drop_last=True,
        )

        for epoch in range(params["epochs"]):
            train_one_epoch(
                model,
                optimizer,
                data_loader_train,
                params["device"],
                epoch,
                print_freq=10,
            )
            scheduler.step()
            evaluate(model, data_loader_test, device=params["device"])

    model.to(T.device("cpu"))
    dummy_image = T.zeros(
        (1, 3, params["final_height"], params["final_width"]), device=T.device("cpu"), dtype=T.float32
    )
    # images, _ = next(iter(data_loader_train))
    model_name = f"model_{params["final_height"]}_{params["final_width"]}.onnx"
    T.onnx.export(
        model,
        (dummy_image,),
        os.path.join(params["model_dir"], model_name),
        opset_version=18,
    )


if __name__ == "__main__":
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    if T.cuda.is_available():
        print(f"Number of available GPUs: {T.cuda.device_count()}")
    else:
        print("No CUDA-enabled GPU is available.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--samples-per-image", type=int, default=1)
    args = parser.parse_args()

    print(f"Training rounds {args.rounds}")
    print(f"Training epochs {args.epochs}")
    print(f"Learning rate {args.lr}")
    print(f"Batch size {args.batch_size}")
    print(f"Final image width {args.width}")
    print(f"Final image height {args.height}")
    print(f"Samples per image {args.samples_per_image}")

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

    tmp_dir = os.environ["TMP_DIR"] if "TMP_DIR" in os.environ else "/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    train(
        {
            "train_dir": os.path.join(os.environ["SM_CHANNEL_TRAIN"]),
            "validation_dir": os.path.join(os.environ["SM_CHANNEL_VALIDATION"]),
            "config_dir": os.path.join(os.environ["SM_CHANNEL_CONFIG"]),
            "output_dir": os.environ["SM_OUTPUT_DIR"],
            "model_dir": os.environ["SM_MODEL_DIR"],
            "checkpoint_dir": checkpoint_dir,
            "tmp_dir": tmp_dir,
            "batch_size": args.batch_size,
            "final_height": args.height,
            "final_width": args.width,
            "epochs": args.epochs,
            "rounds": args.rounds,
            "lr": args.lr,
            "samples_per_image": args.samples_per_image,
            "device": device,
        }
    )

    print("SUCCESS")
    sys.exit(0)
