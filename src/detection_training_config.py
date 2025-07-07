import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    MultiStepLR,
)
import math


class DetectionTrainingConfig:
    """Training configuration for object detection models"""

    def __init__(self, model_type="yolo", num_epochs=100, batch_size=16, lr=0.001):
        self.model_type = model_type.lower()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_optimizer_config(self, model):
        """Get optimizer configuration based on model type"""

        # Base learning rates by model type
        base_lr_configs = {
            "yolo": {"base_lr": 0.01, "weight_decay": 0.0005},
            "retinanet": {"base_lr": 0.01, "weight_decay": 0.0001},
            "fasterrcnn": {"base_lr": 0.005, "weight_decay": 0.0001},
            "detr": {"base_lr": 0.0001, "weight_decay": 0.0001},
            "efficientdet": {"base_lr": 0.008, "weight_decay": 0.00004},
            "ssd": {"base_lr": 0.16, "weight_decay": 0.0005},
        }

        config = base_lr_configs.get(self.model_type, base_lr_configs["yolo"])
        config["base_lr"] = self.lr
        # Adjust learning rate based on batch size (linear scaling rule)
        adjusted_lr = config["base_lr"] * (self.batch_size / 16)

        return {
            "lr": adjusted_lr,
            "weight_decay": config["weight_decay"],
            "momentum": 0.9 if self.model_type in ["yolo", "ssd"] else 0.937,
        }

    def setup_optimizer(self, model):
        """Setup optimizer with proper parameter grouping"""
        config = self.get_optimizer_config(model)

        # Parameter grouping for different learning rates
        param_groups = []

        # Backbone parameters (lower learning rate)
        backbone_params = []
        # Head parameters (higher learning rate)
        head_params = []
        # Bias parameters (no weight decay)
        bias_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "bias" in name:
                bias_params.append(param)
            elif "backbone" in name or "features" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # Create parameter groups
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": config["lr"],
                    "weight_decay": config["weight_decay"],
                }
            )
        
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": config["lr"] * 0.1,  # Lower LR for backbone
                    "weight_decay": config["weight_decay"],
                }
            )

        if bias_params:
            param_groups.append(
                {
                    "params": bias_params,
                    "lr": config["lr"],
                    "weight_decay": 0.0,  # No weight decay for bias
                }
            )

        # If no specific grouping, use all parameters
        if not param_groups:
            param_groups = [{"params": model.parameters(), **config}]

        # Choose optimizer based on model type
        if self.model_type == "detr":
            optimizer = optim.AdamW(
                param_groups, lr=config["lr"], weight_decay=config["weight_decay"]
            )
        else:
            optimizer = optim.SGD(
                param_groups,
                lr=config["lr"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )

        return optimizer

    def get_scheduler_config(self, optimizer, steps_per_epoch):
        """Get learning rate scheduler configuration"""

        total_steps = self.num_epochs * steps_per_epoch

        scheduler_configs = {
            "cosine_warm_restarts": lambda: {
                "scheduler": CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=10,  # Restart every 10 epochs
                    T_mult=2,  # Double the period after each restart
                    eta_min=1e-6,
                ),
                "step_type": "epoch",
                "monitor": "val_loss",
            },
            "one_cycle": lambda: {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=self.get_optimizer_config(None)["lr"],
                    total_steps=total_steps,
                    pct_start=0.3,  # 30% warmup
                    anneal_strategy="cos",
                    cycle_momentum=True,
                    base_momentum=0.85,
                    max_momentum=0.95,
                    div_factor=25.0,
                    final_div_factor=10000.0,
                ),
                "step_type": "batch",
                "monitor": None,
            },
            "reduce_on_plateau": lambda: {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="max",  # Maximizing mAP
                    factor=0.5,
                    patience=5,
                    verbose=True,
                    threshold=0.001,
                    threshold_mode="abs",
                    cooldown=0,
                    min_lr=1e-7,
                ),
                "step_type": "epoch",
                "monitor": "val_map",
            },
            "multi_step": lambda: {
                "scheduler": MultiStepLR(
                    optimizer,
                    milestones=[int(0.6 * self.num_epochs), int(0.8 * self.num_epochs)],
                    gamma=0.1,
                ),
                "step_type": "epoch",
                "monitor": None,
            },
        }

        # Recommended scheduler by model type
        recommended_schedulers = {
            "yolo": "cosine_warm_restarts",
            "retinanet": "reduce_on_plateau",
            "fasterrcnn": "multi_step",
            "detr": "one_cycle",
            "efficientdet": "cosine_warm_restarts",
            "ssd": "multi_step",
        }

        scheduler_name = recommended_schedulers.get(
            self.model_type, "cosine_warm_restarts"
        )
        return scheduler_configs[scheduler_name]()
