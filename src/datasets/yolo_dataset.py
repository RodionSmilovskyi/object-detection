import os
import torch as T
from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as F

def cxcywh_to_xyxy(cx, cy, w, h):
    """
    Converts bounding box coordinates from center-x, center-y, width, height (cxcywh)
    to top-left x, top-left y, bottom-right x, bottom-right y (xyxy).

    Args:
        cx (float): Center-x coordinate.
        cy (float): Center-y coordinate.
        w (float): Width of the bounding box.
        h (float): Height of the bounding box.

    Returns:
        tuple (float, float, float, float): Top-left x, top-left y, bottom-right x, bottom-right y.
    """
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2

def detection_collate_fn(batch):
    """
    Custom collate function for object detection dataloader.

    Args:
        batch: List of tuples (image, target)
            - image: torch.Tensor of shape (C, H, W)
            - target: dict containing bboxes, labels, etc. with varying sizes

    Returns:
        images: torch.Tensor of shape (batch_size, C, H, W)
        targets: List of dicts, where each dict contains the target for an image
    """
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])
        targets.append(sample[1])

    # Stack images into a single tensor
    images = T.stack(images, dim=0)

    return images, targets

class YoloDataset(Dataset):
    def __init__(self, root, transforms, device):
        self.root = root
        self.transforms = transforms
        self.annotations = []
        self.imgs = []
        self.device = device
        image_extensions = [".jpg", ".jpeg"]
        
        workdir = self.root
        text_files = [f for f in os.listdir(workdir) if f.endswith(".txt") and f != 'classes.txt']

        for tf in text_files:
            fn, ext = os.path.splitext(os.path.basename(tf))

            for ie in image_extensions:
                if os.path.exists(os.path.join(workdir, fn + f"{ie}")):
                    self.annotations.append(os.path.join(workdir, tf))
                    self.imgs.append(os.path.join(workdir, fn + f"{ie}"))
                        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = read_image(self.imgs[idx], mode=ImageReadMode.RGB)
        img = tv_tensors.Image(img, device=self.device)
        labels = []
        boxes = []
        areas = []
        img_width = img.shape[2]
        img_height = img.shape[1]

        with open(os.path.join(self.root, self.annotations[idx]), "r") as f:
            for line in f:
                line = line.strip().split(" ")
                ind = int(line[0])
                labels.append(ind + 1)
                cx = float(line[1]) * img_width
                cy = float(line[2]) * img_height
                w = float(line[3]) * img_width
                h = float(line[4]) * img_height
                x1, y1, x2, y2 = cxcywh_to_xyxy(cx, cy, w, h)
                boxes.append([x1, y1, x2, y2])
                areas.append(w * h)

        iscrowd = T.zeros((len(boxes),), dtype=T.int64)

        target = {
            "labels": T.tensor(labels, dtype=T.long, device = self.device),
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(img), device=self.device
            ),
            "image_id": idx,
            "area": T.tensor(areas, dtype=T.float32, device=self.device),
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if img.shape[0] == 4:
            pass

        return img, target