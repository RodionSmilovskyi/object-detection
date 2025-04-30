# pylint: disable=C0114, C0413, E0401, W0621, E1101
from albumentations.core.transforms_interface import ImageOnlyTransform
import os
import numpy as np
import cv2

BACKGROUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backgrounds")

class AddRandomBackground(ImageOnlyTransform):
    """Transform which replaces image background"""
    def __init__(
        self,
        background_images,
        p=1.0,
    ):
        # Only probability needed for initialization
        super().__init__(p=p)
        self.background_images = background_images

    def get_params_dependent_on_data(self, params, data):
        mask = np.zeros_like(data["image"])
        height = params["shape"][0]
        width = params["shape"][1]
        background = self.py_random.choice(self.background_images)

        for x1, y1, x2, y2 in data["bboxes"][:, :4]:
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            mask[y1:y2, x1:x2] = 255

        factor = self.py_random.uniform(0.5, 0.8)
        return {"mask": mask, "background": background, "factor": factor}

    def apply(self, img, mask, background, factor, **params):
        blended = cv2.addWeighted(img, factor, background, 1-factor, 0)
        result = np.where(mask == 255, blended, background)
        return result
