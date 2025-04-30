# pylint: disable=C0114, C0413, E0401, W0621, E1101
import os
import cv2
import numpy as np
import albumentations as A

def read_annotation(annotation):
    bboxes = []
    category_ids = []
    with open(annotation, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            category_ids.append(int(line[0]))
            cx = float(line[1])
            cy = float(line[2])
            w = float(line[3])
            h = float(line[4])
            bboxes.append([cx, cy, w, h])

    return bboxes, category_ids

def merge_images_horizontally(source_file, file_to_merge, class_list):
    source_dir = os.path.dirname(source_file)
    source_name, source_ext = os.path.splitext(os.path.basename(source_file))
    merge_name, merge_ext = os.path.splitext(os.path.basename(file_to_merge))

    source_file = os.path.join(source_dir, source_name + ".txt")
    file_to_merge = os.path.join(source_dir, merge_name + ".txt")

    source_img = cv2.imread(
        os.path.join(source_dir, source_name + ".jpg"), cv2.IMREAD_COLOR_RGB
    )
    merge_img = cv2.imread(
        os.path.join(source_dir, merge_name + ".jpg"), cv2.IMREAD_COLOR_RGB
    )
    s_bb, s_cid = read_annotation(source_file)
    m_bb, m_cid = read_annotation(file_to_merge)

    transform = A.Compose(
        [A.Resize(640, 320)], bbox_params=A.BboxParams("yolo", ["class_labels"])
    )

    source_augmented = transform(
        image=source_img,
        bboxes=s_bb,
        category_ids=s_cid,
        class_labels=[class_list[i] for i in s_cid],
    )
    merge_augmented = transform(
        image=merge_img,
        bboxes=m_bb,
        category_ids=m_cid,
        class_labels=[class_list[i] for i in m_cid],
    )

    for sbb in source_augmented["bboxes"]:
        sbb[0] = sbb[0] / 2
        sbb[2] = sbb[2] / 2

    for mbb in merge_augmented["bboxes"]:
        mbb[0] = 0.5 + mbb[0] / 2
        mbb[2] = mbb[2] / 2

    merged = {
        "image": np.hstack((source_augmented["image"], merge_augmented["image"])),
        "category_ids": source_augmented["category_ids"]
        + merge_augmented["category_ids"],
        "class_labels": source_augmented["class_labels"]
        + merge_augmented["class_labels"],
        "bboxes": source_augmented["bboxes"] + merge_augmented["bboxes"],
    }

    return merged