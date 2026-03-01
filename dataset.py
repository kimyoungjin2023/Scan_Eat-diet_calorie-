"""
COCO 포맷 데이터셋 클래스 (Mask2Former 학습용)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


class COCOInstanceDataset(Dataset):
    """
    COCO JSON을 읽어 Mask2Former 학습에 필요한 형태로 반환
    반환:
        image       : PIL Image
        masks       : (N, H, W) bool tensor
        labels      : (N,) long tensor  (0-indexed class)
        boxes       : (N, 4) float tensor  [x1, y1, x2, y2]
    """

    def __init__(self, images_dir: str, annotation_json: str, processor=None):
        """
        Args:
            images_dir      : 이미지 폴더 경로
            annotation_json : COCO JSON 경로
            processor       : Mask2FormerImageProcessor (None이면 raw 반환)
        """
        self.images_dir = images_dir
        self.processor = processor

        self.coco = COCO(annotation_json)
        self.image_ids = sorted(self.coco.imgs.keys())

        # 유효 이미지만 (annotation이 1개 이상)
        self.image_ids = [
            iid for iid in self.image_ids
            if len(self.coco.getAnnIds(imgIds=iid)) > 0
        ]

        # 카테고리 id → 0-indexed label
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.imgs[image_id]

        # 이미지 로드
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        # 어노테이션 로드
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        masks, labels, boxes = [], [], []
        for ann in anns:
            # segmentation → binary mask
            rle = self.coco.annToRle(ann)
            m = coco_mask.decode(rle).astype(bool)   # (H, W)
            if m.sum() == 0:
                continue

            masks.append(m)
            labels.append(self.cat_id_to_label[ann["category_id"]])
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])

        if len(masks) == 0:
            # 빈 샘플 — DataLoader에서 collate_fn으로 걸러내거나 dummy 반환
            masks  = np.zeros((1, H, W), dtype=bool)
            labels = np.array([0], dtype=np.int64)
            boxes  = np.array([[0, 0, 1, 1]], dtype=np.float32)
        else:
            masks  = np.stack(masks, axis=0)           # (N, H, W)
            labels = np.array(labels, dtype=np.int64)
            boxes  = np.array(boxes,  dtype=np.float32)

        if self.processor is not None:
            # HuggingFace Mask2FormerImageProcessor 사용
            instance_seg = np.zeros((H, W), dtype=np.int32)
            for i, m in enumerate(masks):
                instance_seg[m] = i + 1   # 0 = background

            encoding = self.processor(
                images=image,
                segmentation_maps=Image.fromarray(instance_seg.astype(np.uint8)),
                instance2class_mapping={i + 1: int(labels[i]) for i in range(len(labels))},
                return_tensors="pt",
            )
            # batch 차원 제거
            return {k: v.squeeze(0) for k, v in encoding.items()}

        # processor 없이 raw 반환
        return {
            "image":  image,
            "masks":  torch.from_numpy(masks),
            "labels": torch.from_numpy(labels),
            "boxes":  torch.from_numpy(boxes),
        }


def collate_fn(batch):
    """Mask2Former용 collate — 길이가 다른 masks/labels를 리스트로 묶음"""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask   = torch.stack([b["pixel_mask"]   for b in batch])
    mask_labels  = [b["mask_labels"]  for b in batch]
    class_labels = [b["class_labels"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask":   pixel_mask,
        "mask_labels":  mask_labels,
        "class_labels": class_labels,
    }
