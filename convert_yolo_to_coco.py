"""
YOLO Segmentation → COCO JSON 변환 스크립트
YOLO seg 포맷: <class_id> <x1> <y1> <x2> <y2> ... (normalized polygon)
"""

import os
import json
import glob
from pathlib import Path
from PIL import Image
import numpy as np


def yolo_seg_to_coco(
    images_dir: str,
    labels_dir: str,
    output_json: str,
    class_names: list,
    split: str = "train"
):
    """
    YOLO segmentation 포맷을 COCO JSON으로 변환

    Args:
        images_dir: 이미지 폴더 경로
        labels_dir: YOLO .txt 라벨 폴더 경로
        output_json: 출력 COCO JSON 경로
        class_names: 클래스 이름 리스트 (e.g. ['cat', 'dog'])
        split: 'train' or 'val'
    """
    coco = {
        "info": {"description": f"Converted from YOLO seg - {split}"},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # 카테고리 등록 (COCO는 1-indexed)
    for i, name in enumerate(class_names):
        coco["categories"].append({
            "id": i + 1,
            "name": name,
            "supercategory": "object"
        })

    image_id = 0
    ann_id = 0

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files = sorted(image_files)

    print(f"[{split}] 총 {len(image_files)}개 이미지 변환 중...")

    for img_path in image_files:
        img_path = Path(img_path)
        label_path = Path(labels_dir) / (img_path.stem + ".txt")

        # 이미지 크기 읽기
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"  이미지 열기 실패: {img_path} - {e}")
            continue

        image_id += 1
        coco["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })

        # 라벨 파일 없으면 스킵
        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            lines = f.read().strip().splitlines()

        for line in lines:
            if not line.strip():
                continue

            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            coords = parts[1:]  # normalized x1 y1 x2 y2 ...

            if len(coords) < 6:  # 최소 3점 필요
                continue

            # normalize → pixel 좌표 변환
            pixel_coords = []
            for i in range(0, len(coords), 2):
                px = coords[i] * width
                py = coords[i + 1] * height
                pixel_coords.extend([px, py])

            # bbox 계산
            xs = pixel_coords[0::2]
            ys = pixel_coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            area = bbox_w * bbox_h

            ann_id += 1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id + 1,  # COCO는 1-indexed
                "segmentation": [pixel_coords],
                "bbox": [x_min, y_min, bbox_w, bbox_h],
                "area": float(area),
                "iscrowd": 0
            })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"  저장 완료: {output_json}")
    print(f"  이미지: {len(coco['images'])}개, 어노테이션: {len(coco['annotations'])}개")
    return coco


if __name__ == "__main__":
    import yaml

    # ── 설정 ──────────────────────────────────────────────
    DATASET_YAML = "dataset.yaml"   # YOLO dataset.yaml 경로
    OUTPUT_DIR   = "coco_annotations"
    # ─────────────────────────────────────────────────────

    with open(DATASET_YAML) as f:
        cfg = yaml.safe_load(f)

    class_names = cfg["names"]
    root = cfg.get("path", ".")

    for split in ["train", "val"]:
        if split not in cfg:
            continue
        images_dir = os.path.join(root, cfg[split])
        labels_dir = images_dir.replace("/images", "/labels")
        output_json = os.path.join(OUTPUT_DIR, f"{split}.json")

        yolo_seg_to_coco(images_dir, labels_dir, output_json, class_names, split)
