"""
학습된 DINOv2 + Mask2Former 추론 스크립트
사용법:
    python inference.py --model_dir outputs/best_model --image test.jpg
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)


COLORS = [
    [255, 0,   0  ], [0,   255, 0  ], [0,   0,   255],
    [255, 255, 0  ], [0,   255, 255], [255, 0,   255],
    [255, 128, 0  ], [128, 0,   255], [0,   128, 255],
    [255, 0,   128], [0,   255, 128], [128, 255, 0  ],
]


def run_inference(model_dir: str, image_path: str, threshold: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"모델 로드: {model_dir}")
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    id2label = model.config.id2label

    # 이미지 로드 & 전처리
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 후처리 — instance segmentation
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        target_sizes=[image.size[::-1]],  # (H, W)
    )[0]

    # 시각화
    img_np = np.array(image)
    overlay = img_np.copy()

    legend_patches = []
    for i, segment in enumerate(results["segments_info"]):
        label_id = segment["label_id"]
        label_name = id2label.get(label_id, str(label_id))
        score = segment["score"]
        mask_id = segment["id"]

        mask = (results["segmentation"] == mask_id).cpu().numpy()
        color = COLORS[i % len(COLORS)]

        overlay[mask] = (
            np.array(color) * 0.5 + img_np[mask] * 0.5
        ).astype(np.uint8)

        legend_patches.append(
            mpatches.Patch(
                color=[c / 255 for c in color],
                label=f"{label_name} ({score:.2f})"
            )
        )

    # 결과 출력
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img_np)
    axes[0].set_title("원본 이미지")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Instance Segmentation (threshold={threshold})")
    axes[1].legend(handles=legend_patches, loc="upper right", fontsize=9)
    axes[1].axis("off")

    plt.tight_layout()

    out_path = image_path.replace(".", "_result.")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"결과 저장: {out_path}")
    plt.show()

    print(f"\n감지된 인스턴스: {len(results['segments_info'])}개")
    for seg in results["segments_info"]:
        label = id2label.get(seg["label_id"], str(seg["label_id"]))
        print(f"  - {label}  score={seg['score']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="best_model 경로")
    parser.add_argument("--image",     type=str, required=True, help="추론할 이미지 경로")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_inference(args.model_dir, args.image, args.threshold)
