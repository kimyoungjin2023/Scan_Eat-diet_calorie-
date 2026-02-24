import os
import cv2
import numpy as np

# 설정
IMAGES_DIR = "dataset/images"
LABELS_SEG_DIR = "dataset/labels_seg"
OUTPUT_VIS_DIR = "dataset/visualization"

os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

# 색상 팔레트 (클래스별)
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
]

# 몇 장만 확인
image_files = [
    f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".jpeg", ".png"))
][:10]

for img_file in image_files:
    img_path = os.path.join(IMAGES_DIR, img_file)
    label_path = os.path.join(LABELS_SEG_DIR, os.path.splitext(img_file)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    overlay = img.copy()

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # 정규화 좌표 → 픽셀 좌표
            points = []
            for j in range(0, len(coords), 2):
                px = int(coords[j] * w)
                py = int(coords[j + 1] * h)
                points.append([px, py])

            pts = np.array(points, dtype=np.int32)
            color = COLORS[cls_id % len(COLORS)]

            # 반투명 마스크 그리기
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], True, color, 2)

    # 원본과 마스크 합성 (투명도 40%)
    result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    output_path = os.path.join(OUTPUT_VIS_DIR, img_file)
    cv2.imwrite(output_path, result)
    print(f"시각화 저장: {output_path}")

print(f"\n시각화 결과 확인: {OUTPUT_VIS_DIR}")
