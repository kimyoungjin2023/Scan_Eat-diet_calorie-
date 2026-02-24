import os
import cv2
import numpy as np
from ultralytics import SAM

# ============================================
# 설정 - 본인 경로에 맞게 수정하세요
# ============================================
IMAGES_DIR = "dataset/images"  # 이미지 폴더
LABELS_DIR = "dataset/labels"  # 기존 디텍션 라벨 폴더 (YOLO txt)
OUTPUT_DIR = "dataset/labels_seg"  # 세그먼테이션 라벨 저장 폴더

# ============================================
# SAM 모델 로드
# ============================================
model = SAM("sam_b.pt")  # 처음 실행 시 자동 다운로드

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# 이미지별 처리
# ============================================
image_files = [
    f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".jpeg", ".png"))
]

for idx, img_file in enumerate(image_files):
    img_path = os.path.join(IMAGES_DIR, img_file)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(LABELS_DIR, label_file)

    # 라벨 파일 없으면 건너뛰기
    if not os.path.exists(label_path):
        print(f"[{idx+1}/{len(image_files)}] 라벨 없음, 건너뜀: {img_file}")
        continue

    # 이미지 크기 읽기
    img = cv2.imread(img_path)
    if img is None:
        print(f"[{idx+1}/{len(image_files)}] 이미지 읽기 실패: {img_file}")
        continue
    h, w = img.shape[:2]

    # YOLO 라벨 읽기 → 바운딩 박스 변환
    bboxes = []
    class_ids = []

    with open(label_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            # YOLO 정규화 좌표 → 픽셀 좌표 [x1, y1, x2, y2]
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            bboxes.append([x1, y1, x2, y2])
            class_ids.append(cls_id)

    if not bboxes:
        print(f"[{idx+1}/{len(image_files)}] 바운딩 박스 없음: {img_file}")
        continue

    # ============================================
    # SAM으로 세그먼테이션!
    # ============================================
    results = model(img_path, bboxes=bboxes)

    # ============================================
    # 세그먼테이션 결과 → YOLO 세그먼테이션 형식으로 저장
    # ============================================
    output_path = os.path.join(OUTPUT_DIR, label_file)

    with open(output_path, "w") as f:
        if results[0].masks is not None:
            for i, mask in enumerate(results[0].masks):
                cls_id = class_ids[i] if i < len(class_ids) else 0

                # 마스크 → 폴리곤 좌표 (YOLO seg 형식)
                # masks.xyn = 정규화된 폴리곤 좌표 (0~1)
                segments = mask.xyn[0]  # numpy array [[x,y], [x,y], ...]

                if len(segments) < 3:
                    continue

                # YOLO 세그먼테이션 형식: class_id x1 y1 x2 y2 x3 y3 ...
                line = str(cls_id)
                for point in segments:
                    line += f" {point[0]:.6f} {point[1]:.6f}"
                f.write(line + "\n")

    print(
        f"[{idx+1}/{len(image_files)}] 완료: {img_file} → {len(bboxes)}개 객체 세그먼테이션"
    )

print("\n모든 이미지 처리 완료!")
print(f"세그먼테이션 라벨 저장 위치: {OUTPUT_DIR}")
