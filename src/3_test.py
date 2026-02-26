"""
모델 테스트 및 시각화
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import random
from tqdm import tqdm

from config import *
from utils import *

# 한글 폰트 설정
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial Unicode MS", "Malgun Gothic"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(model_path: Path = None):
    """모델 로드"""
    if model_path is None:
        # 기본 경로에서 모델 찾기
        possible_paths = [
            MODELS_DIR / "yolov11_food" / "weights" / "best.pt",
            MODELS_DIR / "yolov11_food_lite" / "weights" / "best.pt",
        ]

        for path in possible_paths:
            if path.exists():
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(
                "학습된 모델을 찾을 수 없습니다. 먼저 2_train.py를 실행하세요!"
            )

    print(f"모델 로드: {model_path}")
    model = YOLO(model_path)

    # 클래스 이름 로드
    data_config = load_yaml(DATA_DIR / "data.yaml")
    class_names = data_config["names"]

    return model, class_names


def validate_model():
    """검증 데이터로 성능 평가"""
    print_section("📊 모델 성능 평가")

    model, class_names = load_model()

    # 검증 실행
    metrics = model.val(data=str(DATA_DIR / "data.yaml"), split="val")

    print(f"\n전체 성능 지표:")
    print(f"  mAP@0.5:      {metrics.seg.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.seg.map:.4f}")
    print(f"  Precision:    {metrics.seg.mp:.4f}")
    print(f"  Recall:       {metrics.seg.mr:.4f}")

    # 클래스별 성능 (상위 10개)
    if hasattr(metrics.seg, "ap50") and len(metrics.seg.ap50) > 0:
        print(f"\n클래스별 성능 (상위 10개):")

        class_aps = list(zip(class_names, metrics.seg.ap50))
        class_aps.sort(key=lambda x: x[1], reverse=True)

        for i, (class_name, ap) in enumerate(class_aps[:10]):
            print(f"  {i+1:2d}. {class_name:20s}: {ap:.4f}")

    return metrics


def test_on_images():
    """테스트 이미지로 추론"""
    print_section("🧪 테스트 이미지 추론")

    model, class_names = load_model()

    test_images_dir = DATA_DIR / "test" / "images"

    if not test_images_dir.exists():
        print(f"⚠️ 테스트 이미지 디렉토리가 없습니다: {test_images_dir}")
        return

    test_images = list(test_images_dir.glob("*.jpg")) + list(
        test_images_dir.glob("*.png")
    )

    if not test_images:
        print("⚠️ 테스트 이미지가 없습니다!")
        return

    print(f"테스트 이미지: {len(test_images)}개")

    # 추론 실행
    results = model.predict(
        source=str(test_images_dir),
        conf=0.25,
        iou=0.45,
        save=True,
        project=str(RESULTS_DIR),
        name="test_predictions",
        exist_ok=True,
    )

    print(f"추론 완료! 결과 저장: {RESULTS_DIR}/test_predictions")
    return results


def visualize_predictions(num_samples: int = 6):
    """예측 결과 시각화"""
    print_section("🎨 예측 결과 시각화")

    model, class_names = load_model()

    test_images_dir = DATA_DIR / "test" / "images"

    if not test_images_dir.exists():
        print(f"⚠️ 테스트 이미지 디렉토리가 없습니다: {test_images_dir}")
        return

    test_images = list(test_images_dir.glob("*.jpg")) + list(
        test_images_dir.glob("*.png")
    )

    if not test_images:
        print("⚠️ 테스트 이미지가 없습니다!")
        return

    # 랜덤 샘플 선택
    random.seed(42)
    samples = random.sample(test_images, min(num_samples, len(test_images)))

    # 그리드 설정
    cols = 3
    rows = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    # 클래스별 색상 생성
    np.random.seed(42)
    colors = {}
    for i in range(len(class_names)):
        colors[i] = tuple(np.random.randint(100, 255, 3).tolist())

    for idx, img_path in enumerate(samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # 이미지 로드
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 예측
        results = model.predict(img_path, verbose=False)

        detected_foods = []

        if results and results[0].masks is not None:
            result = results[0]

            # 마스크 그리기
            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = class_names[class_id]

                detected_foods.append(f"{class_name}({confidence:.2f})")

                # 마스크 데이터
                mask_data = mask.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_data, (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # 색상 적용
                color_mask = np.zeros_like(img_rgb)
                color_mask[mask_binary == 1] = colors[class_id]

                # 반투명 오버레이
                img_rgb = cv2.addWeighted(img_rgb, 1, color_mask, 0.4, 0)

                # 윤곽선
                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(img_rgb, contours, -1, colors[class_id], 2)

                # 레이블
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name} {confidence:.2f}"

                # 텍스트 배경
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img_rgb,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w + 10, y1),
                    colors[class_id],
                    -1,
                )
                cv2.putText(
                    img_rgb,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        ax.imshow(img_rgb)
        ax.axis("off")

        # 제목 설정
        title = ", ".join(detected_foods) if detected_foods else "탐지 실패"
        ax.set_title(f"{img_path.name}\n{title}", fontsize=8)

    # 빈 subplot 제거
    for idx in range(len(samples), rows * cols):
        row = idx // cols
        col = idx % cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()

    # 저장
    save_path = RESULTS_DIR / "visualization_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"시각화 저장: {save_path}")


def run_full_test():
    """전체 테스트 파이프라인"""
    try:
        # 1. 성능 평가
        metrics = validate_model()

        # 2. 테스트 추론
        test_on_images()

        # 3. 시각화
        visualize_predictions()

        print_section("✅ 테스트 완료!")

        return metrics

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """테스트 메인 함수"""
    logger = setup_logging()

    try:
        metrics = run_full_test()
        return metrics

    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        raise


if __name__ == "__main__":
    main()
