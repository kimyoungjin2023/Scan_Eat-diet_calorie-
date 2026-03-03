"""
모델 테스트 및 시각화
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from depth_utils import DepthEstimator, VolumeCalculator, analyze_food_with_depth

from ultralytics import YOLO
from pathlib import Path

from nutrition_analyzer import NutritionAnalyzer

from config import *
from utils import *

# 한글 폰트 설정
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(model_path: Path = None):
    """모델 로드"""
    if model_path is None:
        # 기본 경로에서 모델 찾기
        possible_paths = [
            MODELS_DIR / "yolov11_food_finetuned_v3" / "weights" / "best.pt"
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
    print(f"  metrics:      {metrics.seg}")
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
        iou=0.70,
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


def visualize_predictions_with_depth(num_samples: int = 4):
    """깊이 추정을 포함한 예측 결과 시각화"""
    print_section("🎨 세그멘테이션 + 깊이 추정 시각화")

    # 모델 로드
    model, class_names = load_model()

    # 깊이 추정 및 부피 계산기 초기화
    depth_estimator = DepthEstimator(model_type="MiDaS_small")  # 빠른 버전
    volume_calculator = VolumeCalculator()

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

    # 시각화 설정: 각 이미지당 3개 패널 (원본+세그멘테이션, 깊이맵, 통합결과)
    rows = len(samples)
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    # 클래스별 색상
    np.random.seed(42)
    colors = {
        i: tuple(np.random.randint(100, 255, 3).tolist())
        for i in range(len(class_names))
    }

    print("\n📊 음식별 부피 분석 결과:")
    print("=" * 80)

    for idx, img_path in enumerate(samples):
        print(f"\n이미지: {img_path.name}")
        print("-" * 60)

        # YOLO 예측
        yolo_results = model.predict(img_path, verbose=False)

        # 깊이 추정 및 통합 분석
        depth_map, food_analysis = analyze_food_with_depth(
            img_path, yolo_results, class_names, depth_estimator, volume_calculator
        )

        # 이미지 로드
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # 1. 원본 + 세그멘테이션
        seg_img = img_rgb.copy()
        detected_info = []

        for food in food_analysis:
            class_id = food["class_id"]
            class_name = food["class_name"]
            confidence = food["confidence"]
            volume_score = food["volume_score"]
            mass_relative = food["estimated_mass_relative"]

            # 마스크 재생성 (시각화용)
            if yolo_results and yolo_results[0].masks is not None:
                mask_data = (
                    yolo_results[0]
                    .masks[food_analysis.index(food)]
                    .data[0]
                    .cpu()
                    .numpy()
                )
                mask_resized = cv2.resize(mask_data, (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # 색상 오버레이
                color_mask = np.zeros_like(seg_img)
                color_mask[mask_binary == 1] = colors[class_id]
                seg_img = cv2.addWeighted(seg_img, 0.7, color_mask, 0.3, 0)

                # 윤곽선
                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(seg_img, contours, -1, colors[class_id], 2)

                # 레이블
                x1, y1, x2, y2 = map(int, food["bbox"])
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(
                    seg_img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            detected_info.append(f"{class_name}({mass_relative:.0f})")

            # 콘솔 출력
            print(
                f"  {class_name:20s} | 신뢰도: {confidence:.2f} | "
                f"부피점수: {volume_score:8.0f} | 상대질량: {mass_relative:.0f}"
            )

        axes[idx, 0].imshow(seg_img)
        axes[idx, 0].set_title(f"Segmentation\n{', '.join(detected_info)}", fontsize=10)
        axes[idx, 0].axis("off")

        # 2. 깊이 맵
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        axes[idx, 1].imshow(depth_colored_rgb)
        axes[idx, 1].set_title("Depth Map", fontsize=10)
        axes[idx, 1].axis("off")

        # 3. 통합 결과 (깊이 + 세그멘테이션)
        combined = img_rgb.copy()

        for food in food_analysis:
            if yolo_results and yolo_results[0].masks is not None:
                mask_data = (
                    yolo_results[0]
                    .masks[food_analysis.index(food)]
                    .data[0]
                    .cpu()
                    .numpy()
                )
                mask_resized = cv2.resize(mask_data, (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # 마스크 영역에 깊이 정보 시각화
                masked_depth = depth_map * mask_binary
                depth_vis = cv2.applyColorMap(
                    (masked_depth * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                depth_vis_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

                # 마스크 영역만 깊이 색상 적용
                mask_3d = np.stack([mask_binary] * 3, axis=-1)
                combined = np.where(
                    mask_3d > 0,
                    cv2.addWeighted(combined, 0.6, depth_vis_rgb, 0.4, 0),
                    combined,
                )

                # 경계선
                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(combined, contours, -1, (255, 255, 255), 2)

        axes[idx, 2].imshow(combined)
        axes[idx, 2].set_title("Depth + Segmentation", fontsize=10)
        axes[idx, 2].axis("off")

    plt.tight_layout()

    # 저장
    save_path = RESULTS_DIR / "depth_segmentation_analysis.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n💾 시각화 결과 저장: {save_path}")

    return food_analysis


def analyze_with_llm_nutrition(num_samples: int = 2):
    """YOLO + Depth + Gemini 통합 영양소 분석"""
    print_section("🤖 AI 통합 영양소 분석 (YOLO + Depth + Gemini)")

    # Google API 키 확인
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ GOOGLE_API_KEY 환경변수를 설정하세요.")
        print("🔗 API 키 발급: https://aistudio.google.com/app/apikey")
        api_key = input("Google AI API Key를 입력하세요: ").strip()
        if not api_key:
            print("❌ API 키가 필요합니다.")
            return

    # 모델 및 도구 초기화
    model, class_names = load_model()
    depth_estimator = DepthEstimator(model_type="MiDaS_small")
    volume_calculator = VolumeCalculator()
    nutrition_analyzer = NutritionAnalyzer(api_key=api_key)

    # 테스트 이미지 선택
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

    all_results = []

    for idx, img_path in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"📸 이미지 분석 {idx}/{len(samples)}: {img_path.name}")
        print(f"{'='*80}")

        try:
            # 1단계: YOLO 세그멘테이션
            print("1️⃣ YOLO 음식 감지 및 세그멘테이션...")
            yolo_results = model.predict(img_path, conf=0.25, verbose=False)

            # 2단계: 깊이 추정 및 부피 계산
            print("2️⃣ 깊이 추정 및 부피 점수 계산...")
            depth_map, food_analysis = analyze_food_with_depth(
                img_path, yolo_results, class_names, depth_estimator, volume_calculator
            )

            if not food_analysis:
                print("⚠️ 감지된 음식이 없습니다.")
                continue

            # 중간 결과 출력
            print(f"✅ {len(food_analysis)}개 음식 감지:")
            for food in food_analysis:
                print(
                    f"   • {food['class_name']}: 부피점수 {food['volume_score']:.0f} ({food['relative_size']})"
                )

            # 3단계: LLM 영양소 분석
            print("3️⃣ LLM 영양소 및 칼로리 분석...")
            nutrition_result = nutrition_analyzer.analyze_nutrition(
                str(img_path), food_analysis
            )

            # 결과 출력
            nutrition_analyzer.print_nutrition_report(nutrition_result)

            # 결과 저장
            result_data = {
                "image_path": str(img_path),
                "image_name": img_path.name,
                "detection_results": food_analysis,
                "nutrition_analysis": nutrition_result,
            }
            all_results.append(result_data)

            # JSON 파일로 저장
            json_path = RESULTS_DIR / f"nutrition_analysis_{img_path.stem}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            print(f"💾 결과 저장: {json_path}")

        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            continue

    # 전체 결과 요약
    if all_results:
        print(f"\n{'='*80}")
        print("📊 전체 분석 완료 요약")
        print(f"{'='*80}")
        print(f"총 분석 이미지: {len(all_results)}개")

        total_calories = 0
        for result in all_results:
            cal = result["nutrition_analysis"]["total"]["calories_kcal"]
            food_count = len(result["nutrition_analysis"]["foods"])
            total_calories += cal
            print(f"  📱 {result['image_name']}: {food_count}개 음식, {cal} kcal")

        avg_calories = total_calories / len(all_results) if all_results else 0
        print(f"\n🏆 평균 칼로리: {avg_calories:.0f} kcal")
        print(f"{'='*80}")

    return all_results


# run_full_test 함수 수정
def run_full_test():
    """전체 테스트 파이프라인"""
    try:
        # LLM 통합 영양소 분석 실행
        results = analyze_with_llm_nutrition(num_samples=2)

        print_section("✅ 모든 분석 완료!")
        return results

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
