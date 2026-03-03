import cv2
import numpy as np
from utils import print_stage
from train import run_train
from finetune import run_finetune
from export import export_onnx
from ultralytics import YOLO
from depth.depth_estimator import DepthEstimator
from depth.visualizer import draw_depth_map, draw_results
from llm.groq_client import GroqClient
from config import FINETUNE_BEST_ONNX


def load_models():
    """모델 한 번만 로드"""
    seg_model   = YOLO(FINETUNE_BEST_ONNX, task="segment")
    depth_model = DepthEstimator(model_name="Intel/dpt-large")
    groq        = GroqClient()
    return seg_model, depth_model, groq


def run_depth_size(image_path: str, seg_model, depth_model, groq):
    # ── 이미지 로드 ────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    # ── YOLO Segmentation ──────────────────────
    seg_results = seg_model(image)[0]

    # ── 탐지 결과 없을 때 예외처리 ───────────────
    if len(seg_results.boxes) == 0:
        print("탐지된 음식이 없습니다.")
        return

    # ── DPT 깊이 추정 ──────────────────────────
    depth_map = depth_model.estimate(image)

    # ── Segmentation + 깊이 정보 수집 ──────────
    size_results      = []
    class_names       = []
    detection_results = []

    for i, box in enumerate(seg_results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id   = int(box.cls[0])
        cls_name = seg_model.names[cls_id]

        if seg_results.masks is not None:
            mask = seg_results.masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        else:
            mask = np.zeros(image.shape[:2])
            mask[y1:y2, x1:x2] = 1

        # ── Segmentation 정보 계산 ─────────────
        pixel_width  = x2 - x1
        pixel_height = y2 - y1
        mask_area    = int(np.sum(mask > 0))
        image_area   = image.shape[0] * image.shape[1]
        area_ratio   = round(mask_area / image_area * 100, 2)

        # ✅ 깊이값 추가 - 마스크 영역의 중앙값
        masked_depth = depth_map[mask > 0]
        avg_depth    = round(float(np.median(masked_depth)), 4) if len(masked_depth) > 0 else 0.0

        # 깊이값 해석 (0=가까움, 1=멀음)
        if avg_depth < 0.3:
            depth_comment = "가까움"
        elif avg_depth < 0.6:
            depth_comment = "중간"
        else:
            depth_comment = "멀음"

        detection_results.append({
            "class"        : cls_name,
            "pixel_width"  : pixel_width,
            "pixel_height" : pixel_height,
            "mask_area"    : mask_area,
            "area_ratio"   : area_ratio,
            "avg_depth"    : avg_depth,      # ✅ 추가
            "depth_comment": depth_comment,  # ✅ 추가
        })

        class_names.append(cls_name)
        size_results.append({"bbox": (x1, y1, x2, y2)})

        print(f"[{cls_name}] "
              f"픽셀 크기: {pixel_width}x{pixel_height} | "
              f"면적 비율: {area_ratio}% | "
              f"깊이: {avg_depth} ({depth_comment})")  # ✅ 깊이 출력 추가

    # ── Groq 분석 ──────────────────────────────
    print_stage("Groq 음식 분석 중...")
    llm_result = groq.analyze(detection_results)
    groq.print_result(llm_result)

    # ── 시각화 저장 ────────────────────────────
    output        = draw_results(image, size_results, class_names)
    depth_overlay = draw_depth_map(image, depth_map)

    cv2.imwrite("output_size.jpg", output)
    cv2.imwrite("output_depth.jpg", depth_overlay)
    print("저장 완료: output_size.jpg / output_depth.jpg")


def main():
    # 1단계: 학습 완료 → 주석 처리
    # print_stage("1단계: 사전 학습 시작")
    # run_train()

    # 2단계: 파인튜닝 완료 → 주석 처리
    # print_stage("2단계: 파인튜닝 시작")
    # run_finetune()

    # 3단계: 경량화 완료 → 주석 처리
    # print_stage("3단계: 경량화 시작")
    # export_onnx()

    # ✅ 모델 한 번만 로드
    seg_model, depth_model, groq = load_models()

    # 4단계: 추론 실행
    print_stage("4단계: 깊이 추정 + 크기 계산 + LLM 분석")
    run_depth_size(
        "./test.jpg",
        seg_model,
        depth_model,
        groq,
    )


if __name__ == "__main__":
    main()