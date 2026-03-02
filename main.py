import cv2
import numpy as np
from utils import print_stage
from train import run_train
from finetune import run_finetune
from export import export_onnx
from ultralytics import YOLO
from depth.depth_estimator import DepthEstimator
from depth.size_calculator import SizeCalculator
from depth.visualizer import draw_depth_map, draw_results


# ── 함수 정의는 항상 위에 ──────────────────────────
def run_depth_size(image_path: str):
    seg_model   = YOLO("/content/runs/segment/finetune/weights/best.pt")
    depth_model = DepthEstimator(model_name="Intel/dpt-large")
    size_calc   = SizeCalculator(focal_length=500.0, real_depth_scale=10.0)

    image       = cv2.imread(image_path)
    seg_results = seg_model(image)[0]
    depth_map   = depth_model.estimate(image)

    size_results = []
    class_names  = []

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

        result = size_calc.calculate(depth_map, mask, (x1, y1, x2, y2))
        size_results.append(result)
        class_names.append(cls_name)

        print(f"[{cls_name}] "
              f"W: {result['real_width_cm']}cm | "
              f"H: {result['real_height_cm']}cm | "
              f"Depth: {result['estimated_depth_m']}m")

    output        = draw_results(image, size_results, class_names)
    depth_overlay = draw_depth_map(image, depth_map)

    cv2.imwrite("output_size.jpg", output)
    cv2.imwrite("output_depth.jpg", depth_overlay)
    print("저장 완료: output_size.jpg / output_depth.jpg")


# ── main()도 위에 ──────────────────────────────────
def main():
    print_stage("1단계: 사전 학습 시작")
    run_train()

    print_stage("2단계: 파인튜닝 시작")
    run_finetune()

    print_stage("3단계: 경량화 시작")
    export_onnx()

    print_stage("4단계: 깊이 추정 + 크기 계산")
    run_depth_size("/content/test_image.jpg")


# ── if __name__ 은 딱 한 번만 ─────────────────────
if __name__ == "__main__":
    main()