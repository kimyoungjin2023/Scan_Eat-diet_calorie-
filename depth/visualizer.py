import cv2
import numpy as np

def draw_depth_map(image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """깊이 맵 컬러 시각화"""
    depth_colored = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_MAGMA)
    return cv2.addWeighted(image, 0.6, depth_colored, 0.4, 0)

def draw_results(
    image: np.ndarray,
    results: list,
    class_names: list,
) -> np.ndarray:
    """YOLO 결과 + 클래스명 시각화"""
    output = image.copy()

    for i, result in enumerate(results):
        x1, y1, x2, y2 = result["bbox"]
        cls_name = class_names[i] if i < len(class_names) else f"obj{i}"

        # 바운딩박스
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ✅ 클래스명만 출력 (real_width_cm 제거)
        cv2.putText(
            output, cls_name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 2,
        )

    return output
