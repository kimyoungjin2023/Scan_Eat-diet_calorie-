"""
깊이 추정 및 부피 계산 유틸리티 (수정됨)
"""

import torch
import cv2
import numpy as np
from typing import Tuple, Dict, List


class DepthEstimator:
    """단안 깊이 추정 클래스"""

    def __init__(self, model_type: str = "MiDaS_small", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"🔍 깊이 추정 모델 로딩: {model_type} (device: {self.device})")

        # MiDaS 모델 로드
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()

        # 전처리 변환 함수 로드
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        print("✓ 깊이 추정 모델 로드 완료")

    def estimate_depth(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RGB 이미지에서 깊이 맵 추정"""
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_raw = prediction.cpu().numpy()

        # 정규화 (시각화용)
        depth_min = depth_raw.min()
        depth_max = depth_raw.max()
        depth_normalized = (depth_raw - depth_min) / (depth_max - depth_min + 1e-8)

        return depth_raw, depth_normalized


class VolumeCalculator:
    """
    부피 점수 계산 클래스 (밀도 가정 없음)
    LLM이 실제 중량을 판단하도록 상대적 지표만 제공
    """

    def calculate_volume_metrics(
        self, mask_binary: np.ndarray, depth_map: np.ndarray, class_name: str = ""
    ) -> Dict[str, float]:
        """
        세그멘테이션 마스크와 깊이 맵으로 상대적 부피 메트릭 계산

        Returns:
            dict: 상대적 부피 관련 메트릭 (물리적 단위 없음)
        """
        # 마스크 영역의 깊이 값만 추출
        masked_depth = depth_map * mask_binary

        # 유효 픽셀 수 (면적 정보)
        valid_pixels = mask_binary > 0
        pixel_count = valid_pixels.sum()

        if pixel_count == 0:
            return {
                "pixel_count": 0,
                "avg_depth": 0.0,
                "volume_score": 0.0,
                "relative_size": "none",
            }

        # 평균 깊이 계산
        avg_depth = masked_depth[valid_pixels].mean()

        # 부피 점수 = 면적 × 깊이 (상대적 지표)
        volume_score = pixel_count * avg_depth

        # 상대적 크기 분류 (LLM 참고용)
        if volume_score > 50000:
            relative_size = "large"
        elif volume_score > 20000:
            relative_size = "medium"
        elif volume_score > 5000:
            relative_size = "small"
        else:
            relative_size = "very_small"

        return {
            "pixel_count": int(pixel_count),
            "avg_depth": float(avg_depth),
            "volume_score": float(volume_score),
            "relative_size": relative_size,
        }


def analyze_food_with_depth(
    image_path: str,
    yolo_results,
    class_names: List[str],
    depth_estimator: DepthEstimator,
    volume_calculator: VolumeCalculator,
) -> Tuple[np.ndarray, List[Dict]]:
    """YOLO 결과와 깊이 추정을 통합하여 음식 분석"""

    # 이미지 로드
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 깊이 추정
    depth_raw, depth_normalized = depth_estimator.estimate_depth(img_rgb)
    depth_resized = cv2.resize(depth_raw, (w, h))
    depth_norm_resized = cv2.resize(depth_normalized, (w, h))

    food_analysis = []

    if yolo_results and yolo_results[0].masks is not None:
        result = yolo_results[0]

        for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = class_names[class_id]

            # 마스크 처리
            mask_data = mask.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_data, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # 부피 메트릭 계산 (밀도 없이)
            volume_metrics = volume_calculator.calculate_volume_metrics(
                mask_binary, depth_resized, class_name
            )

            # 바운딩 박스
            bbox = box.xyxy[0].cpu().numpy().tolist()

            food_info = {
                "class_name": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "bbox": bbox,
                **volume_metrics,
            }

            food_analysis.append(food_info)

    return depth_norm_resized, food_analysis
