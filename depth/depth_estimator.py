import torch
import numpy as np
import cv2
from transformers import DPTForDepthEstimation, DPTImageProcessor

class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-large"):
        """
        DPT 모델 로드
        - dpt-large  : 정확도 높음, 느림
        - dpt-hybrid : 속도/정확도 균형
        """
        print(f"DPT 모델 로딩 중: {model_name}")
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"DPT 모델 로드 완료 ({self.device})")

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 → 깊이 맵 반환
        image: BGR numpy array (cv2로 읽은 이미지)
        return: 정규화된 깊이 맵 (0~1)
        """
        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 전처리
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth  # (1, H, W)

        # 원본 이미지 크기로 리사이즈
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # 정규화 (0~1)
        depth_np = depth.cpu().numpy()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

        return depth_np