import numpy as np

class SizeCalculator:
    def __init__(self, focal_length: float = 500.0, real_depth_scale: float = 10.0):
        """
        focal_length     : 카메라 초점 거리 (픽셀 단위, 카메라 캘리브레이션 값)
        real_depth_scale : 깊이 맵 스케일 → 실제 거리(m) 변환 계수
        
        ※ 정확한 실제 크기를 원하면 카메라 캘리브레이션 필요
          캘리브레이션 없으면 상대적 크기 비교만 가능
        """
        self.focal_length = focal_length
        self.real_depth_scale = real_depth_scale

    def get_object_depth(self, depth_map: np.ndarray, mask: np.ndarray) -> float:
        """마스크 영역의 평균 깊이값 반환"""
        masked_depth = depth_map[mask > 0]
        if len(masked_depth) == 0:
            return 0.0
        return float(np.median(masked_depth))  # 평균보다 중앙값이 더 안정적

    def pixel_to_real_size(
        self,
        pixel_width: int,
        pixel_height: int,
        depth_value: float,
    ) -> dict:
        """
        픽셀 크기 + 깊이값 → 실제 크기 추정
        Z = depth_value * real_depth_scale (실제 거리, m)
        실제크기 = (픽셀크기 * Z) / focal_length
        """
        Z = depth_value * self.real_depth_scale

        real_width  = (pixel_width  * Z) / self.focal_length
        real_height = (pixel_height * Z) / self.focal_length

        return {
            "estimated_depth_m" : round(Z, 3),
            "real_width_m"      : round(real_width, 3),
            "real_height_m"     : round(real_height, 3),
            "real_width_cm"     : round(real_width * 100, 1),
            "real_height_cm"    : round(real_height * 100, 1),
        }

    def calculate(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        bbox: tuple,         # (x1, y1, x2, y2)
    ) -> dict:
        """
        YOLO 마스크 + 바운딩박스 → 실제 크기 계산
        """
        x1, y1, x2, y2 = bbox
        pixel_w = x2 - x1
        pixel_h = y2 - y1

        depth_value = self.get_object_depth(depth_map, mask)
        size_info   = self.pixel_to_real_size(pixel_w, pixel_h, depth_value)

        return {
            "bbox"          : bbox,
            "pixel_width"   : pixel_w,
            "pixel_height"  : pixel_h,
            "depth_value"   : round(depth_value, 4),
            **size_info,
        }