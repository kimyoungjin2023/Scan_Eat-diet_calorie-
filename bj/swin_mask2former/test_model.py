import os
import sys
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Mask2Former 모듈 로드
sys.path.insert(0, os.path.join(os.getcwd(), "Mask2Former"))
try:
    from mask2former import add_maskformer2_config
except ImportError:
    print("❌ Mask2Former 폴더를 찾을 수 없습니다.")

# ---------------------------------------------------------
# 📸 테스트할 이미지 경로를 여기에 적어주세요! (검증 데이터셋 중 1장)
IMAGE_PATH = r"C:\scan_eat\data\valid\images\Img_001_0306_jpg.rf.86ef15c10cab31e15d578b50073c7a06.jpg" 
# ---------------------------------------------------------

def main():
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.set_new_allowed(True)
    
    # 1. 설정 파일 및 방금 학습이 끝난 최종 가중치 로드
    config_path = r"C:\scan_eat\Mask2Former\configs\coco\panoptic-segmentation\swin\maskformer2_swin_tiny_bs16_50ep.yaml"
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = r"C:\scan_eat\output\model_final.pth" # 우리가 만든 뇌!
    
    # 2. 학습 시 적용했던 강제 보정 세팅 (테스트할 때도 동일하게 필요)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 44
    cfg.MODEL.MASK_FORMER.NUM_CLASSES = 44
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.MASK_ON = True
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # 탐지 임계값 (50% 이상 확신하는 객체만 화면에 표시)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

    print("🤖 AI 뇌(model_final.pth) 로딩 중...")
    predictor = DefaultPredictor(cfg)
    
    # 이미지 읽기
    im = cv2.imread(IMAGE_PATH)
    if im is None:
        print(f"❌ 이미지를 찾을 수 없습니다. 경로를 다시 확인해주세요: {IMAGE_PATH}")
        return
        
    print("🔍 음식 윤곽선 분석 중...")
    outputs = predictor(im)
    
    # 결과 시각화 (색칠하기)
    # 등록된 데이터셋이름이 없으므로 빈 메타데이터 사용 (클래스가 숫자로 표시될 수 있음)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("__unused"), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # 결과 이미지 화면에 띄우기
    result_img = out.get_image()[:, :, ::-1]
    cv2.imshow("SCAN Eat AI Result", result_img)
    print("✅ 분석 완료! 이미지가 화면에 띄워졌습니다. (창을 끄려면 띄워진 이미지 클릭 후 아무 키나 누르세요)")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()