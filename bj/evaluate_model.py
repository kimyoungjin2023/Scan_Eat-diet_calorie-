import os
import sys
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

# Mask2Former 모듈 로드
sys.path.insert(0, r"C:\scan_eat\Mask2Former")
try:
    from mask2former import add_maskformer2_config
except ImportError:
    print("❌ Mask2Former 폴더를 찾을 수 없습니다.")

# 검증 데이터셋 경로 (38장)
VAL_JSON = r"C:\scan_eat\data\valid\_annotations.coco_final.json"
VAL_IMG = r"C:\scan_eat\data\valid\images"

def main():
    # 데이터셋 등록
    if "scaneat_val" not in DatasetCatalog.list():
        register_coco_instances("scaneat_val", {}, VAL_JSON, VAL_IMG)

    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.set_new_allowed(True)
    
    # 설정 파일 및 완성된 모델(뇌) 로드
    config_path = r"C:\scan_eat\Mask2Former\configs\coco\instance-segmentation\swin\maskformer2_swin_tiny_bs16_50ep.yaml"
    cfg.merge_from_file(config_path)
    
    # 2. 오늘 학습 완료된 'Phase 2' 모델 로드
    cfg.MODEL.WEIGHTS = r"C:\scan_eat\output_unfreeze_backbone\model_final.pth"
    
    # 학습 때와 동일한 강제 보정 세팅
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

    print("🤖 평가를 위해 모델을 불러오는 중...")
    predictor = DefaultPredictor(cfg)
    
    # COCO 평가기(Evaluator) 생성
    evaluator = COCOEvaluator("scaneat_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "scaneat_val")
    
    print("\n📊 검증 데이터(valid)로 시험을 시작합니다! (mAP 추출 중...)\n")
    # 평가 실행 및 결과 출력
    val_results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    print("\n" + "="*50)
    print("🏆 최종 성적표 ")
    print("="*50)
    # 딕셔너리 형태로 예쁘게 출력
    for task, metrics in val_results.items():
        print(f"[{task}]")
        for metric, score in metrics.items():
            print(f" - {metric}: {score:.3f}")

if __name__ == "__main__":
    main()