import os
import sys
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_train_loader

# Mask2Former 모듈 로드
sys.path.insert(0, r"C:\scan_eat\Mask2Former")
try:
    from mask2former import add_maskformer2_config
    from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import MaskFormerInstanceDatasetMapper
    print("✅ Mask2Former 모듈 로드 성공!")
except ImportError:
    print("❌ Mask2Former 폴더를 찾을 수 없습니다.")

# 데이터셋 경로 (BJ님이 새로 생성하신 _annotations.coco_final.json 사용)
TRAIN_JSON = r"C:\scan_eat\data\train\_annotations.coco_final.json"
TRAIN_IMG = r"C:\scan_eat\data\train\images"
VAL_JSON = r"C:\scan_eat\data\valid\_annotations.coco_final.json"
VAL_IMG = r"C:\scan_eat\data\valid\images"

def register_datasets():
    if "scaneat_train" not in DatasetCatalog.list():
        register_coco_instances("scaneat_train", {}, TRAIN_JSON, TRAIN_IMG)
    if "scaneat_val" not in DatasetCatalog.list():
        register_coco_instances("scaneat_val", {}, VAL_JSON, VAL_IMG)

class MaskFormerTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Mask2Former 전용 데이터 매퍼 사용 (Instance Segmentation)
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

def main():
    register_datasets()
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.set_new_allowed(True) 
    
    # 1. 설정 파일 로드 (Instance Segmentation 전용)
    config_path = r"C:\scan_eat\Mask2Former\configs\coco\instance-segmentation\swin\maskformer2_swin_tiny_bs16_50ep.yaml"
    cfg.merge_from_file(config_path)
    
    # 2. 베테랑 가중치 주입 (처음부터 깨끗하게 학습)
    cfg.MODEL.WEIGHTS = r"C:\scan_eat\weights\model_final_86143f.pkl"

    # ====================================================================
    # ⭐ [핵심 수정 1] 클래스 수 설정 (배경 0 포함하여 총 45개)
    # 새로운 JSON의 ID가 1~44이므로, 모델은 45개의 카테고리 공간이 필요합니다.
    # ====================================================================
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 45
    cfg.MODEL.MASK_FORMER.NUM_CLASSES = 45
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # 모델 내부 크기 세팅 (Config 일관성 유지)
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]

    # 데이터 증강 설정
    cfg.INPUT.MIN_SIZE_TRAIN = (384, 512, 640)
    cfg.INPUT.MAX_SIZE_TRAIN = 640 
    cfg.INPUT.RANDOM_FLIP = "horizontal" 
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512

    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.NUM_WORKERS = 0 

    cfg.DATASETS.TRAIN = ("scaneat_train",)
    cfg.DATASETS.TEST = ("scaneat_val",)
    
    # ====================================================================
    # ⭐ [핵심 수정 2] 학습 파라미터 최적화
    # ====================================================================
    cfg.SOLVER.IMS_PER_BATCH = 2  
    
    # 학습률 상향: 0.00001 -> 0.0001 (새로운 특징을 배우기에 적합한 보폭)
    cfg.SOLVER.BASE_LR = 0.0001  
    
    # 학습 횟수: 12,000회로 늘려 충분히 수렴할 시간 확보
    cfg.SOLVER.MAX_ITER = 12000   
    cfg.SOLVER.STEPS = (8000, 10000) 
    cfg.SOLVER.GAMMA = 0.1 
    cfg.SOLVER.WEIGHT_DECAY = 0.05 

    # 출력 폴더 분리 (새로운 실험 기록 보호)
    cfg.OUTPUT_DIR = "../output_m2f_reboot_final" 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\n🚀 [리부트] 45클래스 & 최적화된 학습률로 재학습을 시작합니다!")
    
    trainer = MaskFormerTrainer(cfg) 
    
    # resume=False: 가중치는 가져오되, Iteration 0부터 새롭게 시작
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()