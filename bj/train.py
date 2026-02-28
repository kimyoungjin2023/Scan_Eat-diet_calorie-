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

# 데이터셋 경로
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
        mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
    
    # # 백본 동결을 위해 build_model 메서드를 오버라이드합니다.
    # @classmethod
    # def build_model(cls, cfg):
    #     """
    #     모델을 빌드한 직후에 백본의 파라미터를 고정(Freeze)합니다.
    #     """
    #     model = super().build_model(cfg)
        
    #     # Swin-Transformer 백본 고정
    #     if hasattr(model, "backbone"):
    #         for param in model.backbone.parameters():
    #             param.requires_grad = False
    #         print("❄️ [실험] 백본(Backbone) 동결 완료! Head만 학습을 진행합니다.")
    #     else:
    #         print("⚠️ 백본을 찾을 수 없어 동결에 실패했습니다.")
            
    #     return model

def main():
    register_datasets()
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.set_new_allowed(True) 
    
    # 1. 베테랑 모델의 설정 파일로 변경 (Instance Segmentation 전용)
    # 기존 panoptic-segmentation 대신 instance-segmentation용 yaml을 쓰는 것이 더 정확합니다.
    config_path = r"C:\scan_eat\Mask2Former\configs\coco\instance-segmentation\swin\maskformer2_swin_tiny_bs16_50ep.yaml"
    cfg.merge_from_file(config_path)
    
    # # 2. 베테랑의 '지식(가중치)' 직접 주입 
    # cfg.MODEL.WEIGHTS = r"C:\scan_eat\weights\model_final_86143f.pkl"

    # 2-1. 한번 학습 완료한 모델을 이어서 학습
    cfg.MODEL.WEIGHTS = r"C:\scan_eat\output_frozen_backbone\model_final.pth" 


    # 3. 클래스 수 설정 (기존 코드 유지)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 44
    cfg.MODEL.MASK_FORMER.NUM_CLASSES = 44
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # ====================================================================
    # 🎛️ [파인튜닝 다이얼 1] 데이터 자동 증강 (Data Augmentation)
    # ====================================================================
    # (의미) 사진을 그대로 보여주지 않고, 크기를 바꾸거나 뒤집어서 데이터가 많은 것처럼 속입니다.
    cfg.INPUT.MIN_SIZE_TRAIN = (384, 512, 640) # 사진의 짧은 변 길이를 384, 512, 640 중 하나로 랜덤하게 변경
    cfg.INPUT.MAX_SIZE_TRAIN = 640             # 아무리 키워도 640은 넘지 않게 제한 (메모리 터짐 방지)
    cfg.INPUT.RANDOM_FLIP = "horizontal"       # 50% 확률로 사진을 거울처럼 좌우 반전해서 학습!
    
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512
    
    # # ====================================================================
    # # 🎛️ [파인튜닝 다이얼 2] 이어달리기 바통 터치 (Pre-trained Weights)
    # # ====================================================================
    # # (의미) 쌩 Swin 백본이 아니라, 어제 1차로 학습을 마친 '우리의 뇌'를 가져옵니다.
    # cfg.MODEL.WEIGHTS = r"C:\scan_eat\output\model_final.pth"
    
    # 모델 내부 크기 세팅 (고정)
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]

    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.NUM_WORKERS = 0 # 윈도우 환경 오류 방지용

    cfg.DATASETS.TRAIN = ("scaneat_train",)
    cfg.DATASETS.TEST = ("scaneat_val",)
    
    # ====================================================================
    # 🎛️ [파인튜닝 다이얼 3] 학습의 강도와 디테일 조절 (Solver)
    # ====================================================================
    # 1. 배치 사이즈 (Batch Size)
    cfg.SOLVER.IMS_PER_BATCH = 2  # 한 번에 몇 장의 사진을 볼 것인가? (VRAM이 넉넉하면 4로 올려도 좋음)
    
    # 2. 학습률 (Learning Rate, 보폭)
    cfg.SOLVER.BASE_LR = 0.00001  
    
    # 3. 최대 학습 횟수 (Max Iterations)
    cfg.SOLVER.MAX_ITER = 10000   # 앞으로 추가로 1만 번 더 사진을 보며 훈련함
    
    # 4. 학습률 스케줄러 (LR Scheduler, 감속 타이밍) ⭐ 새로 추가됨!
    cfg.SOLVER.STEPS = (7000, 9000) # 7천 번, 9천 번 돌았을 때 보폭을 확 줄여서 마무리 조각을 함
    cfg.SOLVER.GAMMA = 0.1          # 위 타이밍이 되면 보폭을 1/10로 확 줄여버림 (0.00005 -> 0.000005)
    
    # 5. 과적합 방지 브레이크 (Weight Decay) ⭐ 새로 추가됨!
    cfg.SOLVER.WEIGHT_DECAY = 0.05  # 모델이 정답만 달달 외우지 못하게 살짝 헷갈리게 하는 규제 (응용력 향상)
    # ====================================================================

    # 저장 폴더를 phase2(2단계)로 분리하여 어제 결과와 안 섞이게 보호합니다.
    cfg.OUTPUT_DIR = "../output_unfreeze_backbone" 
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\n🚀 2차 파인튜닝 시작! (데이터 증강 및 스케줄러 적용 완료)")
    
    trainer = MaskFormerTrainer(cfg) 
    
    # resume=False: 어제 모델(WEIGHTS)은 가져오되, Iteration 카운트는 다시 0부터 깨끗하게 시작함
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()