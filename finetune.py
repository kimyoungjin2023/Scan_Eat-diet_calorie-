from ultralytics import YOLO
from config import FINETUNE_CONFIG, PROJECT_DIR
from utils import get_best_pt_path

def run_finetune(best_pt_path: str = None):
    # 경로 미입력 시 자동 탐색
    if best_pt_path is None:
        best_pt_path = get_best_pt_path(PROJECT_DIR, "pretrain")
    
    model = YOLO(best_pt_path)
    model.train(**FINETUNE_CONFIG)