from ultralytics import YOLO
from config import FINETUNE_BEST_ONNX, DATA_YAML, PROJECT_DIR

# 파인튜닝된 best.pt 경로 (export 전 pt 파일)
FINETUNE_BEST_PT = "./result/run_second_fintuning/weight/best.pt"

def export_onnx():
    """ONNX FP16 변환 - 범용 배포용"""
    model = YOLO(FINETUNE_BEST_PT)
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        half=True,
    )
    print("ONNX FP16 변환 완료")

def export_tensorrt():
    """TensorRT 변환 - NVIDIA GPU 서버 배포용"""
    model = YOLO(FINETUNE_BEST_PT)
    model.export(
        format="engine",
        imgsz=640,
        half=True,
        device=0,
    )
    print("TensorRT 변환 완료")

def export_int8():
    """INT8 양자화 - 최대 경량화"""
    model = YOLO(FINETUNE_BEST_PT)
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        int8=True,
        data=DATA_YAML,
    )
    print("INT8 양자화 변환 완료")