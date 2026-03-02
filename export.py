from ultralytics import YOLO

FINETUNE_BEST_PT = "./result/run_second_fintuning/weight/best.pt"

def export_onnx():
    """ONNX 변환 - 범용적, 대부분의 환경에서 사용 가능"""
    model = YOLO(FINETUNE_BEST_PT)
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,      # 모델 구조 단순화
        opset=17,           # ONNX opset 버전
        half=True,          # FP16 경량화 (절반 용량)
    )

def export_tensorrt():
    """TensorRT 변환 - GPU 추론 최적화 (NVIDIA GPU 필요)"""
    model = YOLO(FINETUNE_BEST_PT)
    model.export(
        format="engine",
        imgsz=640,
        half=True,          # FP16
        device=0,
    )

def export_int8():
    """INT8 양자화 - 가장 작은 용량, 약간의 정확도 손실"""
    model = YOLO(FINETUNE_BEST_PT)
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        int8=True,          # INT8 양자화
        data="./dataset/data.yaml",  # 양자화 캘리브레이션용
    )

def export_all():
    """전부 다 변환"""
    print("── ONNX FP16 변환 ──")
    export_onnx()

    print("── TensorRT 변환 ──")
    export_tensorrt()

    print("── INT8 양자화 변환 ──")
    export_int8()