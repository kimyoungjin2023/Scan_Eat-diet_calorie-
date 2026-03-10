import torch
import torchvision
import os
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# 1. 모델 구조 정의 (학습 시 설정과 동일해야 함)
def get_model(num_classes):
    # 가중치 없이 기본 구조 로드
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    
    # Box 헤드 수정
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Mask 헤드 수정
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model

def main():
    # 설정값
    num_classes = 45 
    checkpoint_path = "best_maskrcnn_bj_top1.pth" # BJ님의 최고 성능 모델 파일명
    onnx_path = "best_maskrcnn_bj_final.onnx"
    quantized_path = "best_maskrcnn_bj_int8.onnx"
    
    # 2. 모델 로드 (CPU에서 변환하는 것이 가장 안전함)
    print("📦 모델 가중치 로드 중...")
    device = torch.device('cpu')
    model = get_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 3. 더미 데이터 생성 (모델의 입력 채널과 크기에 맞춤)
    # Mask R-CNN은 리스트 형태의 입력을 받으므로 아래와 같이 구성합니다.
    dummy_input = [torch.randn(3, 800, 800)] 

    # 4. ONNX 내보내기 (Export)
    # 에러 방지를 위해 dynamic_axes를 'input' 텐서의 차원에 정확히 매칭함
    print("🔄 PyTorch -> ONNX 변환 시작 (잠시만 기다려주세요)...")
    try:
        torch.onnx.export(
            model, 
            (dummy_input,), 
            onnx_path, 
            export_params=True, 
            opset_version=11, 
            do_constant_folding=True, 
            input_names=['input'], 
            output_names=['boxes', 'labels', 'scores', 'masks'],
            # Mask R-CNN 입력 리스트 특성에 맞춰 배치 차원만 가변 설정
            dynamic_axes={'input': {0: 'batch'}} 
        )
        print(f"✅ ONNX 변환 완료: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        return

    # 5. ONNX Runtime 양자화 (INT8)
    print("⚡ INT8 양자화(Quantization) 시작...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            onnx_path, 
            quantized_path, 
            weight_type=QuantType.QUInt8
        )
        
        # 파일 용량 확인
        orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
        
        print("\n" + "="*30)
        print(f"🎉 모든 경량화 작업 완료!")
        print(f"📊 원본 ONNX: {orig_size:.2f} MB")
        print(f"📊 양자화 ONNX: {quant_size:.2f} MB")
        print(f"📉 압축률: {orig_size/quant_size:.1f}배 가벼워짐")
        print("="*30)
        
    except ImportError:
        print("ℹ️ onnxruntime이 설치되지 않았습니다. 'pip install onnxruntime' 후 다시 실행하세요.")

if __name__ == "__main__":
    main()