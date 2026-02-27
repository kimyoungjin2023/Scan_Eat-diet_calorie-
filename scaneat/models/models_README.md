# 📦 학습된 모델 다운로드

모델 파일은 용량이 커서 Git에 포함되지 않습니다.

아래 링크에서 다운로드 후 이 폴더에 저장하세요.

---

## 📥 다운로드 링크

### 1. PyTorch 원본 모델 (.pt)
- **파일명**: `best.pt`
- **크기**: 22.8 MB
- **용도**: 학습, 개발, 고정확도 추론
- **성능**: mAP@50 0.665
- **다운로드**: [Google Drive 링크](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)

### 2. ONNX 최적화 모델
- **파일명**: `best_simplified.onnx`
- **크기**: 35-40 MB
- **용도**: 웹 배포 (Flask)
- **성능**: mAP@50 0.658
- **다운로드**: [Google Drive 링크](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)

### 3. TFLite 모바일 모델
- **파일명**: `best_int8.tflite`
- **크기**: 6 MB
- **용도**: Android 앱
- **성능**: mAP@50 0.63-0.65 (예상)
- **다운로드**: [Google Drive 링크](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)

---

## 📂 다운로드 후 폴더 구조

```
models/
├── README.md           (이 파일)
├── best.pt             (다운로드 필요)
├── best_simplified.onnx (다운로드 필요)
└── best_int8.tflite    (다운로드 필요)
```

---

## 🚀 사용 방법

### PyTorch 모델
```python
from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model.predict('image.jpg')
```

### ONNX 모델
```python
from ultralytics import YOLO

model = YOLO('models/best_simplified.onnx', task='segment')
results = model.predict('image.jpg')
```

### TFLite 모델
```kotlin
// Android
val interpreter = Interpreter(loadModelFile("best_int8.tflite"))
```

---

## 📊 모델 성능 비교

| 모델 | 크기 | Box mAP@50 | Mask mAP@50 | 속도(GPU) |
|------|------|-----------|------------|-----------|
| PyTorch | 22.8MB | 0.665 | 0.654 | ~20ms |
| ONNX | 35MB | 0.658 | 0.631 | ~15ms |
| TFLite | 6MB | 0.63-0.65 | 0.61-0.63 | ~200ms |

---

## ⚠️ 주의사항

1. **모델 파일은 Git에 푸시하지 마세요!**
   - `.gitignore`에 이미 포함됨
   
2. **Google Drive 공유 설정**
   - "링크가 있는 모든 사용자" 권한 설정
   
3. **버전 관리**
   - 모델 업데이트 시 파일명에 버전 추가
   - 예: `best_v1.0.pt`, `best_v1.1.pt`

---

## 📝 모델 정보

- **학습 데이터**: 621장 → 1,273장 (증강)
- **클래스**: 44개 한식
- **Architecture**: YOLOv8s-seg
- **학습 Epoch**: 200 (Early Stop at 78)
- **Device**: Google Colab T4 GPU