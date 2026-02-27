# 🍽️ SCANEAT - AI 음식 칼로리 예측 시스템

<div align="center">

![SCANEAT Logo](docs/images/logo.png)

**음식 사진 한 장으로 칼로리를 자동으로 예측하는 AI 시스템**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Seg-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[데모](#-데모) • [설치](#-설치-방법) • [사용법](#-사용-방법) • [문서](#-문서) • [팀](#-팀)

</div>

---

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 기능](#-주요-기능)
- [성능](#-성능)
- [데모](#-데모)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [프로젝트 구조](#-프로젝트-구조)
- [학습 과정](#-학습-과정)
- [배포](#-배포)
- [로드맵](#️-로드맵)
- [팀](#-팀)
- [라이선스](#-라이선스)

---

## 🎯 프로젝트 소개

**SCANEAT**는 음식 사진을 분석하여 칼로리를 자동으로 예측하는 딥러닝 기반 시스템입니다.

### 🌟 핵심 기능
- 📸 **사진 촬영**: 음식 사진만 찍으면 끝
- 🔍 **AI 인식**: 44종 한식 자동 인식
- ✂️ **영역 분할**: Segmentation으로 정확한 면적 계산
- 📊 **칼로리 예측**: 음식 종류와 양 기반 칼로리 계산 (개발 중)
- 📱 **모바일 지원**: Android 앱 제공

### 🎓 배경
현대인의 건강 관리를 위해 음식 칼로리를 쉽고 정확하게 파악할 수 있는 시스템의 필요성 대두

---

## ✨ 주요 기능

### 1. 🔍 음식 Detection
- **44종 한식** 실시간 인식
- 여러 음식 동시 검출 가능
- 높은 정확도 (mAP@50: 0.665)

### 2. ✂️ Instance Segmentation
- 픽셀 단위 정확한 음식 영역 분할
- 겹친 음식도 분리 인식
- Mask mAP@50: 0.654

### 3. ⚡ 빠른 추론
- GPU: ~20ms/image
- CPU: ~500ms/image
- 모바일: ~200ms/image (TFLite)

### 4. 📱 다양한 플랫폼
- 웹 (Flask)
- Android 앱 (TFLite)
- REST API

---

## 📊 성능

### 모델 성능

| Metric | Score |
|--------|-------|
| **Box mAP@50** | **0.665** |
| **Mask mAP@50** | **0.654** |
| Box mAP@50-95 | 0.532 |
| Mask mAP@50-95 | 0.528 |
| Precision | 0.759 |
| Recall | 0.548 |

### 추론 속도

| Platform | Device | Speed |
|----------|--------|-------|
| Server | GPU (T4) | ~20ms |
| Server | CPU | ~500ms |
| Mobile | Android (GPU) | ~200ms |

### 모델 크기

| Format | Size | Use Case |
|--------|------|----------|
| PyTorch (.pt) | 22.8 MB | 학습/개발 |
| ONNX (optimized) | 35 MB | 웹 배포 |
| TFLite (INT8) | 6 MB | 모바일 |

---

## 🎬 데모

### 웹 데모
![Web Demo](results/sample_predictions/web_demo.gif)

### 모바일 데모
![Mobile Demo](results/sample_predictions/mobile_demo.gif)

### 샘플 결과
<div align="center">
<img src="results/sample_predictions/sample1.jpg" width="45%">
<img src="results/sample_predictions/sample2.jpg" width="45%">
</div>

---

## 🛠️ 기술 스택

### ML/DL
- **Model**: YOLOv8s-seg (Ultralytics)
- **Framework**: PyTorch 2.0+
- **Training**: Google Colab (T4 GPU)

### Data Processing
- **Augmentation**: Albumentations
- **Annotation**: CVAT
- **Dataset**: 621장 → 1273장 (증강)

### Deployment
- **Web**: Flask + ONNX Runtime
- **Mobile**: Android + TFLite
- **API**: REST API

### Tools
- Python 3.10+
- OpenCV
- NumPy, Pandas
- Matplotlib

---

## 📥 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/YOUR_TEAM/scaneat.git
cd scaneat
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 모델 다운로드

모델 파일은 용량이 커서 Git에 포함되지 않습니다. 아래 링크에서 다운로드하세요:

- 🔗 [best.pt (22.8MB) - PyTorch 원본](https://drive.google.com/...)
- 🔗 [best_simplified.onnx (35MB) - 웹용](https://drive.google.com/...)
- 🔗 [best_int8.tflite (6MB) - 모바일용](https://drive.google.com/...)

다운로드 후 `models/` 폴더에 저장:

```bash
scaneat/
└── models/
    ├── best.pt
    ├── best_simplified.onnx
    └── best_int8.tflite
```

---

## 🚀 사용 방법

### 기본 추론

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('models/best.pt')

# 추론
results = model.predict('food_image.jpg', conf=0.25)

# 결과 확인
for r in results:
    for box in r.boxes:
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)
        print(f"{class_name}: {confidence:.2%}")
```

### 학습

```bash
python src/training/train_augmented.py \
    --data data/data.yaml \
    --epochs 200 \
    --batch 16 \
    --imgsz 640
```

### Flask 웹 서버 실행

```bash
cd deployment/flask
python app.py

# 브라우저에서 접속
# http://localhost:5000
```

### Android 앱 빌드

자세한 내용: [Android 설정 가이드](deployment/android/README.md)

---

## 📂 프로젝트 구조

```
scaneat/
├── README.md                    # 프로젝트 소개
├── requirements.txt             # Python 의존성
├── .gitignore                   # Git 제외 파일
│
├── data/
│   ├── data.yaml               # 데이터셋 설정
│   └── sample/                 # 샘플 이미지
│
├── src/
│   ├── training/               # 학습 코드
│   │   ├── train_base.py
│   │   ├── train_augmented.py
│   │   └── data_augmentation.py
│   │
│   ├── models/                 # 모델 관련
│   │   └── compression.py
│   │
│   └── utils/                  # 유틸리티
│       ├── preprocessing.py
│       └── evaluation.py
│
├── deployment/
│   ├── flask/                  # 웹 배포
│   │   ├── app.py
│   │   └── README.md
│   │
│   └── android/                # 안드로이드 앱
│       ├── MainActivity.kt
│       └── README.md
│
├── models/                     # 학습된 모델 (다운로드 필요)
│   ├── README.md
│   └── .gitkeep
│
├── results/                    # 학습 결과
│   ├── training_logs/
│   └── sample_predictions/
│
└── docs/                       # 문서
    ├── SETUP.md
    ├── TRAINING.md
    └── DEPLOYMENT.md
```

---

## 📈 학습 과정

### 1. 데이터 준비
- **데이터 수집**: 621장 한식 이미지
- **라벨링**: CVAT 사용
- **클래스**: 44종 (Bokkeum_Dakgalbi, Grilled_GrilledEel, ...)

### 2. 데이터 증강
- **Albumentations**: Geometric, Color, Blur
- **YOLOv8 내장**: Mosaic, Mixup, Copy-paste
- **결과**: 621장 → 1,273장 (2배)

### 3. 모델 학습
- **Architecture**: YOLOv8s-seg
- **Epochs**: 200 (Early Stop at 78)
- **CV**: 10-Fold StratifiedKFold
- **Device**: Google Colab T4 GPU

### 4. 모델 경량화
- **ONNX**: 포맷 변환 (웹 배포용)
- **TFLite INT8**: 양자화 (모바일용)
- **압축률**: 22.8MB → 6MB (약 4배)

### 학습 결과
![Training Curves](results/training_logs/results.png)

자세한 내용: [TRAINING.md](docs/TRAINING.md)

---

## 🌐 배포

### Flask 웹 서버

```bash
cd deployment/flask
pip install -r requirements.txt
python app.py
```

- **URL**: http://localhost:5000
- **기능**: 이미지 업로드, 실시간 분석, REST API

### Android 앱

```bash
cd deployment/android
# Android Studio에서 빌드
```

- **최소 SDK**: 24 (Android 7.0)
- **크기**: ~15MB (앱 + 모델)
- **기능**: 카메라 촬영, 갤러리 선택, 실시간 분석

자세한 설명: [배포 가이드](docs/DEPLOYMENT.md)

---

## 🗺️ 로드맵

### ✅ 완료
- [x] 기본 모델 학습 (mAP 0.665)
- [x] 데이터 증강 (2배)
- [x] 모델 경량화 (ONNX, TFLite)
- [x] Flask 웹 서버
- [x] Android 앱 프로토타입

### 🚧 진행 중
- [ ] 칼로리 계산 API 연동
- [ ] 음식 양(무게) 추정 알고리즘
- [ ] UI/UX 개선

### 📋 예정
- [ ] 사용자 계정 시스템
- [ ] 식단 기록 및 통계
- [ ] 영양소 분석
- [ ] 음식 추천 시스템
- [ ] iOS 앱
- [ ] 클라우드 배포 (AWS/GCP)

---

## 👥 팀

### 팀원

| 이름 | 역할 | GitHub |
|------|------|--------|
| **OOO** | 팀장, ML 모델 개발 | [@username](https://github.com/username) |
| **OOO** | 데이터 수집, 증강 | [@username](https://github.com/username) |
| **OOO** | 웹 배포, API | [@username](https://github.com/username) |
| **OOO** | Android 앱 개발 | [@username](https://github.com/username) |

### 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🙏 감사의 말

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 베이스 모델
- [Roboflow](https://roboflow.com/) - 데이터셋 호스팅
- [Google Colab](https://colab.research.google.com/) - 무료 GPU 제공
- [Albumentations](https://github.com/albumentations-team/albumentations) - 데이터 증강

---

## 📧 문의

- **프로젝트 관련**: your-email@example.com
- **버그 리포트**: [GitHub Issues](https://github.com/YOUR_TEAM/scaneat/issues)
- **기능 제안**: [GitHub Discussions](https://github.com/YOUR_TEAM/scaneat/discussions)

---

<div align="center">

**⭐ 이 프로젝트가 마음에 드셨다면 Star를 눌러주세요! ⭐**

Made with ❤️ by SCANEAT Team

</div>