# 🍽️ Scan_Eat (diet_calorie)

> 📸 Food Image → 🍱 Food Detection  
> AI 기반 음식 객체 탐지 프로젝트  
> **Team Project | 5 Members**

---

## 📌 Project Overview

**Scan_Eat (diet_calorie)** 는 음식 이미지를 입력받아  
객체 탐지(Object Detection)를 통해 음식의 위치와 종류를 식별하는 AI 프로젝트입니다.

현재는 **Detection 단계**를 구현하고 있으며,  
향후 음식 분할(Segmentation) 및 칼로리 추정 기능까지 확장하는 것을 목표로 합니다.

---

## 🎯 Current Objective (Phase 1)

- 이미지 내 음식 객체 탐지
- Bounding Box 생성
- 음식 클래스 분류
- Detection 성능 지표(mAP, Precision, Recall) 분석

---

## 🧠 Tech Stack

- Python 3.9+
- PyTorch
- YOLO / Faster R-CNN
- OpenCV
- Albumentations
- CUDA (GPU 환경 권장)

---

## 👥 Team Members (5)

| Name | Role | Responsibility |
|------|------|---------------|
| Member 1 | Team Lead | Project Planning, Model Architecture |
| Member 2 | AI Engineer | Model Training & Tuning |
| Member 3 | Data Engineer | Dataset 구축 및 전처리 |
| Member 4 | Backend | Inference Pipeline 구현 |
| Member 5 | Research | 논문 리서치 및 성능 분석 |

> ※ 실제 이름과 역할에 맞게 수정해주세요.

---

## 📂 Project Structure

```
Scan_Eat/
│
├── data/
│   ├── images/
│   ├── labels/
│
├── models/
│   ├── detection_model.pt
│
├── train.py
├── detect.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/Scan_Eat.git
cd Scan_Eat
pip install -r requirements.txt
```

---

## 🏋️ Model Training

```bash
python train.py --data ./data --epochs 50 --batch 16 --img-size 640
```

### 주요 파라미터

- `--epochs` : 학습 반복 횟수
- `--batch` : 배치 사이즈
- `--img-size` : 입력 이미지 크기

---

## 🔎 Inference

```bash
python detect.py --weights models/detection_model.pt --source test.jpg
```

결과 이미지는 `/runs/detect/` 폴더에 저장됩니다.

---

## 📊 Evaluation Metrics

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

---

## 🗺️ Roadmap

### Phase 1 (Current)
- [x] Food Detection 모델 구현
- [ ] 성능 최적화 및 하이퍼파라미터 튜닝

### Phase 2
- [ ] Food Segmentation
- [ ] Portion Size Estimation

### Phase 3
- [ ] Calorie Estimation 모델 통합
- [ ] Web/App 배포

---

## 💡 Expected Applications

- 다이어트 보조 앱
- 스마트 식단 관리 시스템
- 헬스케어 AI 서비스
- B2B 푸드 데이터 분석 솔루션

---

## 👨‍💻 Team

Scan_Eat Team  
AI-based Food Detection & Diet Assistant