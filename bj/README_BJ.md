# 🍕 SCAN Eat - Mask2Former Segmentation Pipeline (BJ's Workspace)

> **"박스(BBox)가 아닌 픽셀(Pixel)을 딴다!"** > SCAN Eat 프로젝트의 정밀한 음식 영역 추출을 위한 Mask2Former 파이프라인 구축 브랜치입니다.

---

## 🏗️ 1. 모델 아키텍처 (Model Architecture)

- **Task**: Panoptic / Instance Segmentation
- **Head**: [Mask2Former](https://github.com/facebookresearch/Mask2Former) - 최신 쿼리 기반 픽셀 분할 알고리즘
- **Backbone**: Swin-Transformer (Tiny) - 시각 지능 엔진 (`swin_tiny_patch4_window7_224.pkl`)
- **Framework**: Detectron2

## 📊 2. 데이터셋 (Dataset)

- **도메인**: 한식 위주의 음식 이미지 (총 44개 클래스)
- **Train**: 620장 / **Valid**: 38장
- **포맷**: COCO format JSON (`_annotations.coco_final.json`)
- **최적화**: 윈도우 환경 메모리 누수 방지를 위한 `NUM_WORKERS = 0` 설정

## 💡 3. 핵심 구현 및 트러블슈팅

- **`MaskFormerTrainer` 커스텀 오버라이딩**:
  기본 Detectron2 트레이너가 폴리곤 마스크를 인식하지 못하는 `gt_masks` 에러 해결을 위해 `MaskFormerInstanceDatasetMapper`를 직접 연결한 전용 트레이너 클래스 구축.
- **Query-based Prediction 이해**:
  학습 결과 BBox mAP가 0으로 나오는 이유는 모델이 박스를 거치지 않고 직접 마스크를 생성하는 구조이기 때문임을 확인. 따라서 **Segm mAP**를 핵심 성능 지표로 설정함.

---

## 📅 작업 일지 (Dev Log)

### 📍 2026-02-23 ~ 2026-02-24 (환경 구축)

- Detectron2 및 Mask2Former 로컬 설치 (Windows용 빌드 whl 적용).
- 라벨링 데이터를 COCO JSON 포맷으로 전처리 완료.

### 📍 2026-02-25 (1차 PoC 학습)

- Swin-Tiny 백본 기반 1차 학습 진행 (10,000 iter).
- **성능 지표**: Segm mAP50 **38.6%** 달성.

### 📍 2026-02-26 (2차 정밀 파인튜닝 및 전략 최적화)

- **전이 학습(Transfer Learning)**: 1차 학습 가중치(`model_final.pth`)를 로드하여 연속 학습 수행.
- **하이퍼파라미터 미세 조정(Fine-tuning)**:
  - `Base LR`: 0.0001 → 0.00005 하향 조정 (정밀 수렴 유도).
  - `LR Scheduler`: Multi-step(7k, 9k iter) 감쇄 전략 적용.
  - `Weight Decay`: 0.05 및 `DropPath(Stochastic Depth)`를 통한 과적합 방지.
- **데이터 증강(Augmentation)**: RandomFlip 및 Multi-scale Training(384~640) 적용.
- **학습 단위 환산**: 10,000 iteration ≒ 약 32.2 Epoch (배치 사이즈 2 기준) 확인을 통한 팀 내 학습량 동기화.

### 📍 2026-02-27 (예정: 검증 및 시각화)
