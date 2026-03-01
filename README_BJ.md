# 🍕 SCAN Eat - Advanced Instance Segmentation Pipeline (BJ's Workspace)

> **"박스(BBox)가 아닌 픽셀(Pixel)을 딴다!"** > SCAN Eat 프로젝트의 정밀한 음식 영역 추출을 위한 파이프라인 구축 브랜치입니다. 소규모 데이터셋의 한계를 극복하기 위해 Mask2Former에서 Mask R-CNN으로 아키텍처를 전환하며 성능을 극대화한 기록을 담고 있습니다.

---

## 🏗️ 1. 모델 아키텍처 (Model Architecture)

### 🚀 [Current SOTA] Mask R-CNN (최종 채택)

- **Task**: Instance Segmentation (음식 개별 객체 분할)
- **Backbone**: ResNet50 + FPN (Feature Pyramid Network)
- **Framework**: PyTorch / Torchvision
- **Strategy**: 2-Stage Detector 구조를 통한 고정밀 ROI 추출 및 픽셀 단위 마스크 생성

### 🕰️ [Legacy] Mask2Former (초기 탐색)

- **Task**: Query-based Pixel Segmentation
- **Head**: Mask2Former (COCO 데이터셋 사전 학습 모델 적용)
- **Backbone**: Swin-Transformer (Tiny)
- **Framework**: Detectron2 (Windows용 커스텀 빌드)

---

## 📊 2. 데이터셋 (Dataset)

- **도메인**: 한식 위주의 음식 이미지 (총 44개 클래스)
- **Train**: 620장 / **Valid**: 38장
- **포맷**: COCO format JSON (`_annotations.coco_final.json`)
- **최적화 (Detectron2)**: 윈도우 환경 메모리 누수 방지를 위한 `NUM_WORKERS = 0` 설정

---

## 💡 3. 핵심 구현 및 트러블슈팅

### [Mask R-CNN 최적화 전략 (현재)]

- **커스텀 데이터 증강 (Custom Augmentation)**: `torchvision`의 한계를 극복, 이미지 반전 시 **BBox 좌표와 Segmentation Mask가 완벽히 동기화되어 반전**되는 수동 Flip 로직 및 Color Jitter(조명 대응) 직접 구현.
- **단계적 파인튜닝 (Progressive Fine-tuning)**: 초반 20에폭은 백본을 동결(Freeze)하여 베테랑 지식을 보존하고 과적합을 방지. 이후 동결을 풀고(Unfreeze) 낮은 학습률(0.0005)로 전체를 미세 조정하여 한식 특유의 질감 완벽 포착.

### [Mask2Former 트러블슈팅 (초기)]

- **베테랑 모델(Pre-trained) 지식 이식**: 인코더(Swin)만 학습된 상태에서 디코더(Mask2Former Head)까지 COCO 데이터로 학습된 전체 가중치를 로드하여 '백지상태'의 헤드 성능 문제를 근본적으로 해결함.
- **네트워크 보안 및 403 Forbidden 해결**: 서버 측 직접 다운로드 차단 문제를 해결하기 위해 GitHub Model Zoo에서 가중치 파일(`86143f` 버전)을 수동 확보 후 로컬 경로 연결.
- **GitHub 인증 복구**: 보안 정책 변경으로 인한 Push 실패를 PAT(Personal Access Token) 발급 및 자격 증명 업데이트를 통해 해결.

---

## 📅 작업 일지 (Dev Log)

### 📍 2026-02-23 ~ 2026-02-25 (환경 구축 및 1차 PoC)

- Detectron2 및 Mask2Former 로컬 설치.
- 1차 학습 진행 (10,000 iter) 결과 Segm mAP50 **38.6%** 달성.

### 📍 2026-02-26 (전략 최적화)

- 하이퍼파라미터 미세 조정: `Base LR` 0.00005 하향, `Weight Decay` 0.05 적용.
- 데이터 증강(RandomFlip, Multi-scale) 적용 완료.

### 📍 2026-02-27 (2차 파인튜닝 및 베테랑 모델 이식)

- **모델 보완**: COCO 데이터셋으로 '칼질 실력'을 쌓은 베테랑 헤드 가중치 이식 성공.
- **성능 분석 (Valid 38장)**:
  - **종합 성적**: Segm AP50 **31.865%** 기록 (수치보다 정교한 경계선 추출에 집중).
  - **주요 성과**: 간장게장(**85.155%**), 진미채볶음(**80.000%**), 닭갈비(**70.957%**) 등 특정 음식군에서 높은 정확도 확보.
  - **특이사항**: BBox mAP 0은 모델 구조적 특징임을 재확인, 검증 셋 부재 클래스는 `nan` 처리됨.

### 📍 2026-02-28 (파인튜닝 전략 심화 및 모델 적응성 분석)

- **실험 1: 백본 동결(Backbone Freeze) 파인튜닝**
  - **결과**: Segm AP50 **24.0%** 기록. 일반 객체(COCO)와 한국 음식 간의 시각적 격차(Domain Gap)가 커서, 백본을 고정할 경우 한식 특유의 질감과 형태를 포착하는 데 한계가 있음을 확인.
- **실험 2: 점진적 전체 해제 학습 (Two-stage Unfreezing)**
  - **결과**: Segm AP50 **29.1%** 달성 (동결 모델 대비 **+5.1%p** 향상). 간장게장(66.3%), 진미채(80.0%) 등 적응력 즉각 상승.
- **📊 Mask2Former 최종 결론**: 픽셀 정밀도는 높으나, 현재의 '스몰 데이터(620장)' 환경에서는 쿼리 기반 모델의 한계가 존재함. 모델의 전 가중치를 도메인 특화에 동원하는 것이 유리하나, 근본적인 구조 변경의 필요성 대두.

### 📍 2026-03-01 (아키텍처 전환 및 60% 고지 점령) 🚀 ✅

- **Mask R-CNN 도입**: 소규모 데이터셋에 더 안정적이고 정밀한 ROI를 제공하는 2-Stage 모델(Mask R-CNN)로 전면 개편.
- **실험 1: 베이스라인 확보**
  - 전체 학습 진행 시 Segm mAP50 **50.9%** 달성 (Mask2Former 최고 기록 압도, 그러나 과적합 증상 발견).
- **실험 2: 전략적 파인튜닝 (Freeze + Augment + Scheduler)**
  - **전략**: `데이터 증강(Flip/Color)` + `초반 20ep 백본 동결(지식 보존)` + `후반 15ep 동결 해제 및 LR 감쇄(미세 조정)`.
  - **최종 결과**: **Segm mAP50 60.0% 달성!**
  - **인사이트**:
    - 특히 까다로운 기준인 **IoU 0.75 점수가 52.8%**로 폭등하며 압도적인 경계선 추출(칼질) 실력 증명.
    - 이전에는 잡지 못했던 중간 크기(Medium) 객체 인식률이 **23.3%**로 안정화됨.
    - **최종 결론**: 데이터가 적은 환경에서는 모델 아키텍처의 선택과 **'어떻게 얼리고 변형하느냐'**의 파인튜닝 기술이 성능의 핵심임을 입증.
