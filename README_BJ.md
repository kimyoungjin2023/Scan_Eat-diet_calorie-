# 🍕 SCAN Eat - Advanced Instance Segmentation Pipeline (BJ's Workspace)

> **"박스(BBox)가 아닌 픽셀(Pixel)을 딴다!"** > SCAN Eat 프로젝트의 정밀한 음식 영역 추출을 위한 파이프라인 구축 브랜치입니다. 소규모 데이터셋의 한계를 극복하기 위해 Mask2Former에서 Mask R-CNN으로 아키텍처를 전환하며 성능을 극대화하고, 최종적으로 모바일/웹 배포를 위한 모델 경량화까지 완료한 기록을 담고 있습니다.

---

## 🏗️ 1. 모델 아키텍처 (Model Architecture)

### 🚀 [Current SOTA & Deployed] Mask R-CNN (최종 채택 및 경량화)

- **Task**: Instance Segmentation (음식 개별 객체 분할)
- **Backbone**: ResNet50 + FPN (Feature Pyramid Network)
- **Framework**: PyTorch ➡️ **ONNX Runtime (INT8 Quantized)**
- **Strategy**: 2-Stage Detector 구조를 통한 고정밀 ROI 추출 및 픽셀 단위 마스크 생성

### 🕰️ [Legacy] Mask2Former (초기 탐색)

- **Task**: Query-based Pixel Segmentation
- **Head**: Mask2Former (COCO 데이터셋 사전 학습 모델 적용)
- **Backbone**: Swin-Transformer (Tiny)
- **Framework**: Detectron2 (Windows용 커스텀 빌드)

---

## 📊 2. 데이터셋 (Dataset)

- **도메인**: 한식 위주의 음식 이미지 (총 44개 클래스)
- **Train**: 620장 / **Valid**: 38장 (오프라인 증강 없이 온라인 증강만으로 성능 극대화)
- **포맷**: COCO format JSON (`_annotations.coco_final.json`)
- **최적화**: 윈도우 환경 메모리 누수 방지를 위한 `NUM_WORKERS = 0` 설정

---

## 💡 3. 핵심 구현 및 기술적 성과

### [Mask R-CNN 최적화 및 경량화 전략 (최종)]

- **3단계 점진적 파인튜닝 (Progressive Unfreezing)**:
  - [Phase 1] 백본 완전 동결 (베테랑 지식 보존)
  - [Phase 2] 상위 레이어(Layer 3, 4) 부분 해제 (한식 도메인 적응)
  - [Phase 3] 전 구간 해제 및 극소 학습률(0.0001) 적용 (픽셀 정밀도 극대화)
- **극한의 온라인 데이터 증강 (Online Augmentation)**: BBox와 Mask 좌표가 동기화되는 수동 Flip 로직 및 음식 조명 변화에 대응하는 미세 Color Jitter(밝기, 대비, 채도) 적용.
- **배포용 모델 경량화 (INT8 Quantization)**: PyTorch 모델(`.pth`)을 범용 환경을 위한 `.onnx`로 변환 후, 동적 양자화(Dynamic Quantization)를 적용하여 정확도 손실 없이 **모델 용량을 3.9배 압축**.

### [Mask2Former 트러블슈팅 (초기)]

- **베테랑 모델 지식 이식**: 인코더(Swin)만 학습된 상태에서 디코더(Mask2Former Head)까지 COCO 가중치를 로드하여 '백지상태'의 성능 문제 해결.
- **네트워크 보안 및 인증 해결**: GitHub Model Zoo 다운로드 차단(403) 우회 및 PAT 발급을 통한 Push 권한 복구.

---

## 📅 작업 일지 (Dev Log)

### 📍 2026-02-23 ~ 2026-02-25 (환경 구축 및 1차 PoC)

- Detectron2 및 Mask2Former 로컬 설치.
- 1차 학습 진행 (10,000 iter) 결과 Segm mAP50 **38.6%** 달성.

### 📍 2026-02-26 ~ 2026-02-27 (전략 최적화 및 한계 확인)

- 하이퍼파라미터 및 증강(RandomFlip, Multi-scale) 조정.
- **성능 분석**: Segm AP50 **31.8%** 기록. 간장게장(85.1%), 진미채볶음(80.0%) 등 특정 클래스에서 우수하나, 전반적인 '스몰 데이터' 환경에서 쿼리 기반 모델의 구조적 한계 체감.

### 📍 2026-02-28 (파인튜닝 전략 심화)

- **실험**: 백본 동결(24.0%) vs 점진적 전체 해제(29.1%).
- **결론**: 도메인 갭(Domain Gap) 극복을 위해 모델 전체 가중치 조정이 필수적임을 확인.

### 📍 2026-03-01 (아키텍처 전환 및 60% 고지 점령)

- **Mask R-CNN 도입**: 소규모 데이터셋에 강건한 2-Stage 모델로 전면 개편.
- **최종 결과**: `데이터 증강` + `초반 동결 & 후반 미세조정` 전략으로 **Segm mAP50 60.0% 달성**.

### 📍 2026-03-02 (최종 최적화 및 경량화 배포 준비) 🏆 ✅

- **극한의 파인튜닝**: 3단계 점진적 동결 해제 및 채도 증강 추가 적용 (총 45 Epoch).
- **최종 성능**: **Segm mAP50 61.3%** 달성.
  - 특히, 가장 까다로운 기준인 **IoU 0.75에서 53.4%**라는 역대 최고 정밀도를 기록하며 픽셀 단위 테두리 추출의 끝판왕임을 증명.
  - 난제였던 소형/중형(Medium) 반찬류 인식률 **26.7%**로 상승.
- **모델 경량화 (Quantization)**: 서비스 연동을 위해 모델을 ONNX로 Export 후 INT8 양자화 수행.
  - **경량화 성과**: 용량 **168.94MB ➡️ 43.62MB (약 3.9배 압축)**. 모바일 및 서버 환경에서 실시간 추론이 가능한 수준으로 최종 최적화 완료.

---

## 🛠️ 결론

본 작업 공간은 단 620장의 한정된 데이터 환경에서 **'어떻게 얼리고 변형하느냐(Fine-tuning & Augmentation)'**가 모델 성능의 핵심임을 입증했습니다. 최종 산출된 43MB의 ONNX 모델은 SCAN Eat 서비스의 **"가장 오차가 적고 정밀한 음식 면적 측정 엔진"**으로 활약할 것입니다.
