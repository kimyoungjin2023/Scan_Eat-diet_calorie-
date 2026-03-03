# 🍕 SCAN Eat - Advanced Instance Segmentation & LLM Pipeline (BJ's Workspace)

> **"박스(BBox)가 아닌 픽셀(Pixel)을 따고, LLM의 두뇌로 칼로리를 잰다!"** > SCAN Eat 프로젝트의 정밀한 음식 영역 추출 및 지능형 영양 분석 파이프라인 구축 브랜치입니다. 소규모 데이터셋의 한계를 극복하기 위해 아키텍처를 전환하며 성능을 극대화하고, 모델 경량화(ONNX INT8)부터 **멀티모달 LLM(Gemini 2.5 Flash) 자동 연동**까지 End-to-End로 완성한 기록을 담고 있습니다.

## 🏗️ 1. 모델 아키텍처 (Model Architecture)

### 🚀 [Current SOTA & Deployed] Mask R-CNN (최종 채택 및 경량화)

- **Task**: Instance Segmentation (음식 개별 객체 분할)

- **Backbone**: ResNet50 + FPN (Feature Pyramid Network)

- **Framework**: PyTorch ➡️ **ONNX Runtime (INT8 Quantized)**

- **Strategy**: 2-Stage Detector 구조를 통한 고정밀 ROI 추출 및 픽셀 단위 마스크 생성

### 🕰️ [Legacy] Mask2Former (초기 탐색)

- **Task**: Query-based Pixel Segmentation

- **Framework**: Detectron2 (Windows용 커스텀 빌드)

## 📊 2. 데이터셋 (Dataset)

- **도메인**: 한식 위주의 음식 이미지 (총 44개 클래스)

- **Train**: 620장 / **Valid**: 38장 (오프라인 증강 없이 온라인 증강만으로 성능 극대화)

- **포맷**: COCO format JSON

- **최적화**: 윈도우 환경 메모리 누수 방지를 위한 `NUM_WORKERS = 0` 설정

## 💡 3. 핵심 구현 및 기술적 성과

### 🎯 [Part 1] Mask R-CNN 최적화 및 경량화 전략

- **3단계 점진적 파인튜닝 (Progressive Unfreezing)**: 백본 완전 동결 ➡️ 상위 레이어 해제 ➡️ 전 구간 미세조정(lr=0.0001)으로 픽셀 정밀도 극대화.

- **극한의 온라인 데이터 증강 (Online Augmentation)**: BBox와 Mask 좌표 동기화 수동 Flip, 조명 변화 대응 미세 Color Jitter 적용.

- **배포용 모델 경량화 (INT8 Quantization)**: 정확도 손실 없이 **모델 용량을 168.94MB ➡️ 43.62MB (약 3.9배 압축)**. 모바일/서버 환경 실시간 추론 확보.

### 🧠 [Part 2] 지능형 후처리 및 멀티모달 LLM 연동 (최종 파이프라인)

- **공학적 후처리 (Post-processing)**:
  - **NMS 로직**: IoU 기반 중복 검출 박스 제거로 음식 개수 오차 방지.

  - **정밀 픽셀 카운팅**: 추출된 마스크 배열 기반으로 각 음식의 'Pixel Area'를 정밀 계산하여 JSON 데이터로 구조화.

- **Gemini 2.5 API 연동 (Multimodal AI)**:
  - 기준물체 없이도, 모델이 뽑은 **'상대적 픽셀 면적 데이터'**와 **'결과 이미지'**를 동시에 LLM에 전송.

  - 단순 분류를 넘어, LLM의 시각적 추론을 통해 **실제 무게(g), 칼로리(kcal), 탄단지 비율 산출 및 맞춤형 영양 조언**을 자동 생성하는 엔드투엔드(End-to-End) 시스템 구축 완료.

## 📅 작업 일지 (Dev Log)

### 📍 2026-02-23 ~ 2026-02-28 (초기 탐색 및 전략 수정)

- Detectron2 기반 Mask2Former 적용 시도.

- 스몰 데이터(620장) 환경에서 쿼리 기반 모델의 구조적 한계 체감 (mAP50 31.8%).

- 도메인 갭 극복을 위해 모델 전체 가중치 조정 및 2-Stage 아키텍처로의 전환 결정.

### 📍 2026-03-01 (아키텍처 전환 및 60% 고지 점령)

- **Mask R-CNN 도입**: 소규모 데이터셋에 강건한 모델로 전면 개편.

- **최종 결과**: `데이터 증강` + `초반 동결 & 후반 미세조정` 전략으로 **Segm mAP50 60.0% 달성**.

### 📍 2026-03-02 (최종 최적화 및 경량화 배포 준비)

- 3단계 점진적 동결 해제 적용으로 **최종 성능 Segm mAP50 61.3%** 달성.

- 난제였던 소형/중형(Medium) 반찬류 인식률 대폭 상승 (26.7%).

- PyTorch 모델을 ONNX로 Export 후 동적 양자화 수행 ➡️ **43.62MB 모델 산출**.

### 📍 2026-03-03 (추론 파이프라인 완성 및 멀티모달 LLM 연동) 🚀 ✅

- ONNX 모델 기반 추론 스크립트(`visualize_results.py`) 작성 및 후처리 로직 구현.

- **마스크 면적(Area) 추출 로직**: 픽셀 수를 합산하여 객체별 상대적 크기를 수치화.

- **LLM 연동 자동화 (`send_to_llm.py`)**:
  - 구글 최신 **Gemini 2.5 Flash** API를 연동하여, JSON 수치 데이터와 이미지를 함께 전송.

  - 모델이 탐지하지 못한 배경 음식(밥, 김치 등)까지 LLM이 시각적으로 보완하여 최종 칼로리(예: 710kcal)를 산출해 내는 **AI 영양사 리포트 자동화 성공**.
