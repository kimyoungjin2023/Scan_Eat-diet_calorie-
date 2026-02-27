# 🍕 SCAN Eat - Mask2Former Segmentation Pipeline (BJ's Workspace)

> **"박스(BBox)가 아닌 픽셀(Pixel)을 딴다!"** > SCAN Eat 프로젝트의 정밀한 음식 영역 추출을 위한 Mask2Former 파이프라인 구축 브랜치입니다.

---

## 🏗️ 1. 모델 아키텍처 (Model Architecture)

- **Task**: Instance Segmentation (음식 개별 객체 분할)
- **Head**: Mask2Former - 쿼리 기반 픽셀 분할 (COCO 데이터셋 사전 학습 모델 적용)
- **Backbone**: Swin-Transformer (Tiny) - 사전 학습된 시각 지능 엔진
- **Framework**: Detectron2 (Windows용 커스텀 빌드)

## 📊 2. 데이터셋 (Dataset)

- **도메인**: 한식 위주의 음식 이미지 (총 44개 클래스)
- **Train**: 620장 / **Valid**: 38장
- **포맷**: COCO format JSON (`_annotations.coco_final.json`)
- **최적화**: 윈도우 환경 메모리 누수 방지를 위한 `NUM_WORKERS = 0` 설정

## 💡 3. 핵심 구현 및 트러블슈팅

- **베테랑 모델(Pre-trained) 지식 이식**:
  인코더(Swin)만 학습된 상태에서 디코더(Mask2Former Head)까지 COCO 데이터로 학습된 전체 가중치를 로드하여 '백지상태'의 헤드 성능 문제를 근본적으로 해결함.
- **네트워크 보안 및 403 Forbidden 해결**:
  서버 측 직접 다운로드 차단 문제를 해결하기 위해 GitHub Model Zoo에서 가중치 파일(`86143f` 버전)을 수동 확보 후 로컬 경로 연결.
- **GitHub 인증 복구**:
  보안 정책 변경으로 인한 Push 실패를 PAT(Personal Access Token) 발급 및 자격 증명 업데이트를 통해 해결.

---

## 📅 작업 일지 (Dev Log)

### 📍 2026-02-23 ~ 2026-02-25 (환경 구축 및 1차 PoC)

- Detectron2 및 Mask2Former 로컬 설치.
- 1차 학습 진행 (10,000 iter) 결과 Segm mAP50 **38.6%** 달성.

### 📍 2026-02-26 (전략 최적화)

- 하이퍼파라미터 미세 조정: `Base LR` 0.00005 하향, `Weight Decay` 0.05 적용.
- 데이터 증강(RandomFlip, Multi-scale) 적용 완료.

### 📍 2026-02-27 (2차 파인튜닝 및 베테랑 모델 이식) ✅

- **모델 보완**: COCO 데이터셋으로 '칼질 실력'을 쌓은 베테랑 헤드 가중치 이식 성공.
- **성능 분석 (Valid 38장)**:
  - **종합 성적**: Segm AP50 **31.865%** 기록 (수치보다 정교한 경계선 추출에 집중).
  - **주요 성과**: 특정 음식군에서 높은 정확도 확보.
    - 간장게장 (GanjangCrab): **85.155%**
    - 진미채볶음 (SpicyDriedSquidBokkeum): **80.000%**
    - 닭갈비 (Dakgalbi): **70.957%**
  - **특이사항**: BBox mAP 0은 모델 구조적 특징임을 재확인, 검증 셋 부재 클래스는 `nan` 처리됨.
