# 🍱 Food Object Detection: NoisyViT + Mask2Former Analysis
## Soyun's Mask2Former Implementation

이 프로젝트는 **NoisyViT (DeiT-Small)** 백본과 **Mask2Former** 디코더를 결합한 음식 객체 탐지 및 세그멘테이션 모델의 성능 개선 과정을 다룹니다.

## 📁 파일 구조
- `config.py`: 학습 하이퍼파라미터 설정
- `models.py`: Mask2Former 모델 클래스 모음
- `utils.py`: 데이터셋 및 손실 함수 정의
- `train_soyun.py`: 메인 학습 스크립트

---

## 📊 1. 초기 성능 리포트 (Initial Performance)

현재 모델은 클래스 분류에 대해서는 유의미한 가능성을 보이고 있으나, 객체의 위치를 특정하는 박스 형성(Bounding Box Localization)에서 어려움을 겪고 있습니다.

| Metric | Value | Status |
| :--- | :--- | :--- |
| **mAP @0.5** | **0.45%** | 🟡 Underperforming |
| **Precision** | **1.83%** | 🟡 Underperforming |
| **Recall** | **1.33%** | 🟡 Underperforming |

### 🔍 기술적 분석 (Technical Analysis)
1. **분류 대비 낮은 위치 정확도**: 클래스 분류 정확도는 일정 수준 유지되고 있으나, 예측된 쿼리가 객체의 경계를 정확히 박스로 둘러싸지 못하는 현상이 발생하고 있습니다.
2. **매칭 전략 최적화 필요**: 헝가리안 매칭(Hungarian Matching)이 부재한 상태에서는 각 쿼리가 객체의 위치 정보를 정밀하게 학습하는 데 한계가 있습니다.
3. **피처 추출 구조 보완**: ViT의 단일 해상도 출력이 멀티스케일 정보를 요구하는 디코더 사양과 일치하지 않아, 정교한 박스 델타(Box Delta) 및 마스크 생성을 위한 공간 정보가 부족한 상태입니다.

---

## 🛠️ 2. 핵심 개선 방안 (Solutions)

위치 학습 성능을 보완하고 탐지 정밀도를 높이기 위해 **검증된 프레임워크 도입**과 **아키텍처 최적화**를 진행합니다.

### ① Hugging Face 공식 Mask2Former 프레임워크 도입
학습의 안정성과 정교한 박스 생성을 위해 `transformers` 라이브러리의 검증된 아키텍처를 활용합니다.
* **Hungarian Matching**: 공식 라이브러리에 내장된 `SetCriterion`을 통해 최적의 쿼리-객체 매칭을 수행하여 박스 위치 학습을 가속화합니다.
* **배경 클래스 통합**: 매칭되지 않은 쿼리에 대해 자동으로 배경을 학습시킴으로써 허위 탐지(False Positive)를 줄이고 박스의 정밀도를 높입니다.



### ② ViT 전용 피처 피라미드 (SimpleFPN) 구성
객체의 위치를 더 세밀하게 포착하기 위해 **Simple Feature Pyramid** 구조를 도입합니다.
* **Issue:** ViT는 단일 해상도 피처만을 생성하여 작은 객체나 정밀한 박스 경계를 찾는 데 공간적 정보가 제한적입니다.
* **Solution:** ViT의 중간 레이어에서 피처를 추출한 뒤, $1/4, 1/8, 1/16, 1/32$ 해상도로 재구성하는 SimpleFPN을 적용하여 박스 형성을 위한 풍부한 멀티스케일 정보를 확보합니다.



### ③ NoisyViT 백본 최적화 및 이식
안정적인 프레임워크 위에서 **NoisyViT(DeiT)**의 강점을 이식합니다.
* **Gaussian Noise 주입**: 패치 임베딩 단계에 `noise_std=0.15`를 적용하여 데이터 변화에 대한 모델의 일반화 성능을 강화합니다.
* **프레임워크 통합**: `Mask2FormerForUniversalSegmentation` 인터페이스 내에 커스텀 백본으로 NoisyViT를 통합하여 시스템의 안정성을 유지합니다.

---

## 💡 3. 기대 효과 (Expected Outcomes)

| 항목 | 기존 직접 구현 모델 | 개선된 공식 기반 모델 |
| :--- | :--- | :--- |
| **분류 정확도** | 기초적인 수준 확보 | **안정적인 클래스 예측** |
| **박스 형성 (Localization)** | **좌표 특정의 어려움** | **정교한 박스/마스크 생성** |
| **피처 구조** | 단일 스케일 (공간 정보 부족) | **SimpleFPN (정밀 위치 정보)** |

---

## 📝 결론
본 프로젝트는 초기 실험에서 확인된 **박스 형성 능력의 한계**를 해결하기 위해 **Hugging Face 공식 프레임워크**와 **SimpleFPN** 아키텍처를 도입합니다. 이를 통해 "무엇인지"에 대한 이해를 넘어, 객체의 위치를 "어디에" 정확히 표시할 수 있는 신뢰할 수 있는 음식 탐지 시스템을 구축하고자 합니다.