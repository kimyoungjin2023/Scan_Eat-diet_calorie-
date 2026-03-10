# 🥗 SCAN Eat : Advanced Instance Segmentation & LLM Pipeline

> **"픽셀 단위의 정밀 탐지와 LLM의 시각적 추론을 결합한 지능형 식단 분석 웹 대시보드"**
> 본 프로젝트는 소규모 데이터셋의 한계를 **2-Stage Detector(Mask R-CNN)**와 **멀티모달 LLM(Gemini)**의 결합으로 극복하고, **ONNX INT8 양자화**를 통해 로컬 실행 환경 최적화까지 달성한 End-to-End AI 서비스입니다.

---

## 🏗️ 1. 시스템 아키텍처 (System Architecture)

### **AI Pipeline Flow**

1. **Image Input**: 사용자가 식단 사진 업로드 (FastAPI 기반 반응형 웹 대시보드)
2. **Instance Segmentation**: **Mask R-CNN (ONNX)** 모델이 음식 객체를 픽셀 단위로 분할 및 면적(Area) 계산
3. **Knowledge Enrichment**: 추출된 픽셀 데이터와 이미지를 **Gemini 2.5 Flash**로 전송하여 영양 성분 추론
4. **Data Storage & Visualization**: 분석 결과(칼로리, 탄단지, 조언)를 **MySQL**에 기록 및 웹 대시보드 시각화

---

## 💡 2. 핵심 기술적 성과 및 해결 과제 (Key Achievements)

### **✅ 모델 최적화 및 경량화 (Model Engineering)**

- **3단계 점진적 미세조정 (Progressive Fine-tuning)**: 백본 동결에서 전 구간 해제로 이어지는 전략을 통해 소규모 데이터(620장) 환경에서 **mAP50 61.3%** 달성
- **ONNX INT8 양자화**: FP32 가중치를 INT8로 최적화하여 모델 크기를 **168MB에서 43MB로 74% 압축**, 로컬 서버 추론 속도 2.8배 향상

### **✅ 지능형 파이프라인 구축 (LLM Integration)**

- **멀티모달 시각 추론**: Mask R-CNN이 계산한 '상대적 픽셀 면적'을 LLM에 전달하여, 정밀한 음식 무게와 칼로리를 추정
- **데이터 파싱 견고화**: LLM의 비정형 응답을 처리하기 위해 **Response MIME Type 강제** 및 **코드 블록 정제 로직**을 구현하여 데이터 무결성 확보

### **✅ 소프트웨어 공학적 접근 (Software Engineering)**

- **모듈화 설계**: 서비스 로직(`main`), AI 추론(`core`), API 통신(`api`), 유틸리티(`scripts`)로 리팩토링하여 유지보수성 극대화
- **동적 경로 제어**: `os.path`를 활용한 상대 경로 시스템 구축으로 파일 위치 변경에 따른 에러 원천 차단

---

## 🛠️ 3. 기술 스택 (Tech Stack)

- **AI/ML**: PyTorch, ONNX Runtime, Mask R-CNN, Gemini 2.5 Flash API
- **Backend**: FastAPI, Uvicorn, PyMySQL
- **Frontend**: Tailwind CSS, HTML5, JavaScript (Fetch API)
- **Database**: MySQL
- **Tools**: Git, Ngrok (외부 접속 테스트), VS Code

---

## 📅 4. 트러블슈팅 및 교훈 (Troubleshooting)

- **문제**: 폴더 구조 리팩토링 후 가중치 파일 및 환경 변수(.env) 로드 실패
- **해결**: `BASE_DIR` 기반의 동적 경로 로직을 도입하여 환경에 독립적인 코드 완성
- **문제**: LLM 응답 시 마크다운 태그 포함으로 인한 JSON 파싱 에러 (`Key Error: total_calories`)
- **해결**: Response 정제 유틸리티를 제작하고 데이터 추출 시 `.get()` 메서드를 적용하여 방어적 코딩 구현

---
