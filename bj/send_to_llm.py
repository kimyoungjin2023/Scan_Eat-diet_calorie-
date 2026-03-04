import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# 1. 환경 변수 로드 (API 키 숨기기)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# API 키 설정
if not api_key:
    raise ValueError("❌ .env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")

genai.configure(api_key=api_key)

def send_to_gemini(image_path, json_data):
    # 2. 모델 설정 (사용하시던 2.5 버전 유지)
    model = genai.GenerativeModel('models/gemini-2.5-pro')

    # 3. 이미지 로드
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        return f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}"
    
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }

    # 4. 프롬프트 구성 (ensure_ascii=False 추가하여 한글 깨짐 방지)
    prompt = f"""
    당신은 전문 영양사입니다. 
    사용자가 올린 식사 사진과 제가 분석한 데이터를 바탕으로 상세 영양 보고서를 작성하세요.

    [분석 데이터]
    {json.dumps(json_data, indent=2, ensure_ascii=False)}

    [요구사항]
    1. 각 음식별로 픽셀 면적을 고려하여 추정 무게(g)와 칼로리(kcal)를 산출하세요.
    2. 총 칼로리와 탄단지(탄수화물, 단백질, 지방) 비율을 알려주세요.
    3. 이 식단의 장단점과 다음 식사를 위한 건강 조언을 3줄 내외로 작성하세요.
    
    한국어로 친절하게 답변해주세요.
    """

    # 5. 답변 생성
    print("🤖 Gemini가 식단을 분석 중입니다...")
    try:
        response = model.generate_content([prompt, image_part])
        return response.text
    except Exception as e:
        # 할당량 초과(429) 등 에러 상황에 대한 안내
        if "429" in str(e):
            return "❌ 오류: 오늘 사용할 수 있는 할당량(Quota)을 초과했습니다. API Studio에서 한도를 확인하거나 잠시 후 시도하세요."
        return f"❌ Gemini 분석 중 오류 발생: {str(e)}"

# 테스트용 (필요 없으면 삭제하세요)
if __name__ == "__main__":
    test_json = {"items": ["example"]}
    # print(send_to_gemini("test_image.jpg", test_json))