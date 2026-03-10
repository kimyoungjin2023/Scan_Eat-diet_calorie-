import google.generativeai as genai
import json
import os
import time
from dotenv import load_dotenv

# 1. 환경 변수 로드
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, "..", ".env"))

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ .env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")

genai.configure(api_key=api_key)

def send_to_gemini(image_path, json_data):
    """
    이미지와 분석 데이터를 바탕으로 상세 영양소 보고서를 생성합니다.
    """
    try:
        # 모델 설정
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config={"response_mime_type": "application/json"}
        )
    except Exception as e:
        return json.dumps({"error": f"모델 초기화 실패: {str(e)}"})

    # 이미지 로드
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        return json.dumps({"error": "이미지 파일을 찾을 수 없습니다."})
    
    image_part = {"mime_type": "image/jpeg", "data": image_bytes}

    # 프롬프트 구성
    prompt = f"""
    당신은 전문 영양사입니다. 다음 데이터를 분석하여 반드시 아래 JSON 형식을 지켜 답변하세요.
    JSON 외에 다른 말은 절대 하지 마세요.

    {{
      "total_calories": 0,
      "total_carbs": 0.0,
      "total_protein": 0.0,
      "total_fat": 0.0,
      "total_sugar": 0.0,
      "total_sodium": 0.0,
      "total_cholesterol": 0.0,
      "items": [
        {{ "name": "음식명", "weight": 0.0, "calories": 0 }}
      ],
      "advice": "전문적인 영양 조언 3줄 내외 (친절하게)"
    }}

    [분석 데이터]
    {json.dumps(json_data, indent=2, ensure_ascii=False)}
    """

    print("🤖 Gemini가 상세 영양 분석을 진행 중입니다...")
    
    try:
        response = model.generate_content([prompt, image_part])
        time.sleep(1) 
        
        # 📍 [수정 포인트] 응답 텍스트 정제 로직 추가
        raw_text = response.text.strip()
        
        # 마크다운 코드 블록(```json)이 포함된 경우 제거
        if raw_text.startswith("```"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        # 정제된 텍스트가 올바른 JSON인지 확인 후 반환
        return raw_text 

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return json.dumps({"error": "API 할당량 초과"})
        return json.dumps({"error": error_msg})