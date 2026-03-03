import google.generativeai as genai
import json



# 1. API 키 설정
genai.configure(api_key="AIzaSyAYA2mmaqifQGm5x0jerOYy0RaIEMl62sI")

# print("🔍 사용 가능한 모델 리스트:")
# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

def send_to_gemini(image_path, json_data):
    # 2. 모델 설정 (반드시 소문자로 작성해야 합니다)
    # 'models/'를 붙여주는 것이 가장 표준적인 형식입니다.
    model = genai.GenerativeModel('models/gemini-2.5-pro')

    # 3. 이미지 로드
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }

    # 4. 프롬프트 구성
    prompt = f"""
    당신은 전문 영양사입니다. 
    사용자가 올린 식사 사진과 제가 분석한 데이터를 바탕으로 상세 영양 보고서를 작성하세요.

    [분석 데이터]
    {json.dumps(json_data, indent=2)}

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
        return f"❌ Gemini 분석 중 오류 발생: {str(e)}"