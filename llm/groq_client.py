from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
import json
import re

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def build_prompt(self, detection_results: list) -> str:
        food_list = ""
        for i, item in enumerate(detection_results, 1):
            food_list += (
                f"{i}. 음식명: {item['class']}\n"
                f"   - 픽셀 가로: {item['pixel_width']}px\n"
                f"   - 픽셀 세로: {item['pixel_height']}px\n"
                f"   - 마스크 면적: {item['mask_area']}px²\n"
                f"   - 이미지 대비 크기 비율: {item['area_ratio']}%\n"
                f"   - 깊이값 (0=가까움 1=멀음): {item['avg_depth']}\n"
                f"   - 카메라와의 거리: {item['depth_comment']}\n"
            )

        prompt = f"""
당신은 음식 영양사입니다.
아래는 이미지(640x640)에서 탐지된 음식과 Segmentation + 깊이 정보입니다.

{food_list}

분석 기준:
- 이미지 대비 크기 비율이 크면 양이 많음
- 깊이값이 낮을수록 (가까울수록) 실제 크기가 더 큰 음식
- 깊이값이 높을수록 (멀수록) 실제 크기가 작을 수 있음
- 위 두 정보를 종합해서 실제 중량을 추정

1. 각 음식의 예상 중량 (g)
2. 각 음식의 예상 칼로리 (kcal)
3. 각 음식의 주요 영양소 (탄수화물, 단백질, 지방)
4. 전체 칼로리 합계
5. 식단 한줄 평가

JSON 형식으로만 응답해주세요. 다른 텍스트 없이 JSON만:
{{
    "foods": [
        {{
            "name": "음식명",
            "weight_g": 숫자,
            "calories_kcal": 숫자,
            "nutrients": {{
                "carbs_g": 숫자,
                "protein_g": 숫자,
                "fat_g": 숫자
            }}
        }}
    ],
    "total_calories": 숫자,
    "diet_comment": "한줄 평가"
}}
"""
        return prompt

    def analyze(self, detection_results: list) -> dict:
        prompt = self.build_prompt(detection_results)

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 음식 영양사입니다. 반드시 JSON 형식으로만 응답하세요."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
            )
            text = response.choices[0].message.content
            text = re.sub(r"```json|```", "", text).strip()
            result = json.loads(text)
            return result

        except Exception as e:
            print(f"Groq 분석 오류: {e}")
            return {}

    def print_result(self, result: dict):
        if not result:
            print("분석 결과 없음")
            return

        print("\n" + "=" * 40)
        print("  🍽️  음식 분석 결과")
        print("=" * 40)

        for food in result.get("foods", []):
            print(f"\n[ {food['name']} ]")
            print(f"  중량    : {food['weight_g']}g")
            print(f"  칼로리  : {food['calories_kcal']}kcal")
            print(f"  탄수화물: {food['nutrients']['carbs_g']}g")
            print(f"  단백질  : {food['nutrients']['protein_g']}g")
            print(f"  지방    : {food['nutrients']['fat_g']}g")

        print(f"\n총 칼로리 : {result.get('total_calories')}kcal")
        print(f"식단 평가 : {result.get('diet_comment')}")
        print("=" * 40)