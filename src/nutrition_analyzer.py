"""
Google Gemini API 기반 영양소 분석 모듈 (안정성 강화 버전)
"""

import json
import os
import re
from typing import Dict, List
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import warnings

# Gemini 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)


class NutritionAnalyzer:
    """Google Gemini를 사용한 한국 음식 영양소 분석 (안정성 강화)"""

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Google AI API 키 (None이면 환경변수에서 가져옴)
        """
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY 환경변수를 설정하거나 api_key를 제공하세요.\n"
                "API 키 발급: https://aistudio.google.com/app/apikey"
            )

        genai.configure(api_key=api_key)

        # 생성 설정 (안정성 최우선)
        self.generation_config = {
            "temperature": 0.1,  # 매우 낮은 온도로 일관성 확보
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,  # 대폭 증가 (잘림 방지)
            "response_mime_type": "application/json",
        }

        # 시스템 지시사항 (간결하고 명확하게)
        self.system_instruction = """당신은 한국 음식 전문 영양사입니다. 
이미지와 AI 분석 데이터를 바탕으로 음식의 칼로리와 영양소를 추정해주세요.

**중요 규칙:**
1. 반드시 완전한 JSON 형식으로만 응답하세요
2. 응답을 중간에 끊지 말고 끝까지 완성하세요
3. 부피점수가 높을수록 양이 많다는 의미입니다
4. 한국 음식 기준: 밥 200g, 반찬 50g 내외로 추정하세요"""

        # 모델 변경: Flash → Pro (안정성 대폭 향상)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",  # 핵심 변경사항
            generation_config=self.generation_config,
            system_instruction=self.system_instruction,
        )

        print("🤖 Google Gemini 영양소 분석기 초기화 완료 (Model: gemini-2.5-flash)")

    def create_food_summary(self, food_analysis: List[Dict]) -> str:
        """AI 분석 결과를 간결하게 정리"""
        if not food_analysis:
            return "감지된 음식이 없습니다."

        summary_lines = []
        total_volume = sum(food["volume_score"] for food in food_analysis)

        for i, food in enumerate(food_analysis, 1):
            volume_ratio = (
                (food["volume_score"] / total_volume * 100) if total_volume > 0 else 0
            )
            summary_lines.append(
                f"{i}. {food['class_name']}: 부피점수 {food['volume_score']:.0f} "
                f"({volume_ratio:.1f}%, {food['relative_size']})"
            )

        return "\n".join(summary_lines)

    def repair_incomplete_json(self, json_str: str) -> str:
        """불완전한 JSON 자동 복구"""
        json_str = json_str.strip()

        # 마크다운 코드 블록 제거
        json_str = re.sub(r"^```json\s*", "", json_str)
        json_str = re.sub(r"\s*```$", "", json_str)
        json_str = json_str.strip()

        # 잘린 JSON 복구 (괄호 균형 맞추기)
        open_braces = json_str.count("{") - json_str.count("}")
        open_brackets = json_str.count("[") - json_str.count("]")

        if open_braces > 0 or open_brackets > 0:
            print(
                f"⚠️ JSON 잘림 감지, 자동 복구 중... (}}:{open_braces}, ]:{open_brackets})"
            )

            # 일반적으로 JSON 구조상 닫는 순서 추정
            json_str += "}" * open_braces
            json_str += "]" * open_brackets

        # 마지막 쉼표 제거 (JSON 구문 오류 방지)
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        return json_str

    def analyze_nutrition(self, image_path: str, food_analysis: List[Dict]) -> Dict:
        """
        Gemini를 사용하여 음식의 칼로리 및 영양소 분석 (재시도 로직 포함)
        """
        print(f"\n🔍 Gemini 영양소 분석 시작: {len(food_analysis)}개 음식")

        # 최대 3회 시도
        for attempt in range(3):
            try:
                # 이미지 로드
                image = Image.open(image_path)

                # 간결한 프롬프트 생성
                food_summary = self.create_food_summary(food_analysis)

                prompt = f"""이 음식 이미지를 분석해주세요.

**감지된 음식:**
{food_summary}

**응답 형식 (완전한 JSON만):**
{{
  "foods": [
    {{
      "name": "음식명",
      "estimated_weight_g": 150,
      "calories_kcal": 225,
      "carbs_g": 45.0,
      "protein_g": 4.5,
      "fat_g": 1.2,
      "sodium_mg": 200,
      "reasoning": "추정 근거"
    }}
  ],
  "total": {{
    "weight_g": 150,
    "calories_kcal": 225,
    "carbs_g": 45.0,
    "protein_g": 4.5,
    "fat_g": 1.2,
    "sodium_mg": 200
  }},
  "analysis": {{
    "meal_type": "한식",
    "balance_score": 75,
    "health_comment": "간단 평가",
    "improvement_tip": "개선 제안"
  }}
}}

위 형식을 정확히 지켜 완전한 JSON만 출력하세요."""

                # Gemini API 호출
                response = self.model.generate_content([image, prompt])
                result_text = response.text.strip()

                # JSON 파싱 시도
                try:
                    nutrition_data = json.loads(result_text)

                    # 성공 시 데이터 검증 및 반환
                    nutrition_data = self._validate_and_fix_data(
                        nutrition_data, food_analysis
                    )
                    print("✅ Gemini 영양소 분석 완료!")
                    return nutrition_data

                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON 파싱 실패 (시도 {attempt+1}/3): {e}")

                    # JSON 복구 시도
                    repaired_json = self.repair_incomplete_json(result_text)
                    try:
                        nutrition_data = json.loads(repaired_json)
                        nutrition_data = self._validate_and_fix_data(
                            nutrition_data, food_analysis
                        )
                        print("✅ JSON 복구 성공!")
                        return nutrition_data
                    except json.JSONDecodeError:
                        if attempt < 2:  # 마지막 시도가 아니면 재시도
                            print(f"   복구 실패, 재시도 중...")
                            continue
                        else:  # 마지막 시도 실패 시 fallback
                            print(f"   최종 복구 실패. 원본 응답 (처음 300자):")
                            print(f"   {result_text[:300]}")
                            raise

            except Exception as e:
                print(f"⚠️ Gemini API 호출 실패 (시도 {attempt+1}/3): {e}")
                if attempt < 2:
                    continue
                else:
                    raise

        # 모든 시도 실패 시 fallback
        print("❌ 모든 Gemini 시도 실패. 기본 데이터로 대체합니다.")
        return self._create_fallback_data(food_analysis)

    def _validate_and_fix_data(self, data: Dict, food_analysis: List[Dict]) -> Dict:
        """응답 데이터 검증 및 보정"""
        # 기본 구조 확인
        if "foods" not in data or not isinstance(data["foods"], list):
            data["foods"] = []

        # 감지된 음식 수와 응답 수가 다르면 보정
        if len(data["foods"]) != len(food_analysis):
            print(
                f"⚠️ 응답 음식 수({len(data['foods'])}) ≠ 감지 수({len(food_analysis)}), 보정 중..."
            )

            # 부족한 경우 추가
            while len(data["foods"]) < len(food_analysis):
                idx = len(data["foods"])
                data["foods"].append(
                    {
                        "name": food_analysis[idx]["class_name"],
                        "estimated_weight_g": 80,
                        "calories_kcal": 100,
                        "carbs_g": 15.0,
                        "protein_g": 5.0,
                        "fat_g": 3.0,
                        "sodium_mg": 200,
                        "reasoning": "자동 보정됨",
                    }
                )

            # 초과한 경우 제거
            data["foods"] = data["foods"][: len(food_analysis)]

        # 각 음식 필드 검증
        for food in data["foods"]:
            food.setdefault("name", "Unknown")
            food.setdefault("estimated_weight_g", 100)
            food.setdefault("calories_kcal", 150)
            food.setdefault("carbs_g", 20.0)
            food.setdefault("protein_g", 5.0)
            food.setdefault("fat_g", 3.0)
            food.setdefault("sodium_mg", 200)
            food.setdefault("reasoning", "자동 추정")

        # 총합 재계산 (AI 계산 오류 방지)
        if "total" not in data:
            data["total"] = {}

        data["total"]["weight_g"] = sum(f["estimated_weight_g"] for f in data["foods"])
        data["total"]["calories_kcal"] = sum(f["calories_kcal"] for f in data["foods"])
        data["total"]["carbs_g"] = round(sum(f["carbs_g"] for f in data["foods"]), 1)
        data["total"]["protein_g"] = round(
            sum(f["protein_g"] for f in data["foods"]), 1
        )
        data["total"]["fat_g"] = round(sum(f["fat_g"] for f in data["foods"]), 1)
        data["total"]["sodium_mg"] = sum(f["sodium_mg"] for f in data["foods"])

        # 분석 정보 기본값
        if "analysis" not in data:
            data["analysis"] = {}

        data["analysis"].setdefault("meal_type", "한식")
        data["analysis"].setdefault("balance_score", 70)
        data["analysis"].setdefault("health_comment", "영양소 분석 완료")
        data["analysis"].setdefault("improvement_tip", "균형잡힌 식사를 유지하세요")

        return data

    def _create_fallback_data(self, food_analysis: List[Dict]) -> Dict:
        """API 실패 시 한국 음식 데이터베이스 기반 대체 데이터"""
        # 사용자의 41개 클래스 완전 매핑 데이터베이스
        nutrition_db = {
            "가지볶음": {
                "weight": 70,
                "cal": 45,
                "carb": 6,
                "protein": 2,
                "fat": 2,
                "sodium": 300,
            },
            "간장게장": {
                "weight": 100,
                "cal": 180,
                "carb": 5,
                "protein": 15,
                "fat": 12,
                "sodium": 800,
            },
            "갈치구이": {
                "weight": 120,
                "cal": 200,
                "carb": 2,
                "protein": 22,
                "fat": 11,
                "sodium": 350,
            },
            "감자채볶음": {
                "weight": 80,
                "cal": 90,
                "carb": 15,
                "protein": 2,
                "fat": 2,
                "sodium": 250,
            },
            "건새우볶음": {
                "weight": 50,
                "cal": 120,
                "carb": 3,
                "protein": 15,
                "fat": 5,
                "sodium": 400,
            },
            "계란말이": {
                "weight": 80,
                "cal": 152,
                "carb": 1.5,
                "protein": 10,
                "fat": 12,
                "sodium": 180,
            },
            "계란볶음밥": {
                "weight": 280,
                "cal": 420,
                "carb": 65,
                "protein": 12,
                "fat": 8,
                "sodium": 600,
            },
            "고등어구이": {
                "weight": 100,
                "cal": 200,
                "carb": 0,
                "protein": 20,
                "fat": 13,
                "sodium": 250,
            },
            "고추": {
                "weight": 20,
                "cal": 5,
                "carb": 1,
                "protein": 0.2,
                "fat": 0,
                "sodium": 1,
            },
            "고추장아찌": {
                "weight": 30,
                "cal": 20,
                "carb": 4,
                "protein": 1,
                "fat": 0.2,
                "sodium": 500,
            },
            "김": {
                "weight": 5,
                "cal": 15,
                "carb": 1,
                "protein": 2,
                "fat": 0.3,
                "sodium": 100,
            },
            "깻잎장아찌": {
                "weight": 25,
                "cal": 15,
                "carb": 2,
                "protein": 1,
                "fat": 0.5,
                "sodium": 400,
            },
            "닭갈비": {
                "weight": 150,
                "cal": 280,
                "carb": 8,
                "protein": 25,
                "fat": 16,
                "sodium": 500,
            },
            "두부김치": {
                "weight": 120,
                "cal": 80,
                "carb": 6,
                "protein": 8,
                "fat": 4,
                "sodium": 450,
            },
            "떡갈비": {
                "weight": 100,
                "cal": 250,
                "carb": 10,
                "protein": 18,
                "fat": 15,
                "sodium": 400,
            },
            "레몬": {
                "weight": 50,
                "cal": 15,
                "carb": 5,
                "protein": 0.5,
                "fat": 0.2,
                "sodium": 1,
            },
            "마늘": {
                "weight": 10,
                "cal": 15,
                "carb": 3,
                "protein": 0.6,
                "fat": 0.1,
                "sodium": 2,
            },
            "마늘구이": {
                "weight": 30,
                "cal": 45,
                "carb": 9,
                "protein": 2,
                "fat": 0.3,
                "sodium": 5,
            },
            "멸치볶음": {
                "weight": 40,
                "cal": 80,
                "carb": 2,
                "protein": 12,
                "fat": 3,
                "sodium": 600,
            },
            "미역국": {
                "weight": 250,
                "cal": 60,
                "carb": 5,
                "protein": 3,
                "fat": 2,
                "sodium": 600,
            },
            "미역줄기볶음": {
                "weight": 70,
                "cal": 50,
                "carb": 6,
                "protein": 2,
                "fat": 2,
                "sodium": 400,
            },
            "배추김치": {
                "weight": 40,
                "cal": 12,
                "carb": 2,
                "protein": 1,
                "fat": 0.2,
                "sodium": 400,
            },
            "상추": {
                "weight": 30,
                "cal": 5,
                "carb": 1,
                "protein": 0.5,
                "fat": 0.1,
                "sodium": 2,
            },
            "새송이버섯": {
                "weight": 60,
                "cal": 20,
                "carb": 3,
                "protein": 2,
                "fat": 0.2,
                "sodium": 5,
            },
            "시금치나물": {
                "weight": 60,
                "cal": 30,
                "carb": 3,
                "protein": 2,
                "fat": 1,
                "sodium": 350,
            },
            "쌀밥": {
                "weight": 210,
                "cal": 315,
                "carb": 68,
                "protein": 5.5,
                "fat": 0.6,
                "sodium": 2,
            },
            "쌈채소": {
                "weight": 50,
                "cal": 10,
                "carb": 2,
                "protein": 1,
                "fat": 0.1,
                "sodium": 5,
            },
            "양념게장": {
                "weight": 120,
                "cal": 200,
                "carb": 8,
                "protein": 18,
                "fat": 10,
                "sodium": 900,
            },
            "양념장어구이": {
                "weight": 150,
                "cal": 350,
                "carb": 10,
                "protein": 20,
                "fat": 25,
                "sodium": 600,
            },
            "잡곡밥": {
                "weight": 210,
                "cal": 330,
                "carb": 65,
                "protein": 7,
                "fat": 2,
                "sodium": 5,
            },
            "잡채": {
                "weight": 100,
                "cal": 180,
                "carb": 25,
                "protein": 5,
                "fat": 6,
                "sodium": 400,
            },
            "장어구이": {
                "weight": 150,
                "cal": 300,
                "carb": 5,
                "protein": 18,
                "fat": 22,
                "sodium": 450,
            },
            "장조림": {
                "weight": 80,
                "cal": 150,
                "carb": 8,
                "protein": 15,
                "fat": 6,
                "sodium": 800,
            },
            "주꾸미볶음": {
                "weight": 120,
                "cal": 140,
                "carb": 8,
                "protein": 18,
                "fat": 4,
                "sodium": 500,
            },
            "진미채볶음": {
                "weight": 50,
                "cal": 100,
                "carb": 5,
                "protein": 10,
                "fat": 4,
                "sodium": 600,
            },
            "청포묵무침": {
                "weight": 80,
                "cal": 50,
                "carb": 8,
                "protein": 3,
                "fat": 1,
                "sodium": 350,
            },
            "총각김치": {
                "weight": 40,
                "cal": 15,
                "carb": 2.5,
                "protein": 1.2,
                "fat": 0.3,
                "sodium": 420,
            },
            "콩나물무침": {
                "weight": 70,
                "cal": 35,
                "carb": 4,
                "protein": 3,
                "fat": 1,
                "sodium": 300,
            },
            "토마토": {
                "weight": 80,
                "cal": 15,
                "carb": 3,
                "protein": 1,
                "fat": 0.2,
                "sodium": 5,
            },
            "피클": {
                "weight": 30,
                "cal": 10,
                "carb": 2,
                "protein": 0.2,
                "fat": 0.1,
                "sodium": 300,
            },
            "호박무침": {
                "weight": 70,
                "cal": 25,
                "carb": 4,
                "protein": 1,
                "fat": 0.5,
                "sodium": 200,
            },
        }

        foods = []
        for food in food_analysis:
            name = food["class_name"]

            # 정확한 매칭 우선
            nutrition = nutrition_db.get(name)
            if nutrition is None:
                # 부분 매칭 시도
                for key in nutrition_db:
                    if key in name or name in key:
                        nutrition = nutrition_db[key]
                        break

            # 기본값
            if nutrition is None:
                nutrition = {
                    "weight": 80,
                    "cal": 100,
                    "carb": 15,
                    "protein": 5,
                    "fat": 3,
                    "sodium": 200,
                }

            # 부피점수 기반 비율 조정 (0.5배~2배 범위)
            volume_ratio = max(0.5, min(2.0, food["volume_score"] / 10000))

            foods.append(
                {
                    "name": name,
                    "estimated_weight_g": int(nutrition["weight"] * volume_ratio),
                    "calories_kcal": int(nutrition["cal"] * volume_ratio),
                    "carbs_g": round(nutrition["carb"] * volume_ratio, 1),
                    "protein_g": round(nutrition["protein"] * volume_ratio, 1),
                    "fat_g": round(nutrition["fat"] * volume_ratio, 1),
                    "sodium_mg": int(nutrition["sodium"] * volume_ratio),
                    "reasoning": f"데이터베이스 기반 추정 (부피점수: {food['volume_score']:.0f})",
                }
            )

        total = {
            "weight_g": sum(f["estimated_weight_g"] for f in foods),
            "calories_kcal": sum(f["calories_kcal"] for f in foods),
            "carbs_g": round(sum(f["carbs_g"] for f in foods), 1),
            "protein_g": round(sum(f["protein_g"] for f in foods), 1),
            "fat_g": round(sum(f["fat_g"] for f in foods), 1),
            "sodium_mg": sum(f["sodium_mg"] for f in foods),
        }

        return {
            "foods": foods,
            "total": total,
            "analysis": {
                "meal_type": "한식",
                "balance_score": 70,
                "health_comment": "데이터베이스 기반 영양소 분석입니다.",
                "improvement_tip": "다양한 채소를 추가하여 영양 균형을 맞춰보세요.",
            },
        }

    def print_nutrition_report(self, nutrition_data: Dict):
        """영양소 분석 결과 출력"""
        if not nutrition_data:
            print("❌ 분석 데이터가 없습니다.")
            return

        print("\n" + "=" * 80)
        print("📊 Gemini 영양소 분석 결과")
        print("=" * 80)

        # 개별 음식
        for i, food in enumerate(nutrition_data.get("foods", []), 1):
            print(f"\n🍽️ {i}. {food.get('name', 'Unknown')}")
            print(f"   중량: {food.get('estimated_weight_g', 0)}g")
            print(f"   칼로리: {food.get('calories_kcal', 0)} kcal")
            print(
                f"   탄수화물: {food.get('carbs_g', 0)}g | 단백질: {food.get('protein_g', 0)}g | 지방: {food.get('fat_g', 0)}g"
            )
            print(f"   나트륨: {food.get('sodium_mg', 0)}mg")
            print(f"   💡 {food.get('reasoning', '')}")

        # 총합
        total = nutrition_data.get("total", {})
        if total:
            print(f"\n📈 총 영양소:")
            print(f"   총 중량: {total.get('weight_g', 0)}g")
            print(f"   총 칼로리: {total.get('calories_kcal', 0)} kcal")
            print(f"   총 탄수화물: {total.get('carbs_g', 0)}g")
            print(f"   총 단백질: {total.get('protein_g', 0)}g")
            print(f"   총 지방: {total.get('fat_g', 0)}g")
            print(f"   총 나트륨: {total.get('sodium_mg', 0)}mg")

        # 분석 의견
        analysis = nutrition_data.get("analysis", {})
        if analysis:
            print(f"\n💡 식단 분석:")
            print(f"   식사 유형: {analysis.get('meal_type', '')}")
            print(f"   균형 점수: {analysis.get('balance_score', 0)}/100")
            print(f"   건강 평가: {analysis.get('health_comment', '')}")
            print(f"   개선 제안: {analysis.get('improvement_tip', '')}")
        print("=" * 80)
