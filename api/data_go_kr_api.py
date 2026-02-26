import requests
import pandas as pd
import json
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

load_dotenv()


class FoodNutritionDBAPI:
    """식품의약품안전처 식품영양성분DB 정보 API"""

    def __init__(self, service_key: str = None):
        self.service_key = (
            "faeb0fa82ae1ee3711ea7bac5594068857fafa29f42e4759d57973650d56690d"
            or os.getenv("PUBLIC_DATA_API_KEY")
        )
        self.base_url = "https://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02"
        self.endpoint = f"{self.base_url}/getFoodNtrCpntDbInq02"

    def get_food_nutrition(
        self,
        food_name: str = None,  # FOOD_NM_KR: 식품명
        research_date: str = None,  # RESEARCH_YMD: 데이터생성일자
        maker_name: str = None,  # MAKER_NM: 업체명
        food_category: str = None,  # FOOD_CAT1_NM: 식품대분류명
        item_report_no: str = None,  # ITEM_REPORT_NO: 품목제조보고번호
        update_date: str = None,  # UPDATE_DATE: 데이터수정일자
        db_class_name: str = None,  # DB_CLASS_NM: 품목대표/상용제품
        page_no: int = 1,
        num_of_rows: int = 20,
        response_type: str = "json",
    ) -> Optional[pd.DataFrame]:
        """
        식품영양성분 조회

        Args:
            food_name: 식품명 (예: "김치", "라면")
            research_date: 데이터생성일자 (예: "20230101")
            maker_name: 업체명
            food_category: 식품대분류명
            item_report_no: 품목제조보고번호
            update_date: 데이터수정일자
            db_class_name: 품목대표/상용제품
            page_no: 페이지 번호
            num_of_rows: 한 페이지 결과 수
            response_type: 응답 형식 ('json' 또는 'xml')

        Returns:
            pd.DataFrame: 영양성분 데이터프레임
        """

        # 필수 파라미터
        params = {
            "serviceKey": self.service_key,
            "pageNo": str(page_no),
            "numOfRows": str(num_of_rows),
            "type": response_type,
        }

        # 선택 파라미터 추가
        if food_name:
            params["FOOD_NM_KR"] = food_name
        if research_date:
            params["RESEARCH_YMD"] = research_date
        if maker_name:
            params["MAKER_NM"] = maker_name
        if food_category:
            params["FOOD_CAT1_NM"] = food_category
        if item_report_no:
            params["ITEM_REPORT_NO"] = item_report_no
        if update_date:
            params["UPDATE_DATE"] = update_date
        if db_class_name:
            params["DB_CLASS_NM"] = db_class_name

        try:
            print(f"API 호출 중: {food_name or '전체'}")
            response = requests.get(self.endpoint, params=params, timeout=30)
            response.raise_for_status()

            if response_type == "json":
                return self._parse_json_response(response.json())
            else:
                return self._parse_xml_response(response.text)

        except requests.exceptions.RequestException as e:
            print(f"API 호출 오류: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print(f"응답 내용: {response.text[:500]}")
            return None

    def _parse_json_response(self, data: dict) -> Optional[pd.DataFrame]:
        """JSON 응답 파싱"""
        try:
            # 응답 구조 확인
            body = data.get("body", {})
            total_count = body.get("totalCount", 0)
            items = body.get("items", [])

            print(f"총 {total_count}건 중 {len(items)}건 조회")

            if not items:
                print("검색 결과가 없습니다.")
                return None

            # DataFrame 생성
            df = pd.DataFrame(items)

            # 컬럼명 한글로 변환 (엑셀 출력 메시지 기준)
            column_mapping = {
                "FOOD_NM_KR": "식품명",
                "DB_CLASS_NM": "품목대표/상용제품",
                "FOOD_CAT1_NM": "식품대분류명",
                "FOOD_CAT2_NM": "식품중분류명",
                "FOOD_CAT3_NM": "식품소분류명",
                "FOOD_CAT4_NM": "식품세분류명",
                "SERVING_SIZE": "영양성분함량기준량",
                "AMT_NUM1": "열량(kcal)",
                "AMT_NUM2": "수분(g)",
                "AMT_NUM3": "단백질(g)",
                "AMT_NUM4": "지방(g)",
                "AMT_NUM5": "회분(g)",
                "AMT_NUM6": "탄수화물(g)",
                "AMT_NUM7": "당류(g)",
                "AMT_NUM8": "식이섬유(g)",
                "AMT_NUM9": "칼슘(mg)",
                "AMT_NUM10": "철(mg)",
                "AMT_NUM11": "인(mg)",
                "AMT_NUM12": "칼륨(mg)",
                "AMT_NUM13": "나트륨(mg)",
                "AMT_NUM14": "비타민A(μg RAE)",
                "AMT_NUM15": "레티놀(μg)",
                "AMT_NUM16": "베타카로틴(μg)",
                "AMT_NUM17": "티아민(mg)",
                "AMT_NUM18": "리보플라빈(mg)",
                "AMT_NUM19": "니아신(mg)",
                "AMT_NUM20": "비타민C(mg)",
                "AMT_NUM21": "비타민D(μg)",
                "AMT_NUM22": "콜레스테롤(mg)",
                "AMT_NUM23": "포화지방산(g)",
                "AMT_NUM24": "트랜스지방산(g)",
                "MAKER_NM": "업체명",
                "ITEM_REPORT_NO": "품목제조보고번호",
                "RESEARCH_YMD": "데이터생성일자",
                "UPDATE_DATE": "데이터수정일자",
            }

            # 존재하는 컬럼만 변환
            rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=rename_dict)

            return df

        except Exception as e:
            print(f"데이터 파싱 오류: {e}")
            return None

    def _parse_xml_response(self, xml_text: str) -> Optional[pd.DataFrame]:
        """XML 응답 파싱"""
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_text)
            items = root.findall(".//item")

            if not items:
                print("검색 결과가 없습니다.")
                return None

            data_list = []
            for item in items:
                item_dict = {}
                for child in item:
                    item_dict[child.tag] = child.text
                data_list.append(item_dict)

            return pd.DataFrame(data_list)

        except ET.ParseError as e:
            print(f"XML 파싱 오류: {e}")
            return None

    def search_foods(
        self, food_names: List[str], num_of_rows: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """여러 식품 검색"""
        results = {}

        for food_name in food_names:
            df = self.get_food_nutrition(food_name=food_name, num_of_rows=num_of_rows)
            if df is not None:
                results[food_name] = df
            print("-" * 50)

        return results

    def get_nutrition_summary(self, food_name: str) -> Optional[pd.DataFrame]:
        """주요 영양성분만 요약 조회"""
        df = self.get_food_nutrition(food_name=food_name, num_of_rows=10)

        if df is None:
            return None

        # 주요 컬럼만 선택
        main_columns = [
            "식품명",
            "열량(kcal)",
            "탄수화물(g)",
            "단백질(g)",
            "지방(g)",
            "당류(g)",
            "나트륨(mg)",
            "콜레스테롤(mg)",
            "영양성분함량기준량",
        ]

        existing_cols = [col for col in main_columns if col in df.columns]
        return df[existing_cols]


# ============================================
# 실행 예시
# ============================================
if __name__ == "__main__":

    # API 키 설정 (직접 입력하거나 .env 파일 사용)
    # API_KEY = "발급받은_서비스키"
    # api = FoodNutritionDBAPI(service_key=API_KEY)

    api = FoodNutritionDBAPI()

    # API 키 확인
    if not api.service_key:
        print("❌ API 키가 설정되지 않았습니다.")
        print("방법 1: .env 파일에 PUBLIC_DATA_API_KEY=서비스키 추가")
        print("방법 2: FoodNutritionDBAPI(service_key='서비스키')로 직접 입력")
        exit()

    print("=" * 60)
    print("식품영양성분DB API 테스트")
    print("=" * 60)

    # 테스트 1: 단일 식품 검색
    print("\n[테스트 1] 김치찌개 영양성분 조회")
    result = api.get_food_nutrition(food_name="김치찌개", num_of_rows=5)
    if result is not None:
        print(result.to_string())

    # 테스트 2: 주요 영양성분 요약
    print("\n" + "=" * 60)
    print("[테스트 2] 라면 주요 영양성분 요약")
    summary = api.get_nutrition_summary("라면")
    if summary is not None:
        print(summary.to_string())

    # 테스트 3: 여러 식품 검색
    print("\n" + "=" * 60)
    print("[테스트 3] 여러 식품 검색")
    foods = ["비빔밥", "삼겹살", "떡볶이"]
    results = api.search_foods(foods, num_of_rows=3)

    for food_name, df in results.items():
        print(f"\n▶ {food_name}: {len(df)}건")
        if "열량(kcal)" in df.columns:
            print(df[["식품명", "열량(kcal)"]].head())

    # 테스트 4: 식품 카테고리로 검색
    print("\n" + "=" * 60)
    print("[테스트 4] 식품대분류명으로 검색")
    result = api.get_food_nutrition(food_category="면류", num_of_rows=5)
    if result is not None:
        print(result[["식품명", "열량(kcal)"]].to_string())

    # 테스트 5: 결과 엑셀 저장
    print("\n" + "=" * 60)
    print("[테스트 5] 결과 엑셀 저장")
    result = api.get_food_nutrition(food_name="김치", num_of_rows=20)
    if result is not None:
        result.to_excel("김치_영양성분.xlsx", index=False)
        print("✅ '김치_영양성분.xlsx' 파일 저장 완료")
