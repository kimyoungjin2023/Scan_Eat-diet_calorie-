"""
프로젝트 전체 설정 파일 - 기존 데이터 구조 직접 활용
"""

from pathlib import Path

# ==================== 경로 설정 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"  # 원본 데이터 직접 사용
BACKUP_DIR = PROJECT_ROOT / "backup"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ==================== 한글 매핑 (중복 통합 포함) ====================
KOREAN_FOOD_MAPPING = {
    # 볶음류
    "Bokkeum_Dakgalbi": "닭갈비",
    "Bokkeum_DriedShrimpBokkeum": "건새우볶음",
    "Bokkeum_DriedSquidBokkeum": "진미채볶음",
    "Bokkeum_EggplantBokkeum": "가지볶음",
    "Bokkeum_Japchae": "잡채",
    "Bokkeum_MiyeokJulgiBokkeum": "미역줄기볶음",
    "Bokkeum_PotatoSliceBokkeum": "감자채볶음",
    "Bokkeum_SpicyDriedSquidBokkeum": "진미채볶음",  # 중복 통합
    "Bokkeum_StirFriedAnchovies": "멸치볶음",
    "Bokkeum_WebfootOctopusBokkeum": "주꾸미볶음",
    # 과일
    "Fruit_Lemon": "레몬",
    "Fruit_Tomato": "토마토",
    # 김/해조류
    "Gim": "김",
    # 구이류
    "Grilled_Garlic": "마늘구이",
    "Grilled_GrilledCutlassfish": "갈치구이",
    "Grilled_GrilledEel": "장어구이",
    "Grilled_GrilledMackerel": "고등어구이",
    "Grilled_GrilledSpicesEel": "양념장어구이",
    "Grilled_Tteokgalbi": "떡갈비",
    # 국/탕류
    "Guk_MiyeokGuk": "미역국",
    # 젓갈류
    "Jeotgal_GanjangCrab": "간장게장",
    "Jeotgal_SpicyMarinatedCrab": "양념게장",
    # 조림류
    "Jorim_Janjorim": "장조림",
    # 김치류 (핵심 중복 통합!)
    "Kimch_Kimch": "배추김치",
    "Kimchi": "배추김치",
    "Kimchi_Kimchi": "배추김치",
    "Kimchi_YoungRadishKimchi": "총각김치",
    # 무침류
    "Muchim_KongnamulMuchim": "콩나물무침",
    "Muchim_ZucchiniMuchim": "호박무침",
    "Muchim_cheongpomungMuchim": "청포묵무침",
    # 버섯류
    "Mushroom_Mushroom_KingOysterMushroom": "새송이버섯",
    # 나물류
    "Namul_Sigeumchinamul": "시금치나물",
    # 기타
    "None_EggFriedRice": "계란볶음밥",
    "None_TofuKimchi": "두부김치",
    "Pancake_EggRoll": "계란말이",
    # 장아찌류
    "Pickled_Gochujangajji": "고추장아찌",
    "Pickled_KkaennipJangajji": "깻잎장아찌",
    "Pickled_Pickle": "피클",
    # 밥류
    "Rice_MixedGrainRice": "잡곡밥",
    "Rice_WhiteRice": "쌀밥",
    # 채소류
    "Vegetable_Garlic": "마늘",
    "Vegetable_Lettuce": "상추",
    "Vegetable_Ssamvegetables": "쌈채소",
    "Vegetable_gochu": "고추",
}

# ==================== 학습 설정 (RTX 3060 최적화) ====================
TRAIN_CONFIG = {
    "model": "yolo11m-seg.pt",
    "epochs": 30,
    "imgsz": 512,
    "batch": 8,
    "device": 0,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience": 30,
    "amp": True,
    "cache": "ram",
    "workers": 4,
    "project": str(MODELS_DIR),
    "name": "yolov11_food",
    "exist_ok": True,
    "save": True,
    "save_period": 20,
    "plots": True,
    "verbose": True,
}

# ==================== 데이터 증강 설정 ====================
AUGMENTATION_CONFIG = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 15,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0001,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.1,
}


if __name__ == "__main__":
    print("프로젝트 전체 설정이 로드되었습니다.")
    print(f"데이터 디렉토리: {DATA_DIR}")
    print(f"모델 저장 디렉토리: {MODELS_DIR}")
    print(f"결과 저장 디렉토리: {RESULTS_DIR}")
    print("학습 설정:", TRAIN_CONFIG)
    print("데이터 증강 설정:", AUGMENTATION_CONFIG)
