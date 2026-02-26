"""
YOLOv11 모델 학습
"""

import torch
from ultralytics import YOLO
from pathlib import Path

from config import *
from utils import *


def check_environment():
    """학습 환경 확인"""
    print_section("🔍 학습 환경 확인")

    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"현재 할당: {allocated:.2f} GB")
        print(f"현재 예약: {reserved:.2f} GB")
    else:
        print("⚠️ GPU를 사용할 수 없습니다. CPU로 학습합니다.")


def validate_data():
    """데이터 유효성 검증"""
    print_section("✅ 데이터 검증")

    data_yaml_path = DATA_DIR / "data.yaml"

    if not data_yaml_path.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_yaml_path}")
        print("먼저 1_preprocess.py를 실행하세요!")
        return False

    try:
        config = load_yaml(data_yaml_path)

        # 필수 키 확인
        required_keys = ["path", "train", "val", "nc", "names"]
        for key in required_keys:
            if key not in config:
                print(f"❌ 필수 키 누락: {key}")
                return False

        # 경로 확인
        base_path = Path(config["path"])

        # train 경로
        train_path = base_path / config["train"]
        if not train_path.exists():
            print(f"❌ Train 경로가 존재하지 않음: {train_path}")
            return False

        # val 경로
        val_path = base_path / config["val"]
        if not val_path.exists():
            print(f"❌ Val 경로가 존재하지 않음: {val_path}")
            return False

        # 파일 개수 확인
        train_images = len(
            list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
        )
        val_images = len(list(val_path.glob("*.jpg")) + list(val_path.glob("*.png")))

        print(f"✅ 데이터 검증 통과")
        print(f"  클래스: {config['nc']}개")
        print(f"  Train: {train_images}개 이미지")
        print(f"  Val: {val_images}개 이미지")

        return True

    except Exception as e:
        print(f"❌ 데이터 검증 실패: {e}")
        return False


def train_model(resume: bool = False):
    """모델 학습 실행"""
    print_section("🚀 YOLOv11 학습 시작")

    # 데이터 검증
    if not validate_data():
        raise ValueError("데이터 검증 실패!")

    data_yaml_path = DATA_DIR / "data.yaml"

    # 모델 로드
    if resume and (MODELS_DIR / "yolov11_food" / "weights" / "last.pt").exists():
        print("이전 학습에서 재개합니다...")
        model = YOLO(MODELS_DIR / "yolov11_food" / "weights" / "last.pt")
    else:
        print(f"새로운 모델 로드: {TRAIN_CONFIG['model']}")
        model = YOLO(TRAIN_CONFIG["model"])

    # 학습 설정 통합
    train_args = {
        **TRAIN_CONFIG,
        **AUGMENTATION_CONFIG,
        "data": str(data_yaml_path),
        "resume": resume,
    }

    # 학습 실행
    try:
        print("학습 시작...")
        results = model.train(**train_args)

        print_section("🎉 학습 완료!")

        best_model_path = MODELS_DIR / "yolov11_food" / "weights" / "best.pt"
        print(f"최고 성능 모델: {best_model_path}")

        return results

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ GPU 메모리 부족!")
            print("\n💡 해결 방법:")
            print("  1. 배치 크기 감소: batch=8 또는 batch=4")
            print("  2. 이미지 크기 감소: imgsz=512")
            print("  3. 작은 모델 사용: yolo11s-seg.pt")
            print("  4. 캐싱 비활성화: cache=False")
        raise


def train_with_fallback():
    """메모리 부족 시 자동 경량화"""
    try:
        return train_model()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("⚠️ 메모리 부족! 경량 설정으로 재시도...")

            # GPU 메모리 초기화
            torch.cuda.empty_cache()

            # 경량 설정 적용
            TRAIN_CONFIG["model"] = "yolo11s-seg.pt"
            TRAIN_CONFIG["batch"] = 4
            TRAIN_CONFIG["imgsz"] = 512
            TRAIN_CONFIG["cache"] = False
            TRAIN_CONFIG["workers"] = 2
            TRAIN_CONFIG["name"] = "yolov11_food_lite"

            return train_model()
        else:
            raise


def main():
    """학습 메인 함수"""
    logger = setup_logging()

    try:
        # 환경 확인
        check_environment()

        # 학습 실행
        results = train_with_fallback()

        print("\n" + "=" * 70)
        print("🎉 학습이 완료되었습니다!")
        print("=" * 70)

        return results

    except Exception as e:
        logger.error(f"학습 실패: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
