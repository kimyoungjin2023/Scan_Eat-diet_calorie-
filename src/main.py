"""
전체 파이프라인 실행
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from utils import *


def run_preprocessing():
    """데이터 전처리 실행"""
    print_section("📦 데이터 전처리")

    import importlib.util

    spec = importlib.util.spec_from_file_location("preprocess", "1_preprocess.py")
    preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess)

    return preprocess.main()


def run_training():
    """모델 학습 실행"""
    print_section("🚀 모델 학습")

    import importlib.util

    spec = importlib.util.spec_from_file_location("train", "2_train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    return train.main()


def run_testing():
    """모델 테스트 실행"""
    print_section("🧪 모델 테스트")

    import importlib.util

    spec = importlib.util.spec_from_file_location("test", "3_test.py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)

    return test.main()


def run_full_pipeline():
    """전체 파이프라인 실행"""
    print_section("🎯 전체 파이프라인 실행")

    logger = setup_logging()

    try:
        # 필수 디렉토리 생성
        create_directories([BACKUP_DIR, MODELS_DIR, RESULTS_DIR])

        # 1. 데이터 전처리
        print("1/3 단계: 데이터 전처리...")
        run_preprocessing()

        # 2. 모델 학습
        print("2/3 단계: 모델 학습...")
        run_training()

        # 3. 모델 테스트
        print("3/3 단계: 모델 테스트...")
        run_testing()

        print_section("🎉 전체 파이프라인 완료!")
        print("\n📁 결과 확인:")
        print(f"  - 원본 백업: backup/")
        print(f"  - 모델: {MODELS_DIR}/yolov11_food/weights/best.pt")
        print(f"  - 시각화: {RESULTS_DIR}/visualization_results.png")
        print(f"  - 예측 결과: {RESULTS_DIR}/test_predictions/")

        return True

    except Exception as e:
        logger.error(f"파이프라인 실행 실패: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Food Nutrition AI Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "train", "test", "full"],
        default="full",
        help="실행 모드 선택",
    )

    args = parser.parse_args()

    # 로고 출력
    print(
        """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║          🍽️  Food Nutrition AI Pipeline  🍽️          ║
    ║                                                       ║
    ║        YOLOv11 기반 음식 세그먼테이션 시스템          ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    )

    # 모드별 실행
    if args.mode == "preprocess":
        run_preprocessing()
    elif args.mode == "train":
        run_training()
    elif args.mode == "test":
        run_testing()
    elif args.mode == "full":
        run_full_pipeline()


if __name__ == "__main__":
    main()
