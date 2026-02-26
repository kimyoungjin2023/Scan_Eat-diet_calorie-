"""
YOLOv11 파인튜닝 V2 - 개선된 전체 모델 미세조정
"""

import torch
from ultralytics import YOLO
from pathlib import Path

from config import *
from utils import *


def analyze_previous_training():
    """이전 학습 결과 분석 및 베이스 모델 선택"""
    print_section("📊 이전 학습 분석")

    possible_models = [
        (
            "Original 30ep",
            MODELS_DIR / "yolov11_food_30ep_clean" / "weights" / "best.pt",
        ),
        ("Finetuned V1", MODELS_DIR / "yolov11_food_finetuned" / "weights" / "best.pt"),
    ]

    available_models = []
    for name, path in possible_models:
        if path.exists():
            available_models.append((name, path))
            print(f"✅ {name}: {path}")

    if not available_models:
        print("❌ 사용 가능한 모델이 없습니다!")
        return None

    # 원본 30 에폭 모델을 베이스로 사용 (더 균형잡힌 시작점)
    for name, path in available_models:
        if "Original" in name:
            print(f"\n🎯 베이스 모델 선택: {name}")
            return path

    # 원본이 없으면 첫 번째 사용
    return available_models[0][1]


def create_optimized_config():
    """RTX 3060 8GB 최적화된 파인튜닝 설정"""
    print_section("⚙️ 최적화된 파인튜닝 설정 V2")

    config = {
        # 기본 설정
        "epochs": 50,  # 적절한 길이
        "imgsz": 512,  # 메모리 안전
        "batch": 8,  # RTX 3060 8GB 안전
        "device": 0,
        # 🔥 핵심 개선사항
        "freeze": None,  # 백본 동결 완전 해제!
        "lr0": 0.0004,  # 학습률 2배 증가 (0.0002→0.0004)
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,  # 정규화 적절히
        "warmup_epochs": 3,
        "patience": 25,  # 조기 종료 완화
        # 메모리 최적화
        "amp": True,  # Mixed Precision 필수
        "cache": "ram",
        "workers": 4,
        # 저장 설정
        "project": str(MODELS_DIR),
        "name": "yolov11_food_finetuned_v2",
        "exist_ok": True,
        "save": True,
        "save_period": 15,
        "plots": True,
        "verbose": True,
    }

    # 균형잡힌 증강 (너무 약하지도 강하지도 않게)
    augmentation = {
        "hsv_h": 0.012,  # 색상 변화 적절히
        "hsv_s": 0.6,  # 채도 변화 적절히
        "hsv_v": 0.35,  # 명도 변화 적절히
        "degrees": 10,  # 회전 적절히
        "translate": 0.08,  # 이동 적절히
        "scale": 0.4,  # 크기 변화 적절히
        "shear": 1.0,  # 전단 변환 약간
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,  # 좌우 반전 유지
        "mosaic": 0.8,  # Mosaic 적절히
        "mixup": 0.08,  # Mixup 적절히
        "copy_paste": 0.05,  # Copy-paste 약간
    }

    print("📋 V2 핵심 개선사항:")
    print(f"  백본 동결:     해제 (전체 모델 학습) ⭐")
    print(f"  학습률:        {config['lr0']} (2배 증가)")
    print(f"  에폭 수:       {config['epochs']}")
    print(f"  증강 강도:     균형잡힌 수준")
    print(f"  메모리 사용:   ~6-7GB (안전)")

    return config, augmentation


def execute_finetune_v2():
    """개선된 파인튜닝 실행"""
    print_section("🚀 파인튜닝 V2 실행")

    # GPU 확인
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ GPU를 사용할 수 없습니다. CPU는 매우 느립니다.")
        response = input("계속하시겠습니까? (y/n): ")
        if response.lower() != "y":
            return None

    # 베이스 모델 선택
    base_model_path = analyze_previous_training()
    if base_model_path is None:
        return None

    print(f"\n모델 로드 중: {base_model_path.name}")
    model = YOLO(base_model_path)

    # 설정 생성
    config, augmentation = create_optimized_config()

    # 데이터 검증
    data_yaml_path = DATA_DIR / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {data_yaml_path}")

    # 학습 실행
    train_args = {
        **config,
        **augmentation,
        "data": str(data_yaml_path),
        "resume": False,
    }

    try:
        print(f"\n🎯 파인튜닝 V2 시작...")
        print(f"예상 소요 시간: 1.5-2시간")
        print(f"예상 메모리 사용: 6-7GB")

        results = model.train(**train_args)

        print_section("🎉 파인튜닝 V2 완료!")

        finetuned_model = MODELS_DIR / config["name"] / "weights" / "best.pt"
        print(f"새 모델: {finetuned_model}")

        return results, base_model_path, finetuned_model

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ GPU 메모리 부족!")
            print("💡 해결 방법:")
            print("  1. config에서 batch=6으로 감소")
            print("  2. imgsz=448로 감소")
            print("  3. cache=False로 설정")
        raise


def compare_all_models():
    """모든 모델 성능 비교"""
    print_section("📊 전체 모델 성능 비교")

    data_yaml_path = DATA_DIR / "data.yaml"

    # 사용 가능한 모든 모델 수집
    models = []
    for model_dir in MODELS_DIR.glob("yolov11_*"):
        best_pt = model_dir / "weights" / "best.pt"
        if best_pt.exists():
            models.append((model_dir.name, best_pt))

    if not models:
        print("❌ 비교할 모델이 없습니다!")
        return

    print(f"발견된 모델: {len(models)}개\n")

    results = []
    for name, model_path in models:
        print(f"평가 중: {name}...")
        try:
            model = YOLO(model_path)
            metrics = model.val(data=str(data_yaml_path), split="val", verbose=False)

            results.append(
                {
                    "name": name,
                    "mAP50": metrics.seg.map50,
                    "mAP50-95": metrics.seg.map,
                    "precision": metrics.seg.mp,
                    "recall": metrics.seg.mr,
                }
            )
        except Exception as e:
            print(f"  ⚠️ 평가 실패: {e}")

    # 결과 정렬 및 출력
    results.sort(key=lambda x: x["mAP50"], reverse=True)

    print(f"\n{'='*85}")
    print(f"🏆 모델 성능 순위")
    print(f"{'='*85}")
    print(
        f"{'순위':<4} {'모델명':<30} {'mAP50':<8} {'mAP50-95':<10} {'Precision':<10} {'Recall':<8}"
    )
    print(f"{'-'*85}")

    for i, result in enumerate(results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        print(
            f"{medal:<4} {result['name']:<30} "
            f"{result['mAP50']:.3f}    "
            f"{result['mAP50-95']:.3f}      "
            f"{result['precision']:.3f}      "
            f"{result['recall']:.3f}"
        )

    print(f"{'='*85}\n")

    # 최고 성능 모델 추천
    if results:
        best = results[0]
        print(f"🏆 최고 성능 모델: {best['name']}")
        print(f"   mAP50: {best['mAP50']:.1%}")

        if best["mAP50"] >= 0.65:
            print(f"   🎯 상업적 수준 달성! VLM 연동 준비 완료 ✅")
        else:
            print(f"   📈 목표(65%)까지 {0.65-best['mAP50']:.3f} 남음")

    return results


def main():
    """메인 함수"""
    logger = setup_logging()

    print(
        """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║      🎯  YOLOv11 파인튜닝 V2 (개선판)  🎯           ║
    ║                                                       ║
    ║    백본 동결 해제로 균형잡힌 성능 향상 달성          ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    )

    try:
        create_directories([MODELS_DIR, RESULTS_DIR])

        # 파인튜닝 V2 실행
        result = execute_finetune_v2()

        if result is None:
            print("파인튜닝이 취소되었습니다.")
            return False

        # 모든 모델 비교
        compare_all_models()

        print_section("✅ 파인튜닝 V2 완료!")
        print("\n🔗 다음 단계:")
        print("  1. python src/3_test.py - 시각화 결과 확인")
        print("  2. 최적 confidence threshold 적용")
        print("  3. VLM 연동 시작")

        return True

    except Exception as e:
        logger.error(f"파인튜닝 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
