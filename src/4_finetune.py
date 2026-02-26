"""
YOLOv11 파인튜닝 - 백본 동결 + 헤드 집중 학습
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime

from config import *
from utils import *


def analyze_training_progress():
    """이전 학습 결과 분석"""
    print_section("📊 이전 학습 분석")

    # 가능한 모델 경로들 확인
    possible_models = [
        MODELS_DIR / "yolov11_food_30ep_clean" / "weights" / "best.pt",
        MODELS_DIR / "yolov11_food" / "weights" / "best.pt",
        MODELS_DIR / "yolov11_food_30ep" / "weights" / "best.pt",
    ]

    base_model = None
    for model_path in possible_models:
        if model_path.exists():
            base_model = model_path
            break

    if base_model is None:
        print("❌ 이전 학습 모델을 찾을 수 없습니다!")
        print("사용 가능한 모델:")
        for model_dir in MODELS_DIR.glob("yolov11_*"):
            if (model_dir / "weights" / "best.pt").exists():
                print(f"  ✓ {model_dir.name}")
        return None

    print(f"✅ 베이스 모델: {base_model}")

    # results.csv 분석
    results_csv = base_model.parent.parent / "results.csv"
    if results_csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(results_csv)

            if len(df) > 0:
                last_row = df.iloc[-1]
                print(f"\n📈 최종 성능 (에폭 {int(last_row['epoch'])}):")
                print(f"  mAP50(Box):  {last_row['metrics/mAP50(B)']:.3f}")
                print(f"  mAP50(Mask): {last_row['metrics/mAP50(M)']:.3f}")
                print(f"  Precision:   {last_row['metrics/precision(B)']:.3f}")
                print(f"  Recall:      {last_row['metrics/recall(B)']:.3f}")

                # 학습 추세 분석
                if len(df) >= 10:
                    recent_trend = (
                        df["metrics/mAP50(M)"].iloc[-1]
                        - df["metrics/mAP50(M)"].iloc[-10]
                    )
                    print(f"\n📊 최근 10 에폭 개선: +{recent_trend:.3f}")

                    if recent_trend > 0.02:
                        print("  → 🚀 여전히 상승 중! 파인튜닝 강력 권장")
                    else:
                        print("  → ⚖️ 안정화 상태. 파인튜닝으로 돌파 가능")
        except Exception as e:
            print(f"⚠️ CSV 분석 실패: {e}")

    return base_model


def create_finetune_config():
    """파인튜닝 최적화 설정"""
    print_section("⚙️ 파인튜닝 설정")

    config = {
        # 기본 설정
        "epochs": 50,  # 추가 50 에폭
        "imgsz": 512,  # 이미지 크기 유지
        "batch": 8,  # RTX 3060 8GB 안전
        "device": 0,
        # 파인튜닝 최적화
        "lr0": 0.0002,  # 낮은 학습률 (1/5)
        "lrf": 0.01,  # 최종 학습률
        "momentum": 0.937,
        "weight_decay": 0.001,  # 정규화 강화
        "warmup_epochs": 3,  # 짧은 워밍업
        "patience": 30,  # 조기 종료
        # 🔥 핵심: 백본 동결
        "freeze": 10,  # 처음 10개 레이어 동결
        # 메모리 최적화
        "amp": True,
        "cache": "ram",
        "workers": 4,
        # 저장 설정
        "project": str(MODELS_DIR),
        "name": "yolov11_food_finetuned",
        "exist_ok": True,
        "save": True,
        "save_period": 15,
        "plots": True,
        "verbose": True,
    }

    # 파인튜닝용 약한 증강
    augmentation = {
        "hsv_h": 0.01,  # 색상 변화 최소화
        "hsv_s": 0.4,  # 채도 변화 감소
        "hsv_v": 0.3,  # 명도 변화 감소
        "degrees": 8,  # 회전 감소 (15→8)
        "translate": 0.05,  # 이동 감소 (0.1→0.05)
        "scale": 0.3,  # 크기 변화 감소 (0.5→0.3)
        "shear": 0.0,  # 전단 변환 비활성화
        "perspective": 0.0,  # 원근 변환 비활성화
        "flipud": 0.0,  # 상하 반전 없음
        "fliplr": 0.5,  # 좌우 반전 유지
        "mosaic": 0.5,  # Mosaic 감소 (1.0→0.5)
        "mixup": 0.05,  # Mixup 감소 (0.1→0.05)
        "copy_paste": 0.0,  # Copy-paste 비활성화
    }

    print("📋 파인튜닝 핵심 설정:")
    print(f"  추가 에폭:     {config['epochs']}")
    print(f"  학습률:        {config['lr0']} (기존 대비 1/5)")
    print(f"  백본 동결:     {config['freeze']}개 레이어")
    print(f"  증강 강도:     약화 (미세조정 모드)")

    return config, augmentation


def finetune_model():
    """파인튜닝 실행"""
    print_section("🚀 파인튜닝 시작")

    # GPU 확인
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ GPU를 사용할 수 없습니다.")
        response = input("CPU로 계속하시겠습니까? (매우 느림, y/n): ")
        if response.lower() != "y":
            return None

    # 이전 모델 로드
    base_model = analyze_training_progress()
    if base_model is None:
        return None

    print(f"\n이전 모델 로드: {base_model.name}")
    model = YOLO(base_model)

    # 설정 생성
    config, augmentation = create_finetune_config()

    # 데이터 검증
    data_yaml_path = DATA_DIR / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {data_yaml_path}")

    # 학습 인자 통합
    train_args = {
        **config,
        **augmentation,
        "data": str(data_yaml_path),
        "resume": False,  # 새로운 학습으로 시작
    }

    # 파인튜닝 실행
    try:
        print(f"\n🎯 파인튜닝 시작... (Device: {device})")
        print(f"예상 소요 시간: 1-2시간")

        results = model.train(**train_args)

        print_section("🎉 파인튜닝 완료!")

        finetuned_model = MODELS_DIR / config["name"] / "weights" / "best.pt"
        print(f"파인튜닝 모델: {finetuned_model}")

        return results, base_model, finetuned_model

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ GPU 메모리 부족!")
            print("💡 해결 방법: config에서 batch=4로 감소 후 재시도")
        raise


def compare_performance(base_model, finetuned_model):
    """성능 비교"""
    print_section("📊 성능 비교 분석")

    data_yaml_path = DATA_DIR / "data.yaml"

    print("원본 모델 평가 중...")
    base = YOLO(base_model)
    base_metrics = base.val(data=str(data_yaml_path), split="val", verbose=False)

    print("파인튜닝 모델 평가 중...")
    finetuned = YOLO(finetuned_model)
    fine_metrics = finetuned.val(data=str(data_yaml_path), split="val", verbose=False)

    # 성능 비교 출력
    print(f"\n{'='*70}")
    print(f"🏆 파인튜닝 성능 개선도")
    print(f"{'='*70}")

    comparisons = [
        ("mAP50(Mask)", base_metrics.seg.map50, fine_metrics.seg.map50),
        ("mAP50-95(Mask)", base_metrics.seg.map, fine_metrics.seg.map),
        ("Precision", base_metrics.seg.mp, fine_metrics.seg.mr),
        ("Recall", base_metrics.seg.mr, fine_metrics.seg.mr),
    ]

    for name, base_val, fine_val in comparisons:
        if base_val > 0:
            improvement = fine_val - base_val
            percent = improvement / base_val * 100
            symbol = "📈" if improvement > 0 else "📉"

            print(
                f"{symbol} {name:15s}: {base_val:.3f} → {fine_val:.3f} ({percent:+.1f}%)"
            )
        else:
            print(f"⚠️ {name:15s}: 측정 불가")

    # 목표 달성도 평가
    target_map50 = 0.70  # 70% 목표
    if fine_metrics.seg.map50 >= target_map50:
        print(
            f"\n🎯 목표 달성! mAP50 {fine_metrics.seg.map50:.1%} ≥ {target_map50:.0%}"
        )
        print("  → 상업적 수준의 성능 달성 ✅")
    else:
        remaining = target_map50 - fine_metrics.seg.map50
        print(f"\n📊 목표까지 {remaining:.3f} 남음 ({remaining/target_map50*100:.1f}%)")
        print("  → 추가 파인튜닝 또는 데이터 보강 고려")

    print(f"{'='*70}\n")

    return fine_metrics


def main():
    """파인튜닝 메인 함수"""
    logger = setup_logging()

    print(
        """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║          🎯  YOLOv11 파인튜닝 시스템  🎯             ║
    ║                                                       ║
    ║        백본 동결로 효율적인 성능 향상 달성           ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    )

    try:
        # 필수 디렉토리 생성
        create_directories([MODELS_DIR, RESULTS_DIR])

        # 파인튜닝 실행
        result = finetune_model()

        if result is None:
            print("파인튜닝이 취소되었습니다.")
            return False

        results, base_model, finetuned_model = result

        # 성능 비교
        final_metrics = compare_performance(base_model, finetuned_model)

        print_section("✅ 파인튜닝 프로세스 완료!")
        print(f"원본 모델:       {base_model}")
        print(f"파인튜닝 모델:   {finetuned_model}")
        print(f"최종 mAP50:      {final_metrics.seg.map50:.1%}")

        print(f"\n🔗 다음 단계:")
        print(f"  1. python src/3_test.py - 시각화 결과 확인")
        print(f"  2. VLM 연동 - 영양소 분석 시스템 구축")

        return True

    except Exception as e:
        logger.error(f"파인튜닝 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
