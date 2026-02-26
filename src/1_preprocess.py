"""
데이터 전처리: 세그먼테이션 규격 검증 + 한글 매핑 + 문제 파일 자동 처리
"""

import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import json
from datetime import datetime

from config import *
from utils import *


def detect_dataset_structure():
    """데이터셋 구조 자동 감지"""
    print_section("🔍 데이터셋 구조 감지")

    val_folder_name = None
    possible_val_names = ["valid", "val", "vailad", "validation"]

    for name in possible_val_names:
        if (DATA_DIR / name).exists():
            val_folder_name = name
            break

    if not val_folder_name:
        raise ValueError("검증 폴더를 찾을 수 없습니다 (valid, val, vailad 등)")

    print(f"✅ 검증 폴더: {val_folder_name}")

    splits = ["train", val_folder_name, "test"]

    for split in splits:
        split_path = DATA_DIR / split
        if not split_path.exists():
            raise ValueError(f"❌ {split} 폴더가 없습니다: {split_path}")

        images_path = split_path / "images"
        labels_path = split_path / "labels"

        if not images_path.exists() or not labels_path.exists():
            raise ValueError(f"❌ {split}에 images 또는 labels 폴더가 없습니다")

        num_images = len(
            list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        )
        num_labels = len(list(labels_path.glob("*.txt")))

        print(f"  {split:10s}: 이미지 {num_images:3d}개, 라벨 {num_labels:3d}개")

    return val_folder_name, splits


def validate_label_line(parts, line_num):
    """
    라벨 라인 하나를 검증하고 수정 시도

    Returns:
        tuple: (is_valid, fixed_parts, error_message)
    """
    if len(parts) < 7:  # class_id + 최소 6개 좌표
        return (
            False,
            None,
            f"라인 {line_num}: 값이 부족 ({len(parts)}개, 최소 7개 필요)",
        )

    try:
        # 클래스 ID 검증
        class_id = int(parts[0])
        if class_id < 0:
            return False, None, f"라인 {line_num}: 클래스 ID가 음수 ({class_id})"

        # 좌표 검증 및 수정
        coords = [float(x) for x in parts[1:]]

        # 좌표 개수 검증 (짝수여야 함)
        if len(coords) % 2 != 0:
            # 홀수면 마지막 좌표 제거
            coords = coords[:-1]
            if len(coords) < 6:  # 수정 후에도 부족하면
                return (
                    False,
                    None,
                    f"라인 {line_num}: 점이 부족 (수정 후 {len(coords)//2}개)",
                )

        # 좌표 범위 검증 및 클리핑
        fixed_coords = []
        out_of_range_count = 0

        for coord in coords:
            if coord < 0.0 or coord > 1.0:
                # 약간 벗어난 경우 클리핑 (허용 오차: ±0.1)
                if -0.1 <= coord <= 1.1:
                    coord = max(0.0, min(1.0, coord))
                    out_of_range_count += 1
                else:
                    return False, None, f"라인 {line_num}: 좌표 범위 초과 ({coord})"

            fixed_coords.append(coord)

        # 수정된 라인 생성
        fixed_parts = [str(class_id)] + [f"{x:.6f}" for x in fixed_coords]

        return True, fixed_parts, None

    except ValueError as e:
        return False, None, f"라인 {line_num}: 데이터 형식 오류 ({str(e)})"


def scan_and_fix_labels(val_folder_name):
    """모든 라벨 파일 스캔 및 자동 수정/제거"""
    print_section("🔍 라벨 파일 정밀 검증 및 수정")

    splits = ["train", val_folder_name, "test"]

    stats = {
        "total_files": 0,
        "valid_files": 0,
        "fixed_files": 0,
        "removed_files": 0,
        "fixed_lines": 0,
        "removed_lines": 0,
    }

    problem_report = {
        "removed_files": [],
        "fixed_files": [],
        "error_summary": Counter(),
    }

    for split in splits:
        labels_path = DATA_DIR / split / "labels"
        images_path = DATA_DIR / split / "images"
        label_files = list(labels_path.glob("*.txt"))

        print(f"\n{split} 검증 중 ({len(label_files)}개 파일)...")

        for label_file in tqdm(label_files, desc=f"{split:10s}"):
            stats["total_files"] += 1

            try:
                # 파일 읽기
                with open(label_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # 빈 파일 확인
                if not lines or not any(line.strip() for line in lines):
                    # 빈 파일 제거
                    label_file.unlink()

                    # 대응 이미지 제거
                    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                        img_file = images_path / (label_file.stem + ext)
                        if img_file.exists():
                            img_file.unlink()
                            break

                    stats["removed_files"] += 1
                    problem_report["removed_files"].append(
                        {"file": str(label_file), "reason": "빈 파일"}
                    )
                    continue

                # 라인별 검증 및 수정
                valid_lines = []
                file_errors = []
                file_modified = False

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    is_valid, fixed_parts, error_msg = validate_label_line(
                        parts, line_num
                    )

                    if is_valid:
                        if fixed_parts != parts:
                            # 수정됨
                            file_modified = True
                            stats["fixed_lines"] += 1

                        valid_lines.append(" ".join(fixed_parts) + "\n")
                    else:
                        # 수정 불가능한 라인
                        stats["removed_lines"] += 1
                        file_errors.append(error_msg)
                        problem_report["error_summary"][
                            error_msg.split(":")[1].strip().split("(")[0]
                        ] += 1

                # 파일 처리 결과
                if not valid_lines:
                    # 유효한 라인이 하나도 없음 -> 파일 제거
                    label_file.unlink()

                    # 대응 이미지 제거
                    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                        img_file = images_path / (label_file.stem + ext)
                        if img_file.exists():
                            img_file.unlink()
                            break

                    stats["removed_files"] += 1
                    problem_report["removed_files"].append(
                        {
                            "file": str(label_file),
                            "reason": "유효한 라인 없음",
                            "errors": file_errors[:3],  # 상위 3개 오류만
                        }
                    )

                elif file_modified:
                    # 수정된 내용으로 저장
                    with open(label_file, "w", encoding="utf-8") as f:
                        f.writelines(valid_lines)

                    stats["fixed_files"] += 1
                    problem_report["fixed_files"].append(
                        {
                            "file": str(label_file),
                            "original_lines": len(lines),
                            "final_lines": len(valid_lines),
                            "errors_fixed": file_errors[:3],
                        }
                    )
                else:
                    # 문제없는 파일
                    stats["valid_files"] += 1

            except Exception as e:
                print(f"⚠️ 파일 처리 오류 {label_file.name}: {e}")
                continue

    # 결과 출력
    print(f"\n{'='*70}")
    print(f"📊 라벨 검증 및 수정 결과")
    print(f"{'='*70}")
    print(f"총 파일:        {stats['total_files']:4d}개")
    print(
        f"정상 파일:      {stats['valid_files']:4d}개 ({stats['valid_files']/stats['total_files']*100:.1f}%)"
    )
    print(f"수정된 파일:    {stats['fixed_files']:4d}개")
    print(f"제거된 파일:    {stats['removed_files']:4d}개")
    print(f"수정된 라인:    {stats['fixed_lines']:4d}개")
    print(f"제거된 라인:    {stats['removed_lines']:4d}개")

    # 오류 유형별 통계
    if problem_report["error_summary"]:
        print(f"\n주요 문제 유형:")
        for error_type, count in problem_report["error_summary"].most_common(5):
            print(f"  • {error_type:30s}: {count:3d}회")

    # 리포트 저장
    report_path = PROJECT_ROOT / "label_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        # Counter를 dict로 변환
        problem_report["error_summary"] = dict(problem_report["error_summary"])
        json.dump(problem_report, f, ensure_ascii=False, indent=2)

    print(f"\n상세 리포트: {report_path}")

    return stats


def analyze_original_labels():
    """원본 라벨 분석"""
    print_section("📊 원본 라벨 분석")

    data_yaml_path = DATA_DIR / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {data_yaml_path}")

    original_config = load_yaml(data_yaml_path)
    original_names = original_config["names"]

    print(f"원본 클래스 수: {len(original_names)}개")

    # 클래스 사용 빈도 분석
    class_counts = Counter()

    val_folder = None
    for name in ["valid", "val", "vailad", "validation"]:
        if (DATA_DIR / name).exists():
            val_folder = name
            break

    splits = ["train", val_folder, "test"]

    for split in splits:
        if split is None:
            continue
        labels_path = DATA_DIR / split / "labels"
        for label_file in labels_path.glob("*.txt"):
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
            except:
                continue

    print(f"\n클래스 사용 빈도 (상위 10개):")
    for class_id, count in class_counts.most_common(10):
        if class_id < len(original_names):
            print(f"  {class_id:2d}. {original_names[class_id]:30s}: {count:4d}회")

    return original_names, class_counts


def create_korean_mapping(original_names):
    """한글 매핑 및 ID 변환 테이블 생성"""
    print_section("🔄 한글 매핑 생성")

    unique_korean_names = sorted(list(set(KOREAN_FOOD_MAPPING.values())))
    korean_to_new_id = {name: idx for idx, name in enumerate(unique_korean_names)}

    old_to_new_mapping = {}
    unmapped_classes = []

    print(f"기존 클래스: {len(original_names)}개")
    print(f"새 클래스: {len(unique_korean_names)}개")
    print(f"중복 제거: {len(original_names) - len(unique_korean_names)}개\n")

    print("변환 매핑 (변경된 것만):")
    for old_id, old_name in enumerate(original_names):
        if old_name in KOREAN_FOOD_MAPPING:
            korean_name = KOREAN_FOOD_MAPPING[old_name]
            new_id = korean_to_new_id[korean_name]
            old_to_new_mapping[old_id] = new_id

            if old_id != new_id:
                print(f"  {old_id:2d} ({old_name:30s}) → {new_id:2d} ({korean_name})")
        else:
            unmapped_classes.append((old_id, old_name))

    if unmapped_classes:
        print(f"\n⚠️ 매핑되지 않은 클래스 ({len(unmapped_classes)}개):")
        for old_id, old_name in unmapped_classes:
            print(f"  {old_id:2d}: {old_name}")

    return unique_korean_names, old_to_new_mapping


def backup_dataset():
    """데이터셋 백업 - 함수명 변경으로 충돌 해결"""
    print_section("💾 백업 생성")

    # utils.py의 create_backup 함수 호출
    backup_path = create_backup(DATA_DIR, BACKUP_DIR)
    print(f"✅ 백업 완료: {backup_path}")
    return backup_path


def convert_labels_inplace(old_to_new_mapping, val_folder_name):
    """라벨 파일을 제자리에서 변환"""
    print_section("📝 라벨 파일 변환 (In-place)")

    splits = ["train", val_folder_name, "test"]

    total_converted = 0
    total_files = 0

    for split in splits:
        labels_path = DATA_DIR / split / "labels"
        label_files = list(labels_path.glob("*.txt"))

        print(f"\n{split} 변환 중 ({len(label_files)}개 파일)...")

        split_converted = 0

        for label_file in tqdm(label_files, desc=f"{split:10s}"):
            total_files += 1

            try:
                with open(label_file, "r") as f:
                    lines = f.readlines()

                new_lines = []
                file_modified = False

                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    try:
                        old_class_id = int(parts[0])
                    except ValueError:
                        continue

                    if old_class_id in old_to_new_mapping:
                        new_class_id = old_to_new_mapping[old_class_id]
                        new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                        new_lines.append(new_line)

                        if old_class_id != new_class_id:
                            file_modified = True

                if new_lines:
                    with open(label_file, "w") as f:
                        f.writelines(new_lines)

                    if file_modified:
                        split_converted += 1
                        total_converted += 1

            except Exception as e:
                continue

        print(f"  {split}: {split_converted}개 파일 수정됨")

    print(f"\n✅ 라벨 변환 완료!")
    print(f"  총 파일: {total_files}개")
    print(f"  수정됨: {total_converted}개")

    return total_converted


def update_data_yaml(korean_names, val_folder_name):
    """data.yaml 업데이트"""
    print_section("📄 data.yaml 업데이트")

    original_yaml = DATA_DIR / "data.yaml"
    backup_yaml = DATA_DIR / "data_original.yaml"

    if not backup_yaml.exists():
        shutil.copy2(original_yaml, backup_yaml)
        print(f"원본 백업: {backup_yaml}")

    new_config = {
        "path": str(DATA_DIR.absolute()),
        "train": "train/images",
        "val": f"{val_folder_name}/images",
        "test": "test/images",
        "nc": len(korean_names),
        "names": korean_names,
    }

    save_yaml(new_config, original_yaml)

    print(f"✅ data.yaml 업데이트 완료!")
    print(f"  클래스 수: {len(korean_names)}개")


def verify_conversion(korean_names):
    """변환 결과 검증"""
    print_section("✅ 변환 결과 검증")

    print(f"새 클래스 목록 ({len(korean_names)}개):")
    for i, name in enumerate(korean_names):
        print(f"  {i:2d}: {name}")

    # 샘플 라벨 확인
    test_labels = DATA_DIR / "test" / "labels"
    if test_labels.exists():
        sample_files = list(test_labels.glob("*.txt"))[:3]

        if sample_files:
            print(f"\n샘플 라벨 확인 (test 세트):")
            for label_file in sample_files:
                print(f"\n  파일: {label_file.name}")
                try:
                    with open(label_file, "r") as f:
                        lines = f.readlines()[:3]
                        for i, line in enumerate(lines):
                            parts = line.strip().split()
                            if parts:
                                try:
                                    class_id = int(parts[0])
                                    if class_id < len(korean_names):
                                        class_name = korean_names[class_id]
                                        print(
                                            f"    라인 {i+1}: ID {class_id} = '{class_name}'"
                                        )
                                    else:
                                        print(f"    ⚠️ 라인 {i+1}: 잘못된 ID {class_id}")
                                except ValueError:
                                    print(f"    ⚠️ 라인 {i+1}: 잘못된 형식")
                except Exception as e:
                    print(f"    오류: {e}")

    print("\n✅ 검증 완료!")


def main():
    """전체 전처리 파이프라인"""
    logger = setup_logging()

    try:
        # 필수 디렉토리 생성
        create_directories([BACKUP_DIR, MODELS_DIR, RESULTS_DIR])

        # 1. 데이터셋 구조 감지
        val_folder_name, splits = detect_dataset_structure()

        # 2. 백업 생성 (함수명 변경됨!)
        backup_path = backup_dataset()

        # 3. 라벨 규격 검증 및 수정
        stats = scan_and_fix_labels(val_folder_name)

        # 4. 원본 라벨 분석
        original_names, class_counts = analyze_original_labels()

        # 5. 한글 매핑 생성
        korean_names, id_mapping = create_korean_mapping(original_names)

        # 6. 라벨 변환 (in-place)
        converted_count = convert_labels_inplace(id_mapping, val_folder_name)

        # 7. data.yaml 업데이트
        update_data_yaml(korean_names, val_folder_name)

        # 8. 검증
        verify_conversion(korean_names)

        print_section("🎉 데이터 전처리 완료!")
        print(f"백업 위치: {backup_path}")
        print(
            f"라벨 수정: 파일 {stats['fixed_files']}개, 라인 {stats['fixed_lines']}개"
        )
        print(
            f"라벨 제거: 파일 {stats['removed_files']}개, 라인 {stats['removed_lines']}개"
        )
        print(f"ID 변환: {converted_count}개 파일")
        print(f"최종 클래스: {len(korean_names)}개")

        return True

    except Exception as e:
        logger.error(f"데이터 전처리 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
