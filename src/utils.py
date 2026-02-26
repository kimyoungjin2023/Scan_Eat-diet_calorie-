"""
공통 유틸리티 함수
"""

import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def create_directories(dirs: List[Path]):
    """디렉토리 생성"""
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def load_yaml(yaml_path: Path) -> Dict:
    """YAML 파일 로드"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, yaml_path: Path):
    """YAML 파일 저장"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def create_backup(source_dir: Path, backup_dir: Path) -> Path:
    """백업 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"backup_{timestamp}"
    shutil.copytree(source_dir, backup_path)
    return backup_path


def print_section(title: str):
    """섹션 구분선 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")
