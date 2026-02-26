import os
from pathlib import Path

def get_best_pt_path(project_dir: str, train_name: str) -> str:
    """1차 학습 best.pt 경로 자동 탐색"""
    path = Path(project_dir) / train_name / "weights" / "best.pt"
    
    if not path.exists():
        raise FileNotFoundError(f"best.pt를 찾을 수 없습니다: {path}")
    
    print(f"best.pt 경로 확인: {path}")
    return str(path)

def print_stage(stage: str):
    """단계 출력"""
    print("\n" + "=" * 40)
    print(f"  {stage}")
    print("=" * 40 + "\n")