import pickle as pkl
import sys
import torch

if __name__ == "__main__":
    input_path = sys.argv[1]
    # 모델 파일을 로드합니다.
    obj = torch.load(input_path, map_location="cpu")["model"]
    # Detectron2 형식으로 구조를 맞춥니다.
    res = {"model": obj, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    print(f"✅ 변환 완료: {sys.argv[2]}")