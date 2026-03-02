from utils import print_stage
from train import run_train
from finetune import run_finetune
from export import export_onnx, export_tensorrt, export_int8

def main():
    # 1단계: 사전 학습
    print_stage("1단계: 사전 학습 시작")
    run_train()

    # 2단계: 파인튜닝 (best.pt 자동 탐색)
    print_stage("2단계: 파인튜닝 시작")
    run_finetune()

    # 3단계: 경량화
    print_stage("3단계: 경량화 시작")
    export_onnx()        # 범용 배포용
    # export_tensorrt()  # NVIDIA GPU 서버 배포용
    # export_int8()      # 최대 경량화
    
    # best.pt 직접 지정할 경우
    # run_finetune("/content/drive/MyDrive/please/best.pt")

if __name__ == "__main__":
    main()