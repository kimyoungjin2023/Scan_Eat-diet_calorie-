from utils import print_stage
from train import run_train
from finetune import run_finetune

def main():
    # 1단계: 사전 학습
    print_stage("1단계: 사전 학습 시작")
    run_train()

    # 2단계: 파인튜닝 (best.pt 자동 탐색)
    print_stage("2단계: 파인튜닝 시작")
    run_finetune()

    # best.pt 직접 지정할 경우
    # run_finetune("/content/drive/MyDrive/please/best.pt")

if __name__ == "__main__":
    main()