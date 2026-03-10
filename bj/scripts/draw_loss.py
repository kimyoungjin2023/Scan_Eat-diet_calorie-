import json
import matplotlib.pyplot as plt

# 1. Detectron2가 남긴 학습 일기장(metrics.json) 읽기
log_file = r"C:\scan_eat\output\metrics.json"

iterations = []
total_losses = []

print("📊 학습 기록을 불러오는 중...")
try:
    with open(log_file, "r") as f:
        for line in f:
            data = json.loads(line)
            # 이터레이션과 total_loss 값이 있는 줄만 추출
            if "iteration" in data and "total_loss" in data:
                iterations.append(data["iteration"])
                total_losses.append(data["total_loss"])
                
    # 2. 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, total_losses, color='blue', linewidth=2, label='Total Loss')
    
    # 3. 그래프 꾸미기
    plt.title("Mask2Former Training Loss Curve", fontsize=16, fontweight='bold')
    plt.xlabel("Iteration (Total 10,000)", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 4. 이미지 파일로 저장!
    save_path = r"C:\scan_eat\loss_graph.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 그래프가 저장되었습니다! 폴더를 확인하세요: {save_path}")
    
except FileNotFoundError:
    print("❌ metrics.json 파일을 찾을 수 없습니다. 경로를 확인하세요.")