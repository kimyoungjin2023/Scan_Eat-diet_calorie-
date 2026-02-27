import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from .config import CFG
from .models import NoisyViTMask2Former
from .utils import FoodDataset, compute_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoisyViTMask2Former(num_classes=20).to(device) # 클래스 수는 데이터에 맞춰 수정
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    
    # 데이터셋 불러오기 로직 (생략 - 기존 parse_samples 호출 필요)
    # train_loader = DataLoader(...)
    
    model.train()
    for epoch in range(CFG["epochs"]):
        for batch in tqdm(train_loader):
            imgs = batch["image"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["masks"].to(device)
            
            cls_logits, pred_masks = model(imgs)
            loss = compute_loss(cls_logits, pred_masks, labels, masks)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()