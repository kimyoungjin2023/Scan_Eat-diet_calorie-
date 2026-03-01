import os
import torch
import torchvision
import random
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import numpy as np

# 1. 경로 설정
DATASET_ROOT = "C:/scan_eat/data"
TRAIN_JSON = os.path.join(DATASET_ROOT, "train/_annotations.coco_final.json")
TRAIN_IMG = os.path.join(DATASET_ROOT, "train/images")

# 2. 데이터셋 클래스 (수동 증강: Flip + Brightness/Contrast)
class ScaneatDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, use_augment=True):
        super().__init__(img_folder, ann_file)
        self.use_augment = use_augment

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        width, height = img.size
        
        boxes, labels, masks = [], [], []
        for ann in target:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0: continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            
            mask_img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask_img)
            for seg in ann['segmentation']:
                draw.polygon(seg, outline=1, fill=1)
            masks.append(np.array(mask_img))

        img_tensor = F.to_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        # 🚨 [데이터 증강] 훈련 시에만 확률적으로 적용
        if self.use_augment:
            # 1. 좌우 반전 (50% 확률)
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                masks = torch.flip(masks, [2])
            
            # 2. 밝기/대비 랜덤 조절 (조명 환경 대응)
            if random.random() > 0.5:
                img_tensor = F.adjust_brightness(img_tensor, random.uniform(0.8, 1.2))
                img_tensor = F.adjust_contrast(img_tensor, random.uniform(0.8, 1.2))

        target_dict = {
            "boxes": boxes, 
            "labels": labels, 
            "masks": masks, 
            "image_id": torch.tensor([img_id])
        }
        return img_tensor, target_dict

# 3. 모델 생성 함수
def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# 4. 메인 학습 루프
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 45
    
    # 데이터로더 설정
    train_ds = ScaneatDataset(TRAIN_IMG, TRAIN_JSON, use_augment=True)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = get_model(num_classes)
    
    # 🔒 [필살기 1] 초반 백본 동결 (베테랑의 지식 보존)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    model.to(device)

    # 학습 가능한 파라미터(헤드 부분)만 옵티마이저에 전달
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 📉 [필살기 2] 학습률 스케줄러 (15, 25에폭에서 보폭 줄이기)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    
    print(f"🚀 [BJ-Edition] 최종 파인튜닝 시작! (Device: {device})")

    for epoch in range(35):
        # 🔓 [필살기 3] 20에폭부터 동결 해제 (전체 미세 조정 시작)
        if epoch == 20:
            print("🔓 [동결 해제] 이제 백본까지 함께 학습합니다!")
            for param in model.parameters():
                param.requires_grad = True
            # 전체 파라미터를 대상으로 옵티마이저 갱신 (학습률은 낮게 유지)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)

        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            if i % 50 == 0:
                print(f"   🔹 Epoch [{epoch+1}] Step [{i}/{len(train_loader)}] - Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📁 Epoch [{epoch+1}/35] 평균 Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # 모델 저장
        torch.save(model.state_dict(), "best_maskrcnn_bj_final.pth")

if __name__ == "__main__":
    main()