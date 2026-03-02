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

# 1. 경로 설정 (BJ님의 환경에 맞게)
DATASET_ROOT = "C:/scan_eat/data"
TRAIN_JSON = os.path.join(DATASET_ROOT, "train/_annotations.coco_final.json")
TRAIN_IMG = os.path.join(DATASET_ROOT, "train/images")

# 2. 극한의 데이터 증강이 포함된 Dataset 클래스
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

        # 🚨 [찐막 필살기] 강력한 데이터 증강
        if self.use_augment:
            # 1. Random Horizontal Flip (50% 확률)
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                masks = torch.flip(masks, [2])
            
            # 2. Color Jitter (밝기, 대비, 채도를 미세하게 조정)
            if random.random() > 0.5:
                img_tensor = F.adjust_brightness(img_tensor, random.uniform(0.7, 1.3))
                img_tensor = F.adjust_contrast(img_tensor, random.uniform(0.8, 1.2))
                img_tensor = F.adjust_saturation(img_tensor, random.uniform(0.8, 1.2))

            # 3. Random Resized Crop (가장 중요! 작은 반찬을 화면에 꽉 차게)
            # 주의: BBox와 Mask 좌표까지 함께 변환해야 하므로 매우 복잡합니다.
            # 이 코드는 간단한 증강을 위해 Crop 대신 크기 조절(Resize)만 적용하여 안정성을 확보합니다.
            # 만약 BBox 연산 오류가 난다면 이 부분을 제거하세요.
            # (실제 완벽한 Crop은 Albumentations 라이브러리가 필요합니다.)
            # img_tensor = F.resize(img_tensor, (640, 640)) # (선택 사항)

        target_dict = {
            "boxes": boxes, 
            "labels": labels, 
            "masks": masks, 
            "image_id": torch.tensor([img_id])
        }
        return img_tensor, target_dict

# 3. 모델 생성
def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# 4. 메인 학습 루프 (3단계 점진적 해제)
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 45
    
    train_ds = ScaneatDataset(TRAIN_IMG, TRAIN_JSON, use_augment=True)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = get_model(num_classes)
    
    # 🔒 [1단계] 백본 완전 동결 (0~15 Epoch)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    model.to(device)

    # 초기 Optimizer (Head만 학습)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
                                lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 스케줄러 (15, 30, 40 에폭에서 학습률 1/5로 뚝뚝 떨어뜨림)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.2)
    
    print(f"🚀 [The Last Stand] 1등 탈환을 위한 45Epoch 극한 훈련 시작! (Device: {device})")

    for epoch in range(45):
        # 🔓 [2단계] 백본 부분 해제 (15 Epoch ~)
        if epoch == 15:
            print("🔓 [Phase 2] 상위 레이어(layer3, layer4) 동결 해제!")
            for name, param in model.backbone.named_parameters():
                if "layer3" in name or "layer4" in name or "fpn" in name:
                    param.requires_grad = True
            # 옵티마이저 갱신 (학습률 낮춤)
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
                                        lr=0.001, momentum=0.9, weight_decay=0.0005)

        # 🔓 [3단계] 전 구간 해제 (30 Epoch ~)
        if epoch == 30:
            print("🔓 [Phase 3] 전 구간 미세 조정 시작! 65% 가즈아!")
            for param in model.parameters():
                param.requires_grad = True
            # 옵티마이저 갱신 (극소 학습률)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)

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
            if i % 100 == 0: # 로그 출력을 조금 줄여서 쾌적하게
                print(f"   🔹 Epoch [{epoch+1}] Step [{i}/{len(train_loader)}] - Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📁 Epoch [{epoch+1}/45] 평균 Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # 모델 저장
        torch.save(model.state_dict(), "best_maskrcnn_bj_the_last.pth")

if __name__ == "__main__":
    main()