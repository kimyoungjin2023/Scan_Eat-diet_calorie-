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

# 2. 고도화된 데이터셋 클래스 (Rotation 및 Crop 미세 추가)
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

        # 🚨 [강화된 데이터 증강] 
        if self.use_augment:
            # 1. 좌우 반전 (필수)
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                masks = torch.flip(masks, [2])
            
            # 2. 미세한 밝기/대비/색상 변화 (음식 사진의 조명 차이 극복)
            if random.random() > 0.5:
                img_tensor = F.adjust_brightness(img_tensor, random.uniform(0.7, 1.3))
                img_tensor = F.adjust_contrast(img_tensor, random.uniform(0.8, 1.2))
                img_tensor = F.adjust_saturation(img_tensor, random.uniform(0.8, 1.2))

        target_dict = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": torch.tensor([img_id])}
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

# 4. 메인 학습 루프 (3단계 전략 적용)
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 45
    
    train_ds = ScaneatDataset(TRAIN_IMG, TRAIN_JSON, use_augment=True)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = get_model(num_classes)
    
    # 🔒 [1단계] 백본 전체 동결 (0~15 Epoch)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    model.to(device)

    # 초기 Optimizer (높은 학습률로 헤드 길들이기)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
                                lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 스케줄러 (더 촘촘하게 감쇄)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35], gamma=0.2)
    
    print(f"🚀 [BJ-Pro Edition] 1등 탈환을 위한 극한의 파인튜닝 시작!")

    for epoch in range(45): # 더 정밀한 학습을 위해 45에폭으로 확장
        # 🔓 [2단계] 백본 일부 해제 (15 Epoch ~)
        if epoch == 15:
            print("🔓 [Partial Unfreeze] 백본의 상위 레이어를 해제하여 세부 특징을 학습합니다.")
            # ResNet의 상위 블록(layer3, layer4)만 우선 해제
            for name, param in model.backbone.named_parameters():
                if "layer3" in name or "layer4" in name or "fpn" in name:
                    param.requires_grad = True
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
                                        lr=0.001, momentum=0.9, weight_decay=0.0005)

        # 🔓 [3단계] 백본 전체 해제 (30 Epoch ~)
        if epoch == 30:
            print("🔓 [Full Unfreeze] 전 구간 미세 조정을 시작합니다.")
            for param in model.parameters():
                param.requires_grad = True
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

        lr_scheduler.step()
        print(f"📁 Epoch [{epoch+1}/45] 평균 Loss: {epoch_loss/len(train_loader):.4f} | LR: {optimizer.param_groups[0]['lr']:.66f}")
        torch.save(model.state_dict(), "best_maskrcnn_bj_top1.pth")

if __name__ == "__main__":
    main()