import os
import torch
import torchvision
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

# 2. 데이터셋 클래스 (PIL 직접 그리기 방식)
class ScaneatDataset(CocoDetection):
    def __init__(self, img_folder, ann_file):
        super().__init__(img_folder, ann_file)

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
        if len(boxes) == 0:
            target_dict = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, height, width), dtype=torch.uint8),
                "image_id": torch.tensor([img_id])
            }
        else:
            target_dict = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id])
            }
        return img_tensor, target_dict

# 3. 모델 생성 함수
def get_model(num_classes):
    # COCO로 사전 학습된 베테랑 가중치 로드
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
    
    # 모델 초기화
    model = get_model(num_classes)

    # ==========================================================
    # 🚨 [파인튜닝 전략 선택 구간] 
    # ==========================================================
    
    # 방식 A: 백본 동결 (현재 활성화 - 헤드만 학습하여 과적합 방지)
    print("🔒 [전략] 백본 동결 모드: 헤드(분류/마스크) 부분만 집중 학습합니다.")
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # 방식 B: 전체 학습 (현재 주석 처리 - 나중에 성능 더 올릴 때 사용)
    # print("🔓 [전략] 전체 학습 모드: 모든 파라미터를 미세 조정합니다.")
    # for param in model.parameters():
    #     param.requires_grad = True

    # ==========================================================
    
    model.to(device)

    # 최적화 대상 설정 (requires_grad=True인 파라미터만 전달)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    train_ds = ScaneatDataset(TRAIN_IMG, TRAIN_JSON)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    print(f"🚀 학습 시작! (Device: {device})")

    for epoch in range(20): # 단계별 실험을 위해 20에폭 설정
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
            if i % 20 == 0:
                print(f"   🔹 Epoch [{epoch+1}] Step [{i}/{len(train_loader)}] - Loss: {losses.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"📁 Epoch [{epoch+1}] 평균 Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "best_maskrcnn_bj_frozen.pth")

if __name__ == "__main__":
    main()