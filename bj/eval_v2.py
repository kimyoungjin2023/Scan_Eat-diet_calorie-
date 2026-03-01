import os
import torch
import torchvision
from PIL import Image
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json

# 1. 경로 설정 (학습 코드와 동일하게)
DATASET_ROOT = "C:/scan_eat/data"
VAL_JSON = os.path.join(DATASET_ROOT, "valid/_annotations.coco_final.json")
VAL_IMG = os.path.join(DATASET_ROOT, "valid/images")
MODEL_PATH = "best_maskrcnn_bj_final.pth"

def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

@torch.no_grad()
def evaluate():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 45
    
    # 모델 로드
    model = get_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # COCO 정답지 로드
    coco_gt = COCO(VAL_JSON)
    img_ids = sorted(coco_gt.getImgIds())
    
    results = []
    print(f"🚀 mAP 측정 시작 (대상: {len(img_ids)}장)...")

    for img_id in img_ids:
        # 이미지 로드
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(device)

        # 추론
        outputs = model([img_tensor])[0]
        
        # 결과 정리 (CPU로 이동)
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        boxes = outputs['boxes'].cpu().numpy()
        masks = outputs['masks'].cpu().numpy()

        for i in range(len(scores)):
            if scores[i] < 0.05: continue  # 아주 낮은 점수는 제외
            
            # Mask를 RLE로 변환 (pycocotools 규격)
            res_mask = masks[i][0]
            res_mask = (res_mask > 0.5).astype(np.uint8)
            
            # Box [x1, y1, x2, y2] -> [x, y, w, h]
            box = boxes[i]
            coco_box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            
            # 결과 저장
            from pycocotools import mask as mask_util
            rle = mask_util.encode(np.asfortranarray(res_mask))
            rle['counts'] = rle['counts'].decode('utf-8')

            results.append({
                "image_id": img_id,
                "category_id": int(labels[i]),
                "bbox": [float(b) for b in coco_box],
                "score": float(scores[i]),
                "segmentation": rle
            })

    # JSON 임시 저장 후 평가
    with open("results.json", "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes("results.json")
    
    print("\n--- [Box mAP 결과] ---")
    coco_eval_box = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_box.evaluate()
    coco_eval_box.accumulate()
    coco_eval_box.summarize()

    print("\n--- [Segmentation mAP 결과] ---")
    coco_eval_seg = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_seg.evaluate()
    coco_eval_seg.accumulate()
    coco_eval_seg.summarize()

if __name__ == "__main__":
    evaluate()