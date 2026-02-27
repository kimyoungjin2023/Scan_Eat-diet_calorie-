import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from .config import CFG

class FoodDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.ms = CFG["mask_size"]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_raw = Image.open(s["image"]).convert("RGB")
        mask = torch.zeros((self.ms, self.ms), dtype=torch.float32)
        for b in s["boxes"]:
            _, xc, yc, bw, bh = b
            x1, y1 = int((xc-bw/2)*self.ms), int((yc-bh/2)*self.ms)
            x2, y2 = int((xc+bw/2)*self.ms), int((yc+bh/2)*self.ms)
            mask[max(0,y1):min(self.ms,y2), max(0,x1):min(self.ms,x2)] = 1.0
        img = self.transform(img_raw) if self.transform else img_raw
        return {"image": img, "labels": torch.tensor(s["label"], dtype=torch.long), "masks": mask.unsqueeze(0)}

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1 - (2. * intersection + 1.0) / (union + 1.0)).mean()

def compute_loss(cls_logits, pred_masks, labels, masks):
    B = labels.shape[0]
    gt_masks = masks.squeeze(1)
    # Greedy matching (Simplified)
    with torch.no_grad():
        mask_iou = (torch.sigmoid(pred_masks) * gt_masks.unsqueeze(1)).sum(dim=(-2,-1))
        best_q = mask_iou.argmax(dim=1)
    
    loss_cls, loss_mask, loss_dice = 0, 0, 0
    for b in range(B):
        q = best_q[b]
        loss_cls += F.cross_entropy(cls_logits[b], torch.full((CFG["num_queries"],), labels[b], device=labels.device, dtype=torch.long))
        loss_mask += F.binary_cross_entropy_with_logits(pred_masks[b, q], gt_masks[b])
        loss_dice += dice_loss(pred_masks[b, q], gt_masks[b])
    
    total = (CFG["w_cls"] * loss_cls + CFG["w_mask"] * loss_mask + CFG["w_dice"] * loss_dice) / B
    return total