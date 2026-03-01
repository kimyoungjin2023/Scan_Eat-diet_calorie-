"""
DINOv2 + Mask2Former Instance Segmentation 학습 스크립트
사용법:
    python train.py --config configs/train_config.yaml
"""

import os
import argparse
import yaml
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Mask2FormerImageProcessor
from tqdm import tqdm

from dataset import COCOInstanceDataset, collate_fn
from model import build_dino_mask2former


# ──────────────────────────────────────────────────────────────
#  학습 설정 로드
# ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
#  Validation (IoU 기반 mAP 대신 간단한 loss 체크)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(val_loader, desc="  Validation", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
    return total_loss / len(val_loader)


# ──────────────────────────────────────────────────────────────
#  메인 학습 루프
# ──────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 학습 장치: {device}")

    # 클래스 정보
    with open(cfg["annotation"]["train_json"]) as f:
        coco_data = json.load(f)
    class_names = [c["name"] for c in sorted(coco_data["categories"], key=lambda x: x["id"])]
    num_classes = len(class_names)
    print(f"📦 클래스 수: {num_classes}  →  {class_names}")

    # ── Processor (전처리) ────────────────────────────────────
    processor = Mask2FormerImageProcessor.from_pretrained(
        cfg["model"]["pretrained_mask2former"],
        ignore_index=255,
        reduce_labels=False,
        size={"height": cfg["data"]["img_size"], "width": cfg["data"]["img_size"]},
    )

    # ── 데이터셋 & DataLoader ─────────────────────────────────
    print("\n📂 데이터셋 로드 중...")
    train_dataset = COCOInstanceDataset(
        images_dir=cfg["data"]["train_images"],
        annotation_json=cfg["annotation"]["train_json"],
        processor=processor,
    )
    val_dataset = COCOInstanceDataset(
        images_dir=cfg["data"]["val_images"],
        annotation_json=cfg["annotation"]["val_json"],
        processor=processor,
    )
    print(f"  Train: {len(train_dataset)}장  |  Val: {len(val_dataset)}장")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── 모델 ──────────────────────────────────────────────────
    print("\n🔧 모델 구성 중...")
    model = build_dino_mask2former(
        num_classes=num_classes,
        dino_model_name=cfg["model"]["dino_backbone"],
        pretrained_mask2former=cfg["model"]["pretrained_mask2former"],
        freeze_dino=cfg["model"].get("freeze_dino", False),
    )
    model.to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"],
        eta_min=float(cfg["train"]["lr"]) * 0.01,
    )

    # ── 출력 폴더 ─────────────────────────────────────────────
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── AMP (자동 혼합 정밀도) ────────────────────────────────
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"  AMP: {'ON' if use_amp else 'OFF'}")

    # ── 학습 루프 ─────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\n🏋️  학습 시작 — {cfg['train']['epochs']} epochs\n")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3}/{cfg['train']['epochs']}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss   = validate(model, val_loader, device)

        print(
            f"  Epoch {epoch:>3}  |  "
            f"train_loss: {avg_train_loss:.4f}  |  "
            f"val_loss: {avg_val_loss:.4f}  |  "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Best 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = output_dir / "best_model"
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"  ✅ Best 모델 저장: {save_path}  (val_loss={best_val_loss:.4f})")

        # 주기적 체크포인트
        if epoch % cfg["train"].get("save_every", 10) == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch}"
            model.save_pretrained(ckpt_path)
            print(f"  💾 체크포인트 저장: {ckpt_path}")

    print(f"\n✅ 학습 완료!  Best val_loss: {best_val_loss:.4f}")
    print(f"   모델 저장 위치: {output_dir / 'best_model'}")


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
