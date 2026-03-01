# DINOv2 + Mask2Former — Instance Segmentation

YOLO Segmentation 데이터를 그대로 사용해 DINOv2 백본 + Mask2Former 헤드로 학습하는 파이프라인입니다.

---

## 📁 폴더 구조

```
dino_mask2former/
├── convert_yolo_to_coco.py   # YOLO seg → COCO JSON 변환
├── dataset.py                # PyTorch Dataset
├── model.py                  # DINOv2 + Mask2Former 모델 빌더
├── train.py                  # 학습 스크립트
├── inference.py              # 추론 & 시각화
├── configs/
│   └── train_config.yaml     # 학습 설정
└── requirements.txt
```

---

## ⚡ 빠른 시작

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. YOLO → COCO 변환

YOLO `dataset.yaml` 파일 경로를 지정하고 변환합니다.

```bash
python convert_yolo_to_coco.py
```

`dataset.yaml` 예시:
```yaml
path: /home/user/my_dataset
train: images/train
val:   images/val
names:
  0: cat
  1: dog
  2: person
```

변환 후 `coco_annotations/train.json`, `coco_annotations/val.json` 생성됩니다.

### 3. 설정 수정

`configs/train_config.yaml`에서 데이터 경로와 하이퍼파라미터를 수정하세요.

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `dino_backbone` | `dinov2-base` | small/base/large/giant 선택 가능 |
| `freeze_dino` | `false` | true면 백본 고정 (빠른 실험용) |
| `batch_size` | `4` | VRAM 12GB 기준. 부족하면 2로 줄이세요 |
| `epochs` | `50` | |
| `amp` | `true` | Mixed Precision으로 VRAM 절약 |

### 4. 학습

```bash
python train.py --config configs/train_config.yaml
```

### 5. 추론

```bash
python inference.py --model_dir outputs/best_model --image test.jpg
```

---

## 🔧 VRAM 가이드

| 모델 | img_size | batch_size | 필요 VRAM |
|------|----------|------------|-----------|
| dinov2-small | 640 | 4 | ~8 GB |
| dinov2-base  | 640 | 4 | ~12 GB |
| dinov2-large | 640 | 2 | ~16 GB |
| dinov2-large | 640 | 4 | ~24 GB |

VRAM이 부족하면:
- `batch_size` ↓
- `img_size` ↓ (예: 512)
- `freeze_dino: true` 설정

---

## 📌 참고

- DINOv2: https://github.com/facebookresearch/dinov2
- Mask2Former: https://huggingface.co/docs/transformers/model_doc/mask2former
