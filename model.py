"""
DINOv2 Backbone + Mask2Former Head 모델
HuggingFace transformers 기반
"""

import torch
import torch.nn as nn
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Dinov2Config,
    Dinov2Model,
)


def build_dino_mask2former(
    num_classes: int,
    dino_model_name: str = "facebook/dinov2-base",
    pretrained_mask2former: str = "facebook/mask2former-swin-base-coco-instance",
    freeze_dino: bool = False,
) -> Mask2FormerForUniversalSegmentation:
    """
    DINOv2 backbone을 Mask2Former에 연결한 모델 생성

    Args:
        num_classes           : 클래스 수 (배경 제외)
        dino_model_name       : HuggingFace DINOv2 모델 이름
        pretrained_mask2former: Mask2Former 가중치 초기화용 (헤드만 사용)
        freeze_dino           : DINOv2 백본 freeze 여부

    Returns:
        model (Mask2FormerForUniversalSegmentation)
    """

    print(f"  DINOv2 백본 로드: {dino_model_name}")
    dino = Dinov2Model.from_pretrained(dino_model_name)

    if freeze_dino:
        for param in dino.parameters():
            param.requires_grad = False
        print("  DINOv2 백본 freeze 완료")

    # DINOv2 hidden_size에 맞춰 Mask2Former config 수정
    dino_cfg: Dinov2Config = dino.config
    hidden_size = dino_cfg.hidden_size          # base=768, large=1024

    print(f"  Mask2Former 설정 구성 (hidden_size={hidden_size})")
    m2f_config = Mask2FormerConfig.from_pretrained(pretrained_mask2former)

    # 백본을 DINOv2로 교체
    m2f_config.backbone_config = dino_cfg
    m2f_config.backbone = dino_model_name
    m2f_config.use_pretrained_backbone = True

    # 클래스 수 업데이트
    m2f_config.num_labels = num_classes
    # id2label / label2id는 학습 후 따로 세팅
    m2f_config.id2label = {i: str(i) for i in range(num_classes)}
    m2f_config.label2id = {str(i): i for i in range(num_classes)}

    print("  Mask2Former 모델 생성 중...")
    model = Mask2FormerForUniversalSegmentation(m2f_config)

    # ── DINOv2 가중치를 백본에 직접 주입 ──────────────────
    # transformers의 AutoBackbone이 DINOv2를 지원하지 않는 경우 수동 연결
    _inject_dino_backbone(model, dino)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  총 파라미터: {total:,}  |  학습 가능: {trainable:,}")

    return model


def _inject_dino_backbone(model: Mask2FormerForUniversalSegmentation,
                          dino: Dinov2Model):
    """
    Mask2Former 내부 backbone 모듈을 DINOv2로 교체하고
    feature projection 레이어를 추가하는 wrapper
    """
    # transformers >= 4.38에서는 model.model.pixel_level_module.encoder가 backbone
    try:
        backbone_module = model.model.pixel_level_module.encoder
        print(f"  기존 백본: {type(backbone_module).__name__} → DINOv2로 교체")
    except AttributeError:
        print("  백본 모듈 경로를 찾지 못했습니다. 수동 확인 필요.")
        return

    # DINOv2를 DINOv2BackboneWrapper로 감싸서 멀티스케일 feature 출력
    dino_channels = dino.config.hidden_size
    m2f_channels = model.config.feature_size  # Mask2Former 내부 채널

    wrapper = DINOv2BackboneWrapper(dino, dino_channels, m2f_channels)
    model.model.pixel_level_module.encoder = wrapper


class DINOv2BackboneWrapper(nn.Module):
    """
    DINOv2를 Mask2Former pixel_level_module의 encoder로 사용하기 위한 wrapper.
    DINOv2의 중간 레이어 feature를 4단계 스케일로 추출한다.
    """

    # DINOv2-base 기준: 12 layers → 3, 6, 9, 12번째 레이어 사용
    LAYER_INDICES = [2, 5, 8, 11]   # 0-indexed

    def __init__(self, dino: Dinov2Model, dino_hidden: int, out_channels: int):
        super().__init__()
        self.dino = dino
        self.dino_hidden = dino_hidden

        # 각 스케일별 1x1 projection (DINOv2 hidden → Mask2Former feature_size)
        self.projections = nn.ModuleList([
            nn.Conv2d(dino_hidden, out_channels, kernel_size=1)
            for _ in self.LAYER_INDICES
        ])

    def forward(self, pixel_values, **kwargs):
        # DINOv2 중간 레이어 feature 추출
        outputs = self.dino(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # tuple of (B, seq_len, C)
        # seq_len = 1(CLS) + (H/14)*(W/14) patches

        B = pixel_values.shape[0]
        H, W = pixel_values.shape[-2], pixel_values.shape[-1]
        ph, pw = H // 14, W // 14   # DINOv2 patch size = 14

        feature_maps = []
        for proj, layer_idx in zip(self.projections, self.LAYER_INDICES):
            feat = hidden_states[layer_idx + 1]  # (B, 1+ph*pw, C)
            feat = feat[:, 1:, :]                # CLS 토큰 제거
            feat = feat.reshape(B, ph, pw, self.dino_hidden)
            feat = feat.permute(0, 3, 1, 2)      # (B, C, ph, pw)
            feat = proj(feat)
            feature_maps.append(feat)

        # Mask2Former는 feature_maps 리스트를 기대함
        return feature_maps
