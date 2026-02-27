import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .config import CFG

class NoisyViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(CFG["backbone_name"], pretrained=True, num_classes=0)
        self.embed_dim = self.vit.embed_dim
        d = len(self.vit.blocks)
        self.out_indices = [d//4-1, d//2-1, 3*d//4-1, d-1]

    def forward(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        if self.training:
            x = x + torch.randn_like(x) * CFG["noise_std"]
        x = torch.cat([self.vit.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        feats = []
        p = int((x.shape[1] - 1) ** 0.5)
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.out_indices:
                feat = x[:, 1:].transpose(1, 2).reshape(B, self.embed_dim, p, p)
                feats.append(feat)
        return feats, x[:, 0]

class PixelDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(in_dim, out_dim, 1) for _ in range(4)])
        self.fusions = nn.ModuleList([
            nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.GroupNorm(8, out_dim), nn.ReLU())
            for _ in range(3)
        ])
        self.out_proj = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, feats):
        projected = [proj(f) for proj, f in zip(self.projs, feats)]
        x = projected[-1]
        for i in range(2, -1, -1):
            x = F.interpolate(x, size=projected[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.fusions[i](x + projected[i])
        x = F.interpolate(x, size=(CFG["mask_size"], CFG["mask_size"]), mode="bilinear", align_corners=False)
        return self.out_proj(x)

class MaskFormerDecoder(nn.Module):
    def __init__(self, num_classes, dim, backbone_dim=384):
        super().__init__()
        self.num_queries = CFG["num_queries"]
        self.queries = nn.Embedding(self.num_queries, dim)
        self.cls_proj = nn.Linear(backbone_dim, dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=CFG["num_dec_layers"])
        self.cls_head = nn.Linear(dim, num_classes + 1)
        self.mask_proj = nn.Linear(dim, dim)

    def forward(self, pixel_feat, cls_token):
        B, C, H, W = pixel_feat.shape
        pixel_seq = pixel_feat.flatten(2).transpose(1, 2)
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1) + self.cls_proj(cls_token).unsqueeze(1)
        out = self.decoder(queries, pixel_seq)
        cls_logits = self.cls_head(out)
        pred_masks = torch.bmm(self.mask_proj(out), pixel_feat.flatten(2)).reshape(B, self.num_queries, H, W)
        return cls_logits, pred_masks

class NoisyViTMask2Former(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = NoisyViTBackbone()
        self.pixel_decoder = PixelDecoder(in_dim=self.backbone.embed_dim, out_dim=CFG["dec_dim"])
        self.mask_decoder = MaskFormerDecoder(num_classes=num_classes, dim=CFG["dec_dim"])

    def forward(self, x):
        feats, cls_token = self.backbone(x)
        pixel_feat = self.pixel_decoder(feats)
        return self.mask_decoder(pixel_feat, cls_token)