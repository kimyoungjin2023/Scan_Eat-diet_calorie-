# soyun/config.py

# ================================================================
#  1. CONFIG
# ================================================================
CFG = {
    "data_root":     "/content/drive/MyDrive/CV_study/Scan_Eat_data",
    "save_dir":      "/content/drive/MyDrive/food_ckpts/mask2former_noisyvit",
    "img_size":       224,    # DeiT-Small 고정 입력 크기
    "mask_size":      128,    # Mask2Former 출력 해상도
    "backbone_name":  "deit_small_patch16_224",  # NoisyViT 백본
    "noise_std":      0.15,
    "num_queries":    100,    # Mask2Former 쿼리 수
    "dec_dim":        256,
    "num_dec_layers": 6,      # Mask2Former Transformer 디코더 레이어 수
    "epochs":         50,
    "batch_size":     4,      # Mask2Former는 메모리 많이 씀
    "lr":             5e-5,
    "seed":           42,
    "dropout":        0.3,
    "early_stop_patience": 10,
    "val_ratio":      0.2,
    # Loss weights
    "w_cls":          2.0,    # Mask2Former는 cls loss 비중 높임
    "w_mask":         5.0,    # mask loss 비중 높임
    "w_dice":         5.0,    # dice loss 비중 높임
}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(CFG["seed"])