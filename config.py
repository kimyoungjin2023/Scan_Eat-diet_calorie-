# 모든 설정값을 한 곳에서 관리

DATA_YAML = "/content/drive/MyDrive/please/data.yaml"
PROJECT_DIR = "runs/segment"

PRETRAIN_CONFIG = dict(
    data=DATA_YAML,
    project=PROJECT_DIR,
    name="pretrain",
    epochs=300,
    imgsz=640,
    batch=8,
    freeze=10,
    patience=100,
    lr0=0.001,
    cos_lr=True,
    # device=0, # 이건 GPU있을 때만 

    # 색상 증강
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,

    # 기하학적 증강
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,

    # 플립
    fliplr=0.5,
    flipud=0.1,

    # 모자이크 / 믹스업
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,

    # 기타
    erasing=0.4,
    crop_fraction=1.0,
)

FINETUNE_CONFIG = dict(
    data=DATA_YAML,
    project=PROJECT_DIR,
    name="finetune",
    epochs=100,
    imgsz=640,
    batch=8,
    patience=50,
    lr0=0.0001,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=3,
    freeze=0,
    device=0,

    # 색상 증강 (약하게)
    hsv_h=0.01,
    hsv_s=0.3,
    hsv_v=0.2,

    # 기하학적 증강 (약하게)
    degrees=5.0,
    translate=0.05,
    scale=0.3,
    shear=1.0,
    perspective=0.0,

    # 플립
    fliplr=0.5,
    flipud=0.0,

    # 모자이크 끔
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.05,

    # 기타
    erasing=0.2,
    crop_fraction=1.0,
)