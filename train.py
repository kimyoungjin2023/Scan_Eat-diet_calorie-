from ultralytics import YOLO
from config import PRETRAIN_CONFIG

def run_train():
    model = YOLO("yolov8n-seg.pt")
    model.train(**PRETRAIN_CONFIG)