import json
import os

def add_dummy_masks(json_path):
    print(f"작업 시작: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    fixed_count = 0
    for ann in data['annotations']:
        # segmentation 키가 없거나 비어있다면 생성
        if 'segmentation' not in ann or not ann['segmentation']:
            # bbox 정보가 제대로 있는지 확인 (x, y, width, height)
            if 'bbox' in ann and len(ann['bbox']) == 4:
                x, y, w, h = ann['bbox']
            else:
                # bbox도 없다면 임의의 작은 박스로 설정
                x, y, w, h = 0, 0, 10, 10 
                ann['bbox'] = [x, y, w, h]
            
            # bbox를 기반으로 4개의 꼭짓점 좌표 생성 (Polygon 형식)
            # COCO 포맷: [x1, y1, x2, y1, x2, y2, x1, y2]
            poly = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            ann['segmentation'] = poly
            fixed_count += 1

    # 같은 파일명으로 덮어쓰기
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 완료! {fixed_count}개의 빈 마스크를 복구했습니다.\n")

# 실행
train_json = r"C:\scan_eat\data\train\_annotations.coco_final.json"
valid_json = r"C:\scan_eat\data\valid\_annotations.coco_final.json"

if os.path.exists(train_json): add_dummy_masks(train_json)
if os.path.exists(valid_json): add_dummy_masks(valid_json)