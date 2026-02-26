import json

def fix_coco_json(json_path, save_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fixed_count = 0
    for anno in data['annotations']:
        # bbox가 없거나 비어있으면 아주 작은 더미 좌표 [0, 0, 1, 1]를 넣어줍니다.
        if "bbox" not in anno or not anno["bbox"] or len(anno["bbox"]) != 4:
            anno["bbox"] = [0, 0, 1, 1] # 최소 크기 좌표
            fixed_count += 1
        elif anno["bbox"][2] <= 0 or anno["bbox"][3] <= 0:
            anno["bbox"][2] = 1
            anno["bbox"][3] = 1
            fixed_count += 1

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 완료: {json_path}")
    print(f"🛠️ 수리된 불량 데이터: {fixed_count}개")

# 파일 수리 실행
train_path = r"C:\scan_eat\data\train\_annotations.coco.json"
val_path = r"C:\scan_eat\data\valid\_annotations.coco.json"

fix_coco_json(train_path, train_path.replace(".json", "_final.json"))
fix_coco_json(val_path, val_path.replace(".json", "_final.json"))