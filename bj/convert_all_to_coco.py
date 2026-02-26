import json
import os
import cv2

# 1. 경로 설정
DATASET_ROOT = "C:/Users/BJ/Desktop/Ai_HealthCare/cv_project/scan_eat/data"
SETS = ["train", "valid", "test"]

# 2. YAML에서 가져온 클래스 리스트
CLASS_NAMES = [
    "Bokkeum_Dakgalbi", "Bokkeum_DriedShrimpBokkeum", "Bokkeum_DriedSquidBokkeum",
    "Bokkeum_EggplantBokkeum", "Bokkeum_Japchae", "Bokkeum_MiyeokJulgiBokkeum",
    "Bokkeum_PotatoSliceBokkeum", "Bokkeum_SpicyDriedSquidBokkeum", "Bokkeum_StirFriedAnchovies",
    "Bokkeum_WebfootOctopusBokkeum", "Fruit_Lemon", "Fruit_Tomato", "Gim",
    "Grilled_Garlic", "Grilled_GrilledCutlassfish", "Grilled_GrilledEel",
    "Grilled_GrilledMackerel", "Grilled_GrilledSpicesEel", "Grilled_Tteokgalbi",
    "Guk_MiyeokGuk", "Jeotgal_GanjangCrab", "Jeotgal_SpicyMarinatedCrab",
    "Jorim_Janjorim", "Kimch_Kimch", "Kimchi", "Kimchi_Kimchi",
    "Kimchi_YoungRadishKimchi", "Muchim_KongnamulMuchim", "Muchim_ZucchiniMuchim",
    "Muchim_cheongpomungMuchim", "Mushroom_Mushroom_KingOysterMushroom", "Namul_Sigeumchinamul",
    "None_EggFriedRice", "None_TofuKimchi", "Pancake_EggRoll", "Pickled_Gochujangajji",
    "Pickled_KkaennipJangajji", "Pickled_Pickle", "Rice_MixedGrainRice", "Rice_WhiteRice",
    "Vegetable_Garlic", "Vegetable_Lettuce", "Vegetable_Ssamvegetables", "Vegetable_gochu"
]

def convert():
    for set_name in SETS:
        set_dir = os.path.join(DATASET_ROOT, set_name)
        images_dir = os.path.join(set_dir, "images")
        labels_dir = os.path.join(set_dir, "labels")
        
        if not os.path.exists(images_dir):
            continue

        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]
        }

        ann_id = 1
        img_list = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📦 {set_name} 변환 중...")

        for img_id, img_file in enumerate(img_list):
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape
            
            coco["images"].append({
                "id": img_id, "file_name": img_file, "width": w, "height": h
            })

            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = list(map(float, line.strip().split()))
                        if len(parts) < 5: continue
                        
                        cls_id = int(parts[0])
                        # YOLO 세그멘테이션 좌표 복원
                        poly = [p * (w if i % 2 == 0 else h) for i, p in enumerate(parts[1:])]
                        
                        coco["annotations"].append({
                            "id": ann_id, "image_id": img_id, "category_id": cls_id,
                            "segmentation": [poly], "area": 0, "bbox": [], "iscrowd": 0
                        })
                        ann_id += 1

        with open(os.path.join(set_dir, "_annotations.coco.json"), "w") as f:
            json.dump(coco, f, indent=4)
        print(f"✅ {set_name} 완료!")

if __name__ == "__main__":
    convert()