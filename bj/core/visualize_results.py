import onnxruntime as ort
import numpy as np
import cv2
import os

# 1. 경로 설정 (상대 경로 적용)
# 현재 파일(core/visualize_results.py) 위치를 기준으로 모델 파일을 찾습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# core 폴더에서 한 단계 위(..)로 가서 models/weights 폴더로 접근합니다.
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "weights", "best_maskrcnn_bj_int8.onnx")

CLASS_NAMES = [
    "background", "Bokkeum_Dakgalbi", "Bokkeum_DriedShrimpBokkeum", "Bokkeum_DriedSquidBokkeum",
    "Bokkeum_EggplantBokkeum", "Bokkeum_Japchae", "Bokkeum_MiyeokJulgiBokkeum",
    "Bokkeum_PotatoSliceBokkeum", "Bokkeum_SpicyDriedSquidBokkeum", "Bokkeum_StirFriedAnchovies",
    "Bokkeum_WebfootOctopusBokkeum", "Fruit_Lemon", "Fruit_Tomato", "Gim",
    "Grilled_Garlic", "Grilled_GrilledCutlassfish", "Grilled_GrilledEel",
    "Grilled_GrilledMackerel", "Grilled_GrilledSpicesEel", "Grilled_Tteokgalbi",
    "Guk_MiyeokGuk", "Jeotgal_GanjangCrab", "Jeotgal_SpicyMarinatedCrab",
    "Jorim_Janjorim", "Kimch_Kimch", "Kimchi", "Kimchi_Kimchi", "Kimchi_YoungRadishKimchi",
    "Muchim_KongnamulMuchim", "Muchim_ZucchiniMuchim", "Muchim_cheongpomungMuchim",
    "Mushroom_Mushroom_KingOysterMushroom", "Namul_Sigeumchinamul", "None_EggFriedRice",
    "None_TofuKimchi", "Pancake_EggRoll", "Pickled_Gochujangajji", "Pickled_KkaennipJangajji",
    "Pickled_Pickle", "Rice_MixedGrainRice", "Rice_WhiteRice", "Vegetable_Garlic",
    "Vegetable_Lettuce", "Vegetable_Ssamvegetables", "Vegetable_gochu"
]

# 2. 모델 세션 로드 (에러 체크 추가)
if not os.path.exists(MODEL_PATH):
    print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
else:
    session = ort.InferenceSession(MODEL_PATH)

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
    inter_x2, inter_y2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)

def visualize_and_process(image_input_path):
    img = cv2.imread(image_input_path)
    if img is None: 
        print(f"❌ 이미지를 읽을 수 없습니다: {image_input_path}")
        return None
    
    # 전처리
    img_resized = cv2.resize(img, (800, 800))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    # 추론 실행
    boxes, labels, scores, masks = session.run(None, {'input': input_tensor})

    # NMS(Non-Maximum Suppression) 처리
    keep_indices = []
    sorted_idx = np.argsort(scores)[::-1]
    for i in sorted_idx:
        if scores[i] < 0.5: continue
        keep = True
        for j in keep_indices:
            if labels[i] == labels[j] and compute_iou(boxes[i], boxes[j]) > 0.4:
                keep = False; break
        if keep: keep_indices.append(i)

    llm_data = {"detected_foods": []}
    vis_img = img_resized.copy()
    
    # 결과 시각화 및 데이터 추출
    for idx in keep_indices:
        label_name = CLASS_NAMES[labels[idx]]
        mask_bool = masks[idx][0] > 0.5
        area = int(np.sum(mask_bool))

        llm_data["detected_foods"].append({
            "name": label_name, "area_px": area, "confidence": round(float(scores[idx]), 2)
        })

        color = [int(c) for c in np.random.randint(0, 255, 3)]
        vis_img[mask_bool] = vis_img[mask_bool] * 0.5 + np.array(color) * 0.5
        box = boxes[idx].astype(int)
        cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(vis_img, label_name, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. 결과 이미지 임시 저장 (파일명은 main.py의 로직과 일치시킴)
    output_temp_path = "final_inference_result.jpg"
    cv2.imwrite(output_temp_path, vis_img)
    
    return llm_data