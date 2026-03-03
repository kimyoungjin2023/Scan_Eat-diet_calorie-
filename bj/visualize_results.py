import onnxruntime as ort
import numpy as np
import cv2
import os
import json

# bj 폴더 내의 send_to_llm.py에서 함수를 가져옵니다.
from send_to_llm import send_to_gemini

# 1. 경로 설정
MODEL_PATH = r"C:\scan_eat\weights\best_maskrcnn_bj_int8.onnx"
IMAGE_PATH = r"C:\scan_eat\data/test/images/Img_081_0353_jpg.rf.485331118d38c4cb638fb58ea5ef50a3.jpg"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "final_inference_result2.jpg")

# 2. 클래스 리스트 (생략 없이 그대로 사용)
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

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
    inter_x2, inter_y2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)

def visualize_and_process():
    session = ort.InferenceSession(MODEL_PATH)
    img = cv2.imread(IMAGE_PATH)
    img_resized = cv2.resize(img, (800, 800))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

    print(f"🚀 추론 시작: {os.path.basename(IMAGE_PATH)}")
    boxes, labels, scores, masks = session.run(None, {'input': input_tensor})

    keep_indices = []
    sorted_idx = np.argsort(scores)[::-1]
    
    for i in sorted_idx:
        if scores[i] < 0.5: continue
        keep = True
        for j in keep_indices:
            if labels[i] == labels[j] and compute_iou(boxes[i], boxes[j]) > 0.4:
                keep = False; break
        if keep: keep_indices.append(i)

    llm_data = {"detected_foods": [], "summary": {}}
    vis_img = img_resized.copy()
    
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
        cv2.putText(vis_img, f"{label_name}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for food in llm_data["detected_foods"]:
        name = food["name"]
        llm_data["summary"][name] = llm_data["summary"].get(name, 0) + food["area_px"]

    cv2.imwrite(OUTPUT_PATH, vis_img)
    print(f"🎉 모델 분석 완료! 결과 이미지 저장: {OUTPUT_PATH}")
    
    # ⭐ [핵심 수정] Gemini에게 데이터를 전달하기 위해 반환값을 설정합니다.
    return llm_data

if __name__ == "__main__":
    # 1. 모델 분석 수행 (JSON 데이터 생성)
    result_data = visualize_and_process() 

    # 2. 분석된 데이터가 있으면 Gemini에게 전송
    if result_data and result_data["detected_foods"]:
        print("📝 분석된 데이터를 기반으로 영양 보고서를 생성합니다...")
        analysis_report = send_to_gemini(OUTPUT_PATH, result_data)
        
        print("\n" + "="*50)
        print("🥗 전문 영양사 Gemini의 분석 결과")
        print("="*50)
        print(analysis_report)
    else:
        print("⚠️ 검출된 음식이 없어 영양 분석을 진행하지 않습니다.")