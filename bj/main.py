from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.send_to_llm import send_to_gemini 
from core.visualize_results import visualize_and_process
import json, os, pymysql, shutil

# 1. 경로 설정 (자동화)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "outputs") 

if not os.path.exists(UPLOAD_DIR): 
    os.makedirs(UPLOAD_DIR)

app = FastAPI()

# 2. CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB 설정
db_config = {
    'host': 'localhost', 'user': 'root', 'password': '0000', 
    'db': 'scaneat_service', 'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor, 'connect_timeout': 5
}

# 3. API 엔드포인트

@app.post("/analyze")
async def analyze_food(request: Request, file: UploadFile = File(...)): # 📍 request 추가
    raw_path = os.path.join(UPLOAD_DIR, f"raw_{file.filename}")
    result_filename = f"res_{file.filename}"
    final_path = os.path.join(UPLOAD_DIR, result_filename)
    
    try:
        # (1) 이미지 저장
        content = await file.read()
        with open(raw_path, "wb") as f: f.write(content)

        # (2) Mask R-CNN 추론
        detection_data = visualize_and_process(raw_path)
        
        temp_result = "final_inference_result.jpg"
        if os.path.exists(temp_result):
            shutil.move(temp_result, final_path)
        else: 
            shutil.copy(raw_path, final_path)

        # (3) Gemini 상세 분석
        report_json = send_to_gemini(final_path, detection_data)
        clean_json = report_json.replace("```json", "").replace("```", "").strip()
        res = json.loads(clean_json)

        # 📍 분석 직후 프론트엔드에서 이미지를 바로 띄울 수 있도록 URL 추가
        base_url = str(request.base_url).rstrip('/')
        res['image_url'] = f"{base_url}/outputs/{result_filename}"

        # (4) DB 저장
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO meal_summary 
                           (image_path, total_calories, total_carbs, total_protein, total_fat, 
                            total_sugar, total_sodium, total_cholesterol, advice)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""", 
                        (result_filename, res.get('total_calories', 0), res.get('total_carbs', 0), 
                         res.get('total_protein', 0), res.get('total_fat', 0), res.get('total_sugar', 0), 
                         res.get('total_sodium', 0), res.get('total_cholesterol', 0), res.get('advice', "")))
            m_id = conn.insert_id()
            
            for i in res.get('items', []):
                cur.execute("INSERT INTO meal_items (meal_id, food_name, estimated_weight, item_calories) VALUES (%s, %s, %s, %s)",
                            (m_id, i.get('name', '음식'), i.get('weight', 0), i.get('calories', 0)))
        conn.commit()
        conn.close()
        return res
        
    except Exception as e:
        print(f"❌ 분석 오류 상세: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(request: Request):
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM meal_summary ORDER BY created_at DESC")
            meals = cur.fetchall()
            
            base_url = str(request.base_url).rstrip('/')
            
            for m in meals:
                cur.execute("SELECT * FROM meal_items WHERE meal_id = %s", (m['meal_id'],))
                m['items'] = cur.fetchall()
                
                # 📍 이미지 주소를 접속 환경(Ngrok/Local)에 맞게 동적 생성
                m['image_url'] = f"{base_url}/outputs/{m['image_path']}"
                # 📍 전후 비교를 위한 원본 이미지 주소도 명시적으로 전달 (선택사항)
                m['raw_image_url'] = f"{base_url}/outputs/raw_{m['image_path'].replace('res_', '')}"
                
        conn.close()
        return meals
    except Exception as e:
        print(f"❌ 이력 로드 실패: {e}")
        return []

@app.delete("/meal/{meal_id}")
async def delete_meal(meal_id: int):
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM meal_items WHERE meal_id = %s", (meal_id,))
            cur.execute("DELETE FROM meal_summary WHERE meal_id = %s", (meal_id,))
        conn.commit()
        conn.close()
        return {"message": "삭제 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. 정적 파일 마운트 (반드시 순서 준수!)
app.mount("/outputs", StaticFiles(directory=UPLOAD_DIR), name="outputs")
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)