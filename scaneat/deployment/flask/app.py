"""
SCANEAT - Flask Web Server
음식 이미지 → Detection + Segmentation
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import base64
import cv2
import os

app = Flask(__name__)
CORS(app)  # CORS 허용

# ============================================
# 모델 로드
# ============================================
MODEL_PATH = 'best_simplified.onnx'  # 또는 best.pt

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ 모델 파일 없음: {MODEL_PATH}")
    print("   best.pt 또는 best.onnx 파일을 같은 폴더에 넣으세요")
    MODEL_PATH = 'best.pt'  # fallback

print(f"🔥 모델 로딩: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task='segment')
print("✅ 모델 로드 완료!")


# ============================================
# HTML 템플릿 (간단한 웹 UI)
# ============================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SCANEAT - 음식 칼로리 예측</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            background: #f8f9ff;
            border-color: #764ba2;
        }
        .upload-area.dragover {
            background: #e8ebff;
            border-color: #667eea;
        }
        input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
            transition: transform 0.2s;
        }
        .btn:hover { transform: scale(1.05); }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        #preview {
            max-width: 100%;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        #result {
            margin-top: 20px;
            display: none;
        }
        .result-item {
            background: #f8f9ff;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        .food-name {
            font-weight: bold;
            color: #667eea;
            font-size: 1.2em;
        }
        .confidence {
            color: #764ba2;
            font-weight: bold;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .stats {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .emoji { font-size: 3em; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍽️ SCANEAT</h1>
        <p class="subtitle">음식 사진으로 칼로리 예측하기</p>
        
        <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
            <div class="emoji">📸</div>
            <p style="font-size: 1.2em; color: #667eea; font-weight: bold;">
                클릭하거나 드래그해서 사진 업로드
            </p>
            <p style="color: #999; margin-top: 10px;">
                JPG, PNG 지원
            </p>
        </div>
        
        <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])">
        
        <img id="preview" src="" alt="Preview">
        
        <button class="btn" id="analyzeBtn" onclick="analyzeImage()" disabled>
            🔍 음식 분석하기
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px; color: #667eea;">분석 중...</p>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        let currentFile = null;
        
        // 드래그 앤 드롭
        const uploadArea = document.getElementById('uploadArea');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });
        
        // 파일 처리
        function handleFile(file) {
            if (!file) return;
            
            currentFile = file;
            const reader = new FileReader();
            
            reader.onload = (e) => {
                const preview = document.getElementById('preview');
                preview.src = e.target.result;
                preview.style.display = 'block';
                document.getElementById('analyzeBtn').disabled = false;
                document.getElementById('result').style.display = 'none';
            };
            
            reader.readAsDataURL(file);
        }
        
        // 분석
        async function analyzeImage() {
            if (!currentFile) return;
            
            const formData = new FormData();
            formData.append('file', currentFile);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('분석 실패: ' + data.error);
                }
            } catch (error) {
                alert('에러 발생: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
        
        // 결과 표시
        function displayResults(data) {
            const resultDiv = document.getElementById('result');
            
            if (data.detections.length === 0) {
                resultDiv.innerHTML = `
                    <div class="result-item">
                        <p style="text-align: center; color: #999;">
                            음식을 찾지 못했습니다 😢
                        </p>
                    </div>
                `;
            } else {
                let html = '<h2 style="color: #667eea; margin-bottom: 15px;">검출 결과</h2>';
                
                data.detections.forEach((det, idx) => {
                    html += `
                        <div class="result-item">
                            <div class="food-name">${idx + 1}. ${det.class}</div>
                            <div style="margin-top: 5px;">
                                신뢰도: <span class="confidence">${(det.confidence * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    `;
                });
                
                // 통계
                html += `
                    <div class="stats">
                        <strong>📊 통계</strong><br>
                        총 검출: ${data.detections.length}개<br>
                        처리 시간: ${data.processing_time.toFixed(2)}초
                    </div>
                `;
                
                resultDiv.innerHTML = html;
            }
            
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""


# ============================================
# API 엔드포인트
# ============================================

@app.route('/')
def index():
    """메인 페이지"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """음식 예측 API"""
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400
    
    try:
        # 이미지 로드
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # RGB 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 추론
        import time
        start_time = time.time()
        
        results = model.predict(
            image,
            conf=0.25,
            iou=0.5,
            verbose=False
        )
        
        processing_time = time.time() - start_time
        
        # 결과 파싱
        detections = []
        
        for r in results:
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    detection = {
                        'class': model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist(),
                    }
                    
                    # Mask 정보 (선택)
                    if r.masks is not None and i < len(r.masks):
                        # Mask는 용량이 크므로 간단한 정보만
                        mask_data = r.masks[i].data.cpu().numpy()
                        detection['mask_size'] = mask_data.shape
                    
                    detections.append(detection)
        
        # 성공 응답
        return jsonify({
            'success': True,
            'detections': detections,
            'processing_time': processing_time,
            'model': MODEL_PATH,
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_PATH,
        'model_loaded': model is not None
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """모델 클래스 목록"""
    return jsonify({
        'classes': list(model.names.values()),
        'num_classes': len(model.names)
    })


# ============================================
# 실행
# ============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🍽️ SCANEAT Flask Server")
    print("="*70)
    print(f"모델: {MODEL_PATH}")
    print(f"클래스: {len(model.names)}개")
    print("="*70)
    print("\n🌐 서버 시작!")
    print("   로컬: http://localhost:5000")
    print("   외부: http://0.0.0.0:5000")
    print("\n종료: Ctrl+C")
    print("="*70 + "\n")
    
    # 실행
    app.run(
        host='0.0.0.0',  # 외부 접속 허용
        port=5000,
        debug=True  # 개발 모드 (배포 시 False)
    )