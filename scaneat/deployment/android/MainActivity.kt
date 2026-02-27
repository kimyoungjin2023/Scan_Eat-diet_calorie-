// ============================================
// SCANEAT Android App
// MainActivity.kt
// ============================================

package com.example.scaneat

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.scaneat.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private lateinit var tfliteInterpreter: Interpreter
    private lateinit var resultAdapter: ResultAdapter
    
    private var currentPhotoPath: String? = null
    
    // 모델 설정
    companion object {
        private const val MODEL_FILE = "best_int8.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val INPUT_SIZE = 640
        private const val CONFIDENCE_THRESHOLD = 0.25f
    }
    
    // 카메라 런처
    private val takePicture = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            currentPhotoPath?.let { path ->
                val bitmap = BitmapFactory.decodeFile(path)
                displayImage(bitmap)
                analyzeImage(bitmap)
            }
        }
    }
    
    // 갤러리 런처
    private val pickImage = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
            displayImage(bitmap)
            analyzeImage(bitmap)
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // TFLite 모델 로드
        loadModel()
        
        // RecyclerView 설정
        resultAdapter = ResultAdapter()
        binding.recyclerViewResults.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = resultAdapter
        }
        
        // 버튼 클릭
        binding.btnCamera.setOnClickListener {
            checkPermissionAndTakePhoto()
        }
        
        binding.btnGallery.setOnClickListener {
            pickImage.launch("image/*")
        }
    }
    
    // ============================================
    // 모델 로드
    // ============================================
    
    private fun loadModel() {
        try {
            val tfliteModel = loadModelFile()
            
            val options = Interpreter.Options().apply {
                setNumThreads(4)  // CPU 스레드 수
                // GPU 사용 시:
                // addDelegate(GpuDelegate())
            }
            
            tfliteInterpreter = Interpreter(tfliteModel, options)
            
            Toast.makeText(this, "✅ 모델 로드 완료", Toast.LENGTH_SHORT).show()
            
        } catch (e: Exception) {
            Toast.makeText(this, "❌ 모델 로드 실패: ${e.message}", 
                Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }
    }
    
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    // ============================================
    // 이미지 분석
    // ============================================
    
    private fun analyzeImage(bitmap: Bitmap) {
        binding.progressBar.visibility = android.view.View.VISIBLE
        binding.textViewStatus.text = "분석 중..."
        
        // 백그라운드 스레드에서 실행
        Thread {
            try {
                val startTime = System.currentTimeMillis()
                
                // 1. 전처리
                val inputBuffer = preprocessImage(bitmap)
                
                // 2. 추론
                val outputs = runInference(inputBuffer)
                
                // 3. 후처리
                val detections = postprocess(outputs)
                
                val processingTime = System.currentTimeMillis() - startTime
                
                // 4. UI 업데이트 (메인 스레드)
                runOnUiThread {
                    displayResults(detections, processingTime)
                    binding.progressBar.visibility = android.view.View.GONE
                }
                
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this, "분석 실패: ${e.message}", 
                        Toast.LENGTH_LONG).show()
                    binding.progressBar.visibility = android.view.View.GONE
                    binding.textViewStatus.text = "준비"
                }
                e.printStackTrace()
            }
        }.start()
    }
    
    // ============================================
    // 전처리
    // ============================================
    
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // 1. 리사이즈
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap, 
            INPUT_SIZE, 
            INPUT_SIZE, 
            true
        )
        
        // 2. ByteBuffer 생성 (Float32)
        val inputBuffer = ByteBuffer.allocateDirect(
            4 * INPUT_SIZE * INPUT_SIZE * 3  // 4 bytes per float
        ).apply {
            order(ByteOrder.nativeOrder())
        }
        
        // 3. 정규화 및 변환
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        for (pixel in pixels) {
            // RGB 추출 및 정규화 (0-255 → 0-1)
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
        
        return inputBuffer
    }
    
    // ============================================
    // 추론
    // ============================================
    
    private fun runInference(inputBuffer: ByteBuffer): Array<FloatArray> {
        // 출력 텐서 크기는 모델에 따라 다름
        // YOLOv8-seg: [1, 116, 8400] 형태
        // 116 = 4(bbox) + 80(classes) + 32(mask)
        
        val numDetections = 8400
        val numOutputs = 116
        
        val outputBuffer = Array(1) { FloatArray(numDetections * numOutputs) }
        
        tfliteInterpreter.run(inputBuffer, outputBuffer)
        
        return outputBuffer
    }
    
    // ============================================
    // 후처리 (NMS)
    // ============================================
    
    private fun postprocess(outputs: Array<FloatArray>): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        // 클래스 라벨 로드
        val labels = loadLabels()
        
        val output = outputs[0]
        val numDetections = 8400
        
        for (i in 0 until numDetections) {
            val startIdx = i * 116
            
            // Bounding Box (중심좌표, 너비, 높이)
            val cx = output[startIdx]
            val cy = output[startIdx + 1]
            val w = output[startIdx + 2]
            val h = output[startIdx + 3]
            
            // 클래스 확률 (인덱스 4-83)
            var maxConf = 0f
            var maxClass = 0
            
            for (c in 0 until 80) {  // 80 classes (COCO)
                val conf = output[startIdx + 4 + c]
                if (conf > maxConf) {
                    maxConf = conf
                    maxClass = c
                }
            }
            
            // Confidence threshold
            if (maxConf > CONFIDENCE_THRESHOLD) {
                // 좌표 변환 (중심 → 좌상단)
                val x1 = (cx - w / 2) * INPUT_SIZE
                val y1 = (cy - h / 2) * INPUT_SIZE
                val x2 = (cx + w / 2) * INPUT_SIZE
                val y2 = (cy + h / 2) * INPUT_SIZE
                
                val className = if (maxClass < labels.size) {
                    labels[maxClass]
                } else {
                    "Unknown"
                }
                
                detections.add(
                    Detection(
                        className = className,
                        confidence = maxConf,
                        bbox = RectF(x1, y1, x2, y2)
                    )
                )
            }
        }
        
        // NMS (Non-Maximum Suppression)
        return applyNMS(detections)
    }
    
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float = 0.5f): List<Detection> {
        // IoU 기반 NMS 구현
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val finalDetections = mutableListOf<Detection>()
        
        for (detection in sortedDetections) {
            var shouldAdd = true
            
            for (finalDet in finalDetections) {
                if (calculateIoU(detection.bbox, finalDet.bbox) > iouThreshold) {
                    shouldAdd = false
                    break
                }
            }
            
            if (shouldAdd) {
                finalDetections.add(detection)
            }
        }
        
        return finalDetections
    }
    
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionArea = Math.max(0f, Math.min(box1.right, box2.right) - Math.max(box1.left, box2.left)) *
                              Math.max(0f, Math.min(box1.bottom, box2.bottom) - Math.max(box1.top, box2.top))
        
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        
        val unionArea = box1Area + box2Area - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
    
    // ============================================
    // 결과 표시
    // ============================================
    
    private fun displayImage(bitmap: Bitmap) {
        binding.imageViewPreview.setImageBitmap(bitmap)
        binding.imageViewPreview.visibility = android.view.View.VISIBLE
    }
    
    private fun displayResults(detections: List<Detection>, processingTime: Long) {
        if (detections.isEmpty()) {
            binding.textViewStatus.text = "음식을 찾지 못했습니다 😢"
            resultAdapter.submitList(emptyList())
        } else {
            binding.textViewStatus.text = 
                "${detections.size}개 검출 (${processingTime}ms)"
            resultAdapter.submitList(detections)
        }
    }
    
    // ============================================
    // 라벨 로드
    // ============================================
    
    private fun loadLabels(): List<String> {
        return try {
            assets.open(LABELS_FILE).bufferedReader().readLines()
        } catch (e: Exception) {
            // Fallback - 기본 COCO 클래스
            listOf("person", "bicycle", "car", /* ... */)
        }
    }
    
    // ============================================
    // 권한 및 카메라
    // ============================================
    
    private fun checkPermissionAndTakePhoto() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            takePhoto()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                100
            )
        }
    }
    
    private fun takePhoto() {
        val photoFile = createImageFile()
        val photoURI = FileProvider.getUriForFile(
            this,
            "${packageName}.fileprovider",
            photoFile
        )
        currentPhotoPath = photoFile.absolutePath
        takePicture.launch(photoURI)
    }
    
    private fun createImageFile(): File {
        val timeStamp = System.currentTimeMillis().toString()
        val storageDir = getExternalFilesDir(null)
        return File.createTempFile("SCANEAT_${timeStamp}_", ".jpg", storageDir)
    }
}

// ============================================
// Data Classes
// ============================================

data class Detection(
    val className: String,
    val confidence: Float,
    val bbox: RectF
)

// ============================================
// RecyclerView Adapter
// ============================================

class ResultAdapter : RecyclerView.Adapter<ResultAdapter.ViewHolder>() {
    
    private var detections = listOf<Detection>()
    
    class ViewHolder(val binding: ItemResultBinding) : RecyclerView.ViewHolder(binding.root)
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemResultBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ViewHolder(binding)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val detection = detections[position]
        holder.binding.apply {
            textViewClassName.text = "${position + 1}. ${detection.className}"
            textViewConfidence.text = "${(detection.confidence * 100).toInt()}%"
        }
    }
    
    override fun getItemCount() = detections.size
    
    fun submitList(newDetections: List<Detection>) {
        detections = newDetections
        notifyDataSetChanged()
    }
}