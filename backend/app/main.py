import base64
import os
import cv2
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.scanner_service import detect_document
from backend.ocr_service import ocr_service
from backend.vector_db import index_ocr_blocks

app = FastAPI(title="Noto API", version="0.1.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Noto Spatial Engine Active"}

@app.post("/scan")
async def scan_document(file: UploadFile = File(...)):
    contents = await file.read()
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)
        
    try:
        # OpenCV perspective warping
        warped_image = detect_document(temp_path)
        
        # Extract OCR layout using your service
        ocr_json_result = ocr_service.extract_layout(warped_image)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Encode warped image to base64 string for UI rendering
    _, encoded_img = cv2.imencode('.jpg', warped_image)
    base64_image = base64.b64encode(encoded_img).decode('utf-8')

    # Index OCR text blocks into Qdrant Cloud
    page_id = f"doc_{uuid.uuid4().hex[:8]}"
    text_blocks = ocr_json_result.get("text_blocks", [])
    indexed_count = index_ocr_blocks(text_blocks, page_id=page_id)

    return {
        "page_id": page_id,
        "indexed_blocks": indexed_count,
        "image_base64": f"data:image/jpeg;base64,{base64_image}",
        "ocr_data": ocr_json_result
    }