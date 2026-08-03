import base64
import os
import cv2
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from backend.scanner_service import detect_document
from backend.ocr_service import ocr_service

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
    # read the uploaded file bytes
    contents = await file.read()
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)
        
    try:
        warped_image = detect_document(temp_path)
        ocr_json_result = ocr_service.extract_layout(warped_image) # add ocr
    finally:
        # clean up the temporary file so our server doesn't get cluttered
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # encode warped image to base64 string to send inside JSON 
    _, encoded_img = cv2.imencode('.jpg', warped_image)
    base64_image = base64.b64encode(encoded_img).decode('utf-8')
    
    return JSONResponse(content={
        "image_base64": f"data:image/jpeg;base64,{base64_image}",
        "ocr_data": ocr_json_result
    })