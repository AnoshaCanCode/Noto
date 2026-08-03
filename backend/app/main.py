import os
import cv2
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from backend.scanner_service import detect_document

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
    finally:
        # clean up the temporary file so our server doesn't get cluttered
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # convert warped OpenCV array back into a JPEG format 
    _, encoded_img = cv2.imencode('.jpg', warped_image)
    
    image_stream = io.BytesIO(encoded_img.tobytes())
    
    # return the image directly in the HTTP response
    return StreamingResponse(image_stream, media_type="image/jpeg")