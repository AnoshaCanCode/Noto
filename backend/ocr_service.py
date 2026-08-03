import cv2
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

class SuryaOCRService:
    def __init__(self):
        print("Initializing Surya OCR models...")
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()

    def extract_layout(self, image_cv2): # accepts an OpenCV image (numpy array), converts it, extracts text, bounding boxes, layout structures

        # convert OpenCV BGR image to PIL RGB Image
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        predictions = self.recognition_predictor([pil_image], [self.detection_predictor])

        # parse results into a clean JSON-serializable structure
        page_result = predictions[0]
        extracted_blocks = []

        for item in page_result.text_lines:
            block = {
                "text": item.text,
                "confidence": item.confidence,
                "bbox": item.bbox  # [x1, y1, x2, y2] coordinate format
            }
            extracted_blocks.append(block)

        return {
            "image_size": {"width": pil_image.width, "height": pil_image.height},
            "text_blocks": extracted_blocks
        }

# singleton instance to avoid reloading heavy weights on every API call
ocr_service = SuryaOCRService()