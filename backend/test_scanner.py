import cv2
from scanner_service import detect_document

image_path = "test_img.jpg"

try:
    # run perspective transform
    scanned_image = detect_document(image_path)
    
    # save or display result
    cv2.imwrite("scanned_result.jpg", scanned_image)
    print("Success! Scanned document saved as 'scanned_result.jpg'.")

except Exception as e:
    print(f"Error during scanning: {e}")