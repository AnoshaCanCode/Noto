# document scanning implementation with OpenCV

# 3. Perspective Correction
# since OpenCV needs the 4 corners in a very strict order [top-left, top-right, bottom-right, bottom-left]
# we will first force any 4 coordinates into that order

import numpy as np
import cv2

def order_points(pts): # take a list of 4 coordinates & order them
    rect = np.zeros((4, 2), dtype="float32") #OpenCV requires float32 for transforms

    # math logic: the top-left point will have the smallest sum (x + y),
    # the bottom-right point will have the largest sum (x + y).
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left
    rect[2] = pts[np.argmax(s)] # bottom-right

    # calculate the difference between the points (y - x)
    # top-right point will have the smallest difference, bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left

    return rect

# now we to map the skewed document to a flat rectangle
# this next function will calculate the true width & height of the paper, create a destination rectangle, 
# and use OpenCV's built-in getPerspectiveTransform and warpPerspective tools to mathematically stretch the pixels into place

def four_point_transform(image, pts): # get an image and a set of 4 corners
    rect = order_points(pts) # consistent order of the points 
    (tl, tr, br, bl) = rect

    # compute width of our new flat image
    # find the max distance between the bottom corners or the top corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute height of our new flat image
    # find the max distance between the right corners or the left corners
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct the set of destination points
    # perfect rectangle, starting at (0,0) and going to max width/height
    dst = np.array([
        [0, 0],                          # top-left
        [maxWidth - 1, 0],               # top-right
        [maxWidth - 1, maxHeight - 1],   # bottom-right
        [0, maxHeight - 1]               # bottom-left
    ], dtype="float32")

    # calculate the transform matrix - the map of how to squish and stretch the image
    M = cv2.getPerspectiveTransform(rect, dst)

    # apply the matrix to the actual image to create the final top-down view
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped # return the flattened document! 


# 1. Image Preprocessing & Edge Detection
def detect_document(image_path): # find doc in an image & apply the perspective warp
    image = cv2.imread(image_path) # load the raw image

    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    
    orig = image.copy()
    
    # convert to Grayscale (color makes the math slower)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # apply Gaussian Blur
    # the (5,5) kernel blurs the image slightly to remove high-frequency noise (like film grain)
    # so the edge detector doesn't get confused by tiny specks
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # canny edge detection - find the actual outlines
    edged = cv2.Canny(blurred, 75, 200) # 75 and 200 are the lower and upper thresholds for what counts as an edge

    # now we have an image containing just the sharp white outlines (edges) on a black background
    
    # 2. Contour Detection
    # find the actual connected shapes (contours) within those lines

    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort from largest to smallest & only keep the top 5 largest shapes (assume doc is the main subject)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    document_contour = None
    
    for c in contours:
        perimeter = cv2.arcLength(c, True)

        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True) # 0.02 * perimeter - precision threshold (how closely the math should hug the original jagged line)
        # True -> we expect a closed shape
        
        if len(approx) == 4: # if the approximated contour has exactly 4 points, we found our doc
            document_contour = approx
            break
            
    # safety check: if no 4-point shape was found, we stop here
    if document_contour is None:
        raise ValueError("Could not find a document outline in the image.")

    # reshape from (4, 1, 2) to (4, 2) so the transform function can read it
    pts = document_contour.reshape(4, 2)

    warped = four_point_transform(orig, pts)
    
    return warped