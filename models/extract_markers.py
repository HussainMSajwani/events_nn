import numpy as np
import os
from pathlib import Path
import cv2

FRAMES_DIR = Path('rotate_circle/frames')

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
    #use a blob detector to find circles in thresholded image and draw them
    #fit a cannny edge detector to the gray image
    edges = cv2.Canny(gray, 100, 100)
    print(edges)
    return edges
    """
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 350
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)
    im_with_keypoints = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return im_with_keypoints
    """

for frame_id in sorted(os.listdir(FRAMES_DIR)):
    img = cv2.imread(str(FRAMES_DIR / frame_id))[25:215, 85:265]
    cv2.imshow('og', img)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = preprocess_image(img)
    cv2.imshow('img', img)
    key = cv2.waitKey()
    if key & 0xFF == ord('q'):
        break
    else:
        continue


cv2.destroyAllWindows()   
