import cv2 as cv
import numpy as np

def apply_morphology(thresh):
    kernel = np.ones((5,5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    return thresh