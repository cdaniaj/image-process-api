import cv2 as cv
import numpy as np

def normalize_image(image, p_min, p_max):
    min, max = np.percentile(image, (p_min, p_max))
    image_estavel = np.clip(image, min, max)
        
    image = cv.normalize(image_estavel, None, 0, 255, cv.NORM_MINMAX)
    image = np.uint8(image)
    return image
