import os
import cv2 as cv

# Caminho relativo que funciona localmente e em Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_FOLDER = os.path.join(BASE_DIR, "outputfile", "gerados")
def get_file(file_name, amostra, contorno):
    os.makedirs(IMG_FOLDER, exist_ok=True)
    
    full_path = os.path.join(IMG_FOLDER, file_name)    
    res_visual = cv.cvtColor(amostra, cv.COLOR_GRAY2BGR)
    cv.drawContours(res_visual, [contorno], -1, (0, 255, 0), 1)
    cv.imwrite(full_path, res_visual)