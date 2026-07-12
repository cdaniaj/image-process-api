import os
import cv2 as cv

# Caminho absoluto para bater com o volume do Compose
IMG_FOLDER = "/app/outputfile/gerados"
def get_file(file_name, amostra, contorno):
    os.makedirs(IMG_FOLDER, exist_ok=True)
    
    full_path = os.path.join(IMG_FOLDER, file_name)    
    res_visual = cv.cvtColor(amostra, cv.COLOR_GRAY2BGR)
    cv.drawContours(res_visual, [contorno], -1, (0, 255, 0), 1)
    cv.imwrite(full_path, res_visual)