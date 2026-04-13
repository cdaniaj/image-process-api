import cv2 as cv

def get_file(file_name, amostra, contorno):
    res_visual = cv.cvtColor(amostra, cv.COLOR_GRAY2BGR)
    cv.drawContours(res_visual, [contorno], -1, (0, 255, 0), 1)
    cv.imwrite(file_name, res_visual)