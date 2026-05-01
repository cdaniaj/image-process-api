import cv2 as cv
import numpy as np

def get_contours(thresh):
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_central_contour(contours, img_shape):
    # 1. Encontra o centro do crop (ROI)
    altura, largura = img_shape
    centro_img_x = largura // 2
    centro_img_y = altura // 2
    
    melhor_contorno = None
    menor_distancia = float('inf')
    
    for cnt in contours:
        # 2. Ignora ruídos minúsculos (pó branco gerado pelo threshold)
        if cv.contourArea(cnt) < 50:
            continue
            
        # 3. Calcula o centro de massa do contorno atual
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            
        # 4. Calcula a distância euclidiana entre o contorno e o centro da imagem
        distancia = np.sqrt((cX - centro_img_x)**2 + (cY - centro_img_y)**2)
        
        # 5. Mantém apenas o que for mais próximo do centro
        if distancia < menor_distancia:
            menor_distancia = distancia
            melhor_contorno = cnt
            
    return melhor_contorno

    
def get_interface_contour_data(contorno_final):
    area_final = cv.contourArea(contorno_final)
    hull = cv.convexHull(contorno_final)
    hull_area = cv.contourArea(hull)
    solidity = area_final / hull_area if hull_area > 0 else 0
    perimetro = cv.arcLength(contorno_final, True)
    circularidade = (4 * np.pi * area_final) / (perimetro**2) if perimetro > 0 else 0

    return {
        "area": area_final,
        "perimeter": perimetro,
        "circularity": circularidade,
        "solidity": solidity
    }
    
def get_train_contour_data(contorno_final):
    # 1. Área e Perímetro (Base para tudo)
    area_final = cv.contourArea(contorno_final)
    perimetro = cv.arcLength(contorno_final, True)
    
    # 2. Compactness (Fórmula do Dataset: perimeter^2 / area - 1.0)
    # Nota: No dataset, isso mede a irregularidade. 
    compactness = (perimetro**2 / (area_final)) - 1.0 if area_final > 0 else 0
    
    # 3. Concavity (Mapeado via Solidez)
    # O dataset mede concavidade. No OpenCV, a solidez é o oposto.
    hull = cv.convexHull(contorno_final)
    hull_area = cv.contourArea(hull)
    solidity = area_final / hull_area if hull_area > 0 else 0
    concavity = 1 - solidity # Aproximação para bater com o dataset
    
    # 4. Radius Mean (Raio médio aproximado)
    (x, y), radius = cv.minEnclosingCircle(contorno_final)

    return {
       "radius_mean": radius,
        "perimeter_mean": perimetro,
        "area_mean": area_final,
        "compactness_mean": compactness,
        "concavity_mean": concavity,
        "max_contour": contorno_final
    }

def get_contours_data(contours, img_shape):
    contorno_final = get_central_contour(contours, img_shape)
    
    if contorno_final is None:
        raise ValueError("Nenhum contorno válido encontrado.")

    get_train_data = get_train_contour_data(contorno_final)
    get_interface_data = get_interface_contour_data(contorno_final)
    return {
        "train_data": get_train_data,
        "interface_data": get_interface_data
    }