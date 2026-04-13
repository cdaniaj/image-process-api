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


def get_contours_data(contours, img_shape):
    # Agora passamos o shape da imagem para achar o centro
    contorno_final = get_central_contour(contours, img_shape)
    
    if contorno_final is None:
        raise ValueError("Nenhum contorno válido encontrado próximo ao centro.")

    area_final = cv.contourArea(contorno_final)
    hull = cv.convexHull(contorno_final)
    hull_area = cv.contourArea(hull)
    solidity = area_final / hull_area if hull_area > 0 else 0
    perimetro = cv.arcLength(contorno_final, True)
    circularidade = (4 * np.pi * area_final) / (perimetro**2) if perimetro > 0 else 0
    
    return {
        "area": area_final * 0.01,  # Convertendo para mm²
        "perimetro": perimetro * 0.1,  # Convertendo para mm
        "circularity": circularidade,
        "solidity": solidity,
        "max_contour": contorno_final # Mantivemos o nome da chave para não quebrar seu main
    }