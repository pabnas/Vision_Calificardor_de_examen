import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
from scipy.spatial.distance import euclidean
from image_transformer import ImageTransformer

def find_puntos(img):
    plt.figure(1)
    plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.draw()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,50])
    upper = np.array([180,25,255])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img,img, mask= mask)
    img = res
    plt.figure(2)
    plt.imshow(res,cmap='gray')
    plt.draw()

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=15)
    img = cv2.dilate(img, kernel, iterations=30)
    img = cv2.erode(img, kernel, iterations=20)
    _,img = cv2.threshold(img,10,255,cv2.THRESH_BINARY)

    plt.figure(3)
    plt.imshow(img,cmap='gray')
    plt.draw()

    cnts = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        # if our approximated contour has four points, then we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    puntos = []
    for x in screenCnt:
        a = x[0]
        a = [a[0], a[1]]
        puntos.append(a)
    #print(puntos)
    return screenCnt,puntos

def click_and_crop(event, x, y, flags, param):
    global puntos, contador
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append((x, y))
        contador += 1

contador = 0
puntos = []

def click_puntos(img):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        cv2.imshow("image", img)
        cv2.waitKey(1)
        if contador == 4:
            break
    cv2.destroyAllWindows()
    return puntos

def tralacion(img,puntos):
    pts1 = np.float32(puntos)
    max = 600
    pts2 = np.float32([[0, 0], [max, 0], [max, max], [0, max]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (max, max))
    imgBuilding2 = img.copy()
    cv2.circle(imgBuilding2, (pts1[0][0], pts1[0][1]), 5, 255, -1)
    cv2.circle(imgBuilding2, (pts1[1][0], pts1[1][1]), 5, 255, -1)
    cv2.circle(imgBuilding2, (pts1[2][0], pts1[2][1]), 5, 255, -1)
    cv2.circle(imgBuilding2, (pts1[3][0], pts1[3][1]), 5, 255, -1)
    return imgBuilding2,dst

def match(img,template,threshold = 0.8):
    w, h = template.shape[::-1]
    ultimo = (0,0)
    img1 = img.copy()
    img_marcada = img.copy()
    puntos = []
    res = cv2.matchTemplate(img1, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        #print(pt)
        if euclidean(ultimo, pt) > 20:
            puntos.append(pt)
            ultimo = pt
            cv2.rectangle(img_marcada, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    #print(puntos)
   # puntos = sorted(puntos , key=lambda k: [k[0], k[1]])
   # temp = []
    #ultimo = (0,0)
    #for pt in puntos:
     #   if euclidean(ultimo,pt)>20:
    #        temp.append(pt)
     #       ultimo = pt
    #puntos = temp
    #print(puntos)
    return img1,puntos,img_marcada

def marcado(img):
    preguntas = []
    centro = (85, 168)
    distancia = (50, 80)
    for pregunta in range(0, 5):
        temp = []
        for linea in range(0, 4):
            x = centro[0] + (distancia[0] * linea)
            y = centro[1] + (distancia[1] * pregunta)
            preg = img[y][x]
            if preg == 255:
                preg = 1
            temp.append(preg)
        preguntas.append(temp)
    centro = (406, 168)
    distancia = (50, 80)
    for pregunta in range(0, 5):
        temp = []
        for linea in range(0, 4):
            x = centro[0] + (distancia[0] * linea)
            y = centro[1] + (distancia[1] * pregunta)
            preg = img[y][x]
            if preg == 255:
                preg = 1
            temp.append(preg)
        preguntas.append(temp)
    return preguntas


if __name__ == "__main__":
    #img_original = cv2.imread('7.jpeg')
    img_original = cv2.imread('1.jpg')
    img_a = cv2.imread('A.png',0)
    img_x = cv2.imread('Cruz.jpg',0)
    img = img_original
    try:
        screenCnt, puntos = find_puntos(img)
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
        plt.figure(4)
        plt.imshow(img, cmap='gray')
        plt.draw()
    except:
        puntos = click_puntos(img)

    puntos = list(reversed(puntos))
    #print(puntos)

    img_antes,img = tralacion(img_original,puntos)
    plt.figure(5)
    plt.subplot(121)
    plt.imshow(img_antes, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.title('Projective'), plt.xticks([]), plt.yticks([])
    plt.draw()
    it = ImageTransformer(img, None)
    for grado in range(0,360,90):
        rotated_img = it.rotate_along_axis(theta=0, phi=0, gamma=grado, gray=1)
        img_rotada , puntos ,_ = match(rotated_img,img_a,threshold=0.65)
        if len(puntos)>0:
            break


    rotated_img = cv2.adaptiveThreshold(rotated_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 10)
    img_rotada, puntos,img_marcada = match(rotated_img, img_x, threshold=0.7)

    temp = []
    for a in puntos:
        temp.append((a[0]+20,a[1]+20))
    puntos = temp
    #print(puntos)

    plt.figure(6)
    plt.imshow(img_marcada, cmap='gray')
    plt.draw()

    if puntos[2][0]<puntos[3][0]:
        temp = puntos[2]
        puntos[2] = puntos[3]
        puntos[3] = temp
    #print(puntos)
    img_antes,img = tralacion(img_rotada,puntos)
    plt.figure(7)
    plt.imshow(img, cmap='gray')
    plt.draw()

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 10)
    plt.figure(8)
    plt.imshow(img, cmap='gray')
    plt.draw()

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=8)
    img = cv2.erode(img, kernel, iterations=12)
    plt.figure(9)
    plt.imshow(img, cmap='gray')
    plt.draw()
    preguntas = marcado(img)
    print(preguntas)

    #f = open(args["answers"])
    f = open('answers.txt')
    puntaje = 0
    for a in range(0, 10):
        respuesta = f.readline()
        respuesta = respuesta[-2:]
        respuesta = respuesta[0]

        if sum(preguntas[a]) < 3:
            marcado = "NOT_VALID"
            answer = "FAIL"
        else:
            if preguntas[a][0] == 0:
                marcado = "A"
            elif preguntas[a][1] == 0:
                marcado = "B"
            elif preguntas[a][2] == 0:
                marcado = "C"
            elif preguntas[a][3] == 0:
                marcado = "D"
            else:
                marcado = "EMPTY"
            if respuesta == marcado:
                puntaje += 1
                answer = "OK"
            else:
                answer = "FAIL"
        print(str(a + 1) + ". Answer: " + marcado + ", Correct: " + respuesta + ", " + answer)
    print("Score: " + str(puntaje) + "/10 = " + str((puntaje / 10) * 5))

    plt.show()
