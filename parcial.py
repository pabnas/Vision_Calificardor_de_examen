import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def click_and_crop(event, x, y, flags, param):
    global puntos, contador
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append((x, y))
        contador += 1

contador = 0
puntos = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--answers", required = True, help = "Archivo de las respuestas")
    ap.add_argument("-i", "--image", required = True, help = "Imagen a calificar")
    args = vars(ap.parse_args())
    img = cv2.imread(args["image"])
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        cv2.imshow("image", img)
        cv2.waitKey(1)
        if contador == 4:
            break
    #print(puntos)
    cv2.destroyAllWindows()
    pts1 = np.float32(puntos)
    max = 600
    pts2 = np.float32([[0, 0], [max, 0], [max, max], [0, max]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (max, max))
    plt.subplot(121)
    imgBuilding2 = img.copy()
    cv2.circle(imgBuilding2, (pts1[0][0], pts1[0][1]), 5, 255, -1)
    cv2.circle(imgBuilding2, (pts1[1][0], pts1[1][1]), 5, 255, -1)
    cv2.circle(imgBuilding2, (pts1[2][0], pts1[2][1]), 5, 255, -1)
    cv2.circle(imgBuilding2, (pts1[3][0], pts1[3][1]), 5, 255, -1)
    plt.figure(1)
    plt.imshow(imgBuilding2, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(dst, cmap='gray')
    plt.title('Projective'), plt.xticks([]), plt.yticks([])
    plt.draw()
    img = dst
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 10)
    plt.figure(2)
    plt.imshow(img, cmap='gray')
    plt.draw()
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=8)
    img = cv2.erode(img, kernel, iterations=12)
    plt.figure(3)
    plt.imshow(img, cmap='gray')
    plt.draw()
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
    #print(preguntas)
    f = open(args["answers"])
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

if __name__ == "__main__":
    main()
