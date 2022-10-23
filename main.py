from PIL import ImageGrab
from time import time
import cv2 as cv
import numpy as np
import os
from windowCapture import WindowCapture

os.chdir(os.path.dirname(os.path.abspath(__file__)))

wincap = WindowCapture("League of Legends")


def findClickPositions(image_path, template_path, threshold=0.44, debug_mode=None):

    img = cv.imread(image_path)
    barra = cv.imread(template_path)

    player_w = barra.shape[1]
    player_h = barra.shape[0]

    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(img, barra, method)

    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))

    rectangles = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), player_w, player_h]
        rectangles.append(rect)
        rectangles.append(rect)

    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)

    points = []
    if len(rectangles):
        print('Jogador encontrado')

        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        marker_color = (0, 255, 0)
        marker_type = cv.MARKER_CROSS

        # Percorre todas as detecções e desenha um retângulo
        for (x, y, w, h) in rectangles:

            # Centro da hitbox do player
            center_x = x + int(w/2)
            center_y = y + int(h*7/2)

            # Guarda os pontos
            points.append((center_x, center_y))

            if debug_mode == "rectangles":
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv.rectangle(img, top_left, bottom_right,
                             line_color, line_type)
            elif debug_mode == "points":
                cv.drawMarker(img, (center_x, center_y),
                              marker_color, marker_type)
        if debug_mode:
            cv.imshow('Matches', img)
            cv.waitKey(0)
    else:
        print('Jogador não encontrado')


loop_time = time()
while True:
    screenshot = wincap.get_screenshot()

    cv.imshow("CV", screenshot)

    print(f"FPS {1/(time()-loop_time)}")
    loop_time = time()

    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        break
