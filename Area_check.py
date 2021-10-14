import cv2 as cv
import numpy as np
from tqdm import tqdm



def area_check(img, xyxy, arg):
    tmp_img = np.zeros(img.shape[0:2], dtype=np.uint8)
    area = 0

    # cx = [-1, -1, 0, 1, 1, 1, 0, -1]
    # cy = [0, 1, 1, 1, 0, -1, -1, -1]
    cx = [-1, 0, 1, 0]
    cy = [0, 1, 0, -1]

    for i in xyxy:
        tmp_img[i[1]:i[3], i[0]:i[2]] = 255

    for y in tqdm(range(img.shape[0])):
        for x in range(img.shape[1]):

            if tmp_img[y, x] != 255:
                continue

            if tmp_img[y, x] == 255:
                area += 1

            for i in range(4):
                nx = x + cx[i]
                ny = y + cy[i]

                if nx < 0 or ny < 0 or nx >= img.shape[1] or ny >= img.shape[0]:
                    continue

                if tmp_img[ny, nx] == 0 and tmp_img[y, x] == 255:
                    if arg == 1:
                        img[y - 1:y + 1, x - 1:x + 1] = [0, 255, 0]
                    if arg == 3:
                        img[y - 1:y + 1, x - 1:x + 1] = [0, 0, 255]
                    break

    full_area = img.shape[0] * img.shape[1]
    water_percentage = area / full_area

    return img, water_percentage

if __name__=="__main__":
    cv.imread()