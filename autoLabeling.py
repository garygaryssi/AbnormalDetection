
import numpy as np
import cv2

# save_type 0 = 원본, classes, yolo방식의 txt 저장
# 아쉽게도 다른 기능은 없음ㅋㅋ


def lamp_save(img, xy, path, name, class_name, is_saved, way=0):
    if way == 0:

        # 기본 yolo 방식의 좌표 생성
        x = round((int(xy[0]) + int(xy[2])) / 2 / img.shape[1], 6)
        y = round((int(xy[1]) + int(xy[3])) / 2 / img.shape[0], 6)
        w = round((int(xy[2]) - int(xy[0])) / img.shape[1], 6)
        h = round((int(xy[3]) - int(xy[1])) / img.shape[0], 6)

        txt_name = name.split(".")[0] + ".txt"
        path = path.replace("\\", "/")

        # path = run/detect/exp?/...jpg 무려 이름까지 저장함
        # 그래서 마지막 / 의 인덱스 값을 구하고 뒤에를 자르는거임
        idx_tmp = 0
        for i in range(0, path.count("/")):
            idx_tmp = path.index("/", idx_tmp+1)
        path = path[:idx_tmp]

        # 텍스트 파일을 지정한 경로에 만드는 작업임
        with open(f"{path}/{txt_name}", "w") as file:
            tmp = "0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
            file.write(tmp)

        # class_name 안에 기본적으로 지정한 클래스와 conf 값이 같이 저장되어 있음
        # 그래서 스플릿 하고 이름만 빼오는거임
        if is_saved == 0:
            class_name = class_name.split(" ")[0]
            with open(f"{path}/classes.txt", "w") as class_file:
                class_file.write(class_name)
