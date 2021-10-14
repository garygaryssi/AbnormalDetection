import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from autoLabeling import lamp_save
import os
from Area_check import area_check


def detect(source: str, name='exp', weights='best_rain.pt', savetype=0, savelabel=False, save_img=False,
           save_path='static/'):

    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    opt.img_size = 640  # 기본적인 이미지 사이즈, 학습 기준으로 설정할 것
    opt.conf_thres = 0.25  # 아마 디텍트 퍼센트 한계치인듯, 찾고 나서 확률이 0.25 보다 낮으면 삭제
    opt.iou_thres = 0.45  # 넌 또 누구냐
    opt.view_img = False  # True 하면 진행되고 있는 사진 이미지를 보여줌
    opt.save_conf = True
    opt.nosave = False  # True 하면 사진 저장 안함
    opt.device = "0"  # 0 = gpu
    opt.classes = None  # 실험하진 않았는데.. 만약 넣으면 해당 클래스만 출력하는 듯?
    opt.agnostic_nms = True
    opt.augment = True
    opt.exist_ok = True

    # 수정 할 애들
    opt.project = save_path  # 저장 경로를 의미하는 듯
    opt.source = source  # 불러올 데이터 경로
    opt.name = name  # 저장할 이름
    opt.weights = weights  # 가지고 올 pt
    save_type = savetype
    opt.save_txt = savelabel  # 나온 라벨도 저장할래?

    check_requirements(exclude=('pycocotools', 'thop'))

    # @yun detect 실행시 static내에 directory 계속해서 생성해냄
    fol = os.listdir(opt.project)
    # print(fol, "fol cheking ************")
    nameLen = len(opt.name)
    # print(nameLen, " nameLen cheking***********")
    fornum = 1
    while opt.name in fol:
        if fornum == 1:
            opt.name = opt.name+f"{fornum}"
        else:
            opt.name = opt.name[:nameLen]+f"{fornum}"
        fornum += 1
    # print(opt.name)

    with torch.no_grad():
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    xyxy_tmp = []

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results # 여기서 클래스 조정할것
                    for *xyxy, conf, cls in reversed(det):
                        # @yun xyxy 좌표 저장
                        xyxy_tmp.append([int(x) for x in xyxy])
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            # @yun detect_original 기능
                            if save_type == 0:
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results # 이미지크기 조절할것
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                # detect된것 저장
                if save_img:
                    if dataset.mode == 'image':
                        s_ = s.split(" ")
                        try:

                            s_[2]

                            # 0 == 원본(detect.py)
                            if save_type == 0:
                                cv2.imwrite(save_path, im0)

                            # 1 == Auto_labeling
                            if save_type == 1:
                                cv2.imwrite(save_path, im0)
                                lamp_save(im0, xyxy, save_path, p.name, 'headlamp', 0, 0)



                            # 2 == 검출된 좌표 merge
                            if save_type == 2:
                                im1, area = area_check(im0, xyxy_tmp, 1)
                                cv2.imwrite(save_path, im1)

                            # 3 == 크롭된 사진 저장, 가장 좋은 정확도 기준
                            elif save_type == 3:
                                roi = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                                cv2.imwrite(save_path, roi)

                            # 4 == result save
                            elif save_type == 4:
                                im3, area = area_check(im0, xyxy_tmp, 3)

                                # 면적계산
                                # text = f"{round(area, 3) * 100}%"
                                # cv2.putText(im3, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                #             1, (0, 255, 0), 1)

                                # static 폴더 내에 rain 안에 result 저장하기위해 슬라이싱 @yun
                                cv2.imwrite("static/result.jpg", im3)
                                cv2.imwrite(save_path[:-8] + "result.jpg", im3)

                        except:
                            print("not found xyxy!!")
                            cv2.imwrite(save_path, im0)


                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        print(s_)

    return save_dir


if __name__ == "__main__":

    detect(source='./data/DLproduct/TestSet/Detected/', weights='weights/best_product.pt', name="product",
           savetype=1, save_path="runs/")
    pass

    # detect(source='C:/Users/AI-00/Desktop/1007/', weights='weights/120e_32b.pt', name="exp",
    #        savetype=3, save_path="runs/1007")
    # pass

    # path = detect.detect(source='C:/Users/AI-00/Desktop/1007/',
    #                      weights='weights/120e_32b.pt',
    #                      name="exp",
    #                      savetype=0,
    #                      save_path="runs/1007"
    #                      )

    # path = str(path)
    #
    #
    # detect.detect(source=path,
    #               weights="weights/best_rain.pt",
    #               name="exp",
    #               save_path="runs/final")
