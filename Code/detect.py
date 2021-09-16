import argparse
import os
import platform
import shutil
import time
import numpy as np
from numpy import random
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, plot_area, strip_optimizer
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    max_speed = 20
    last = 0
    flag_red = 0
    flag_danger = 0
    flag_peaton = 0
    
    sem = [0,0,0,0,0,0]
    last_sem = [0,0,0,0,0,0]
    current_frame_sem = 0
    last_frame_sem = 0
    
    vel = [0,0,0,0,0,0,0,0,0]
    last_vel = [0,0,0,0,0,0,0,0,0]
    current_frame_vel = 0
    last_frame_vel = 0
    
    sigDer = [0,0,0]
    last_sigDer = [0,0,0]
    current_frame_sigDer = 0
    last_frame_sigDer = 0
    
    sigIzq = [0,0,0]
    last_sigIzq = [0,0,0]
    current_frame_sigIzq = 0
    last_frame_sigIzq = 0
    
    out, source, weights1, weights2, weights3, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights1, opt.weights2, opt.weights3, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_1 = attempt_load(weights1, map_location=device)  # load FP32 model
    model_2 = attempt_load(weights2, map_location=device)  # load FP32 model
    model_3 = attempt_load(weights3, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model_1.stride.max())  # check img_size
    if half:
        model_1.half()  # to FP16
        model_2.half()  # to FP16
        model_3.half()  # to FP16
        
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names_1 = model_1.module.names if hasattr(model_1, 'module') else model_1.names
    colors_1 = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names_1))]
    
    names_2 = model_2.module.names if hasattr(model_2, 'module') else model_2.names
    colors_2 = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names_2))]
    
    names_3 = model_3.module.names if hasattr(model_3, 'module') else model_3.names
    colors_3 = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names_3))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model_1(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        
        current_frame_sem = dataset.frame
        current_frame_vel = dataset.frame
        current_frame_sigIzq = dataset.frame
        current_frame_sigDer = dataset.frame
        flag_peaton = 0
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred_1 = model_1(img, augment=opt.augment)[0]
        pred_2 = model_2(img, augment=opt.augment)[0]
        pred_3 = model_3(img, augment=opt.augment)[0]

        # Apply NMS
        pred_1 = non_max_suppression(pred_1, 0.6, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms) #COCO
        pred_2 = non_max_suppression(pred_2, 0.7, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms) #SIGNAL
        pred_3 = non_max_suppression(pred_3, 0.3, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms) #SEMFORO
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred_1 = apply_classifier(pred_1, modelc, img, im0s)
            pred_2 = apply_classifier(pred_2, modelc, img, im0s)
            pred_3 = apply_classifier(pred_3, modelc, img, im0s)
            
        # Print max speed
        slide = Image.open('slides/' + str(max_speed) + '.png')
        im0s = cv2.cvtColor(im0s,cv2.COLOR_BGR2RGB)
        im0s = Image.fromarray(im0s)
        im0s.paste(slide, (0,0) ,slide)
        im0s = np.array(im0s)
        im0s = cv2.cvtColor(im0s,cv2.COLOR_RGB2BGR)
        
        # Plot danger area
        plot_area(im0s)
        
        # Process detections SEMAFOROS ------------------------------------------------------------------------------
        for i, det in enumerate(pred_3):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names_3[int(c)])  # add to string
                
                for *xyxy, conf, cls in det:
                    sem[int(cls)] = sem[int(cls)]+1
                    if sem[int(cls)] > 15: #15
                        name = names_3[int(cls)]
                        slide = Image.open('slides/' + str(name) + '.png')
                        im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)
                        im0 = Image.fromarray(im0)
                        im0.paste(slide, (0,0) ,slide)
                        im0 = np.array(im0)
                        im0 = cv2.cvtColor(im0,cv2.COLOR_RGB2BGR)
                        if name in ['rojo','rojoIzq'] and flag_red == 0:
                            flag_red = 1
                            last = max_speed
                            max_speed = 0
                        if name in ['verde','ambar','verdeIzq','OFF'] and flag_red == 1:
                            flag_red = 0
                            max_speed = last
    
                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names_3[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors_3[int(cls)], line_thickness=3, name = names_3[int(cls)])
            
        if (last_frame_sem + 1) == current_frame_sem:
            last_frame_sem = current_frame_sem
            for i in range(0,len(sem)):
                if last_sem[i] == sem[i]:
                    last_sem[i] = 0
                    sem[i] = 0
                else:
                    last_sem[i] = sem[i]
        else:
            last_frame_sem = current_frame_sem
            for i in range(0,len(sem)):
                last_sem[i] = 0
                sem[i] = 0  

        # Process detections COCO --------------------------------------------------------------------------------
        for i, det in enumerate(pred_1):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                labels_coco = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']
                for c in det[:, -1].unique():
                    if names_1[int(c)] in labels_coco:
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names_1[int(c)])  # add to string
                        
                for *xyxy, conf, cls in det:
                    name = names_1[int(cls)]
                    if name in labels_coco:
                        c1 = (int(xyxy[0]), int(xyxy[1]))
                        c2 = (int(xyxy[2]), int(xyxy[3]))
                        area = abs((c2[0] - c1[0])*(c2[1] - c1[0]))
                        center = (int((c2[0]+c1[0])/2), int((c2[1]+c1[1])/2))
                        
                        if (500<center[0]<950 and 500<center[1]<720): # Peligro proximidad
                            if area > 60000: #60.000
                                slide = Image.open('slides/' + 'danger' + '.png')
                                im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)
                                im0 = Image.fromarray(im0)
                                im0.paste(slide, (0,0) ,slide)
                                im0 = np.array(im0)
                                im0 = cv2.cvtColor(im0,cv2.COLOR_RGB2BGR)
                                max_speed = 0
                                
                        if (200<center[0]<1080 and 400<center[1]<720): # Paso peatones
                            if (area > 40000) and (name == 'person'): #40.000
                                flag_peaton = 1
    
            
                # Write results
                for *xyxy, conf, cls in det:
                    if names_1[int(cls)] in labels_coco:
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names_1[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors_1[int(cls)], line_thickness=3, name=names_1[int(cls)])
    
        # Process detections SIGNALS ------------------------------------------------------------------------------
        for i, det in enumerate(pred_2):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names_2[int(c)])  # add to string
                    
                # Print signs
                list = ['ceda', 'stop', 'paso']
                velocity = ['20','30','40','50','60','80','90','100','120']
                
                for *xyxy, conf, cls in det:
                    name = names_2[int(cls)]
                    if name in velocity:
                        vel[int(cls)-6] = vel[int(cls)-6]+1
                    if (name in velocity) and (vel[int(cls)-6] > 15):
                        max_speed = int(name)

                    if name in list:
                        c1 = (int(xyxy[0]), int(xyxy[1]))
                        c2 = (int(xyxy[2]), int(xyxy[3]))
                        area = abs((c2[0] - c1[0])*(c2[1] - c1[0]))
                        center = (int((c2[0]+c1[0])/2), int((c2[1]+c1[1])/2))
                        if center[0] < 640:
                            sigIzq[int(cls)-15] = sigIzq[int(cls)-15]+1
                            if sigIzq[int(cls)-15] > 20: #20
                                slide = Image.open('slides/' + str(name) + 'Izq.png')
                                im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)
                                im0 = Image.fromarray(im0)
                                im0.paste(slide, (0,0) ,slide)
                                im0 = np.array(im0)
                                im0 = cv2.cvtColor(im0,cv2.COLOR_RGB2BGR)
                                if (area > 30000) and (name == 'stop'): #30.000
                                    max_speed = 0
                                if (name == 'paso') and (flag_peaton > 0):
                                    max_speed = 0
                                
                        if center[0] > 640:
                            sigDer[int(cls)-15] = sigDer[int(cls)-15]+1
                            if sigDer[int(cls)-15] > 20: #20
                                slide = Image.open('slides/' + str(name) + 'Der.png')
                                im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2RGB)
                                im0 = Image.fromarray(im0)
                                im0.paste(slide, (0,0) ,slide)
                                im0 = np.array(im0)
                                im0 = cv2.cvtColor(im0,cv2.COLOR_RGB2BGR)
                                if (area > 30000) and (name == 'stop'): #30.000
                                    max_speed = 0
                                if (name == 'paso') and (flag_peaton > 0):
                                    max_speed = 0
                               

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names_2[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors_2[int(cls)], line_thickness=3, name = names_2[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
        
        if (last_frame_vel + 1) == current_frame_vel:
            last_frame_vel = current_frame_vel
            for i in range(0,len(vel)):
                if last_vel[i] == vel[i]:
                    last_vel[i] = 0
                    vel[i] = 0
                else:
                    last_vel[i] = vel[i]
        else:
            last_frame_vel = current_frame_vel
            for i in range(0,len(vel)):
                last_vel[i] = 0
                vel[i] = 0
        
        if (last_frame_sigIzq + 1) == current_frame_sigIzq:
            last_frame_sigIzq = current_frame_sigIzq
            for i in range(0,len(sigIzq)):
                if last_sigIzq[i] == sigIzq[i]:
                    last_sigIzq[i] = 0
                    sigIzq[i] = 0
                else:
                    last_sigIzq[i] = sigIzq[i]
        else:
            last_frame_sigIzq = current_frame_sigIzq
            for i in range(0,len(sigIzq)):
                last_sigIzq[i] = 0
                sigIzq[i] = 0
        
        if (last_frame_sigDer + 1) == current_frame_sigDer:
            last_frame_sigDer = current_frame_sigDer
            for i in range(0,len(sigDer)):
                if last_sigDer[i] == sigDer[i]:
                    last_sigDer[i] = 0
                    sigDer[i] = 0
                else:
                    last_sigDer[i] = sigDer[i]
        else:
            last_frame_sigDer = current_frame_sigDer
            for i in range(0,len(sigDer)):
                last_sigDer[i] = 0
                sigDer[i] = 0
                
        # Stream results ---------------------------------------------------------------------------------------
        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights1', nargs='+', type=str, default='weights/COCO.pt', help='model.pt path(s)')
    parser.add_argument('--weights2', nargs='+', type=str, default='weights/SEÃƒâ€˜AL.pt', help='model.pt path(s)')
    parser.add_argument('--weights3', nargs='+', type=str, default='weights/SEMAFARO.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='videos/dia.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights1 in ['weights/COCO.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights1)
        else:
            detect()