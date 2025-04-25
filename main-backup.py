# import python library ↓
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QMessageBox
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal, QTime, QDateTime
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from pathlib import Path
import sys
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2
import modbus_rtu
import modbus_tcp
import _thread
import serial
# import private library ↓
from sql_folder import SQL_write
from ui_files.main_win import Ui_mainWindow
from ui_files.dialog.rtsp_win import Window
from ui_files.setting_form import Ui_setting
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam, LoadStreams
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, clean_str
from utils.plots import Annotator, colors, save_one_box, plot_one_box

from utils.torch_utils import select_device, time_sync, load_classifier
from utils.capnums import Camera
from sql_folder.SQL_write import writesql, closesql


# ↓ set global variable 设置全局变量  ↓
results = None
modbus_flag = Falseresults = []
okCounter = 0
ngCounter = 0
loopCounter = 0
output_box_list = [0, 0]
emit_frame_flag = True  # 测试传输frame到UI ,对检查循环速度的影响

# 光电传感器状态初始化，默认为不活跃
current_state = False  # 初始状态为不活跃
# 光电传感器COM口设定
ser2 = None
ret2 = None
feedback_data_D3 = None

# Initialization function "SQL_is_open", SQL writing disabled by default //初始化函数：SQL_is_open，默认不开启SQL写入
SQL_is_open = False
sensor_is_open = False
modbus_tcp_ip = "192.168.3.110"  # Modbus 服务器 IP ,可使用 sokit等串口调试工具建立一个 TCP服务器 地址
modbus_tcp_port = 502  # 默认 Modbus TCP 端口
sql_server_ip = '172.18.136.183'  # 不在一个域内的账户 只能使用IP 连接SQL服务器 , 不可使用计算机名称, SUMITOMORIKO/***
area_dic = {}
area_min = None

# ↓ Class detector  检测类 created by yoloV5
class DetThread(QThread):  # 检测功能主线程  继承 QThread
    # 定义PYQT信号
    send_img_ch0 = pyqtSignal(np.ndarray)  # ## CH0 output image
    send_img_ch1 = pyqtSignal(np.ndarray)  # ## CH1 output image
    send_img_ch2 = pyqtSignal(np.ndarray)  # ## CH2 output image
    send_img_ch3 = pyqtSignal(np.ndarray)  # ## CH3 output image
    send_img_ch4 = pyqtSignal(np.ndarray)  # ## CH4 output image
    send_img_ch5 = pyqtSignal(np.ndarray)  # ## CH5 output image
    send_statistic = pyqtSignal(dict)  # ##
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.vid_cap = None  # 240229
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.device = '0'
        self.conf_thres = 0.50  # 置信度 阈值
        self.iou_thres = 0.45  # iou 阈值
        self.jump_out = False  # jump out of the loop
        self.is_continue = False  # continue/pause
        self.percent_length = 1000  # progress bar
        self.rate_check = True  # Whether to enable delay
        self.rate = 100
        self.save_folder = None  ####'./auto_save/jpg'
        self.pred_flag = False  # pred_CheckBox

    @torch.no_grad()
    def run(self,
            imgsz=640,  # 1440 # inference size (pixels)//推理大小 current value=640
            max_det=3,  # maximum detections per image//每个图像的最大检测次数 current value =40
            # self.source = '0'
            # self.device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=True,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)//边界框厚度
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=True,  # use FP16 half-precision inference
            loop_flag=True
            , streams_list=None):

        save_img = not nosave and not self.source.endswith('.txt')  # save inference images
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        # try:
        set_logging()
        device = select_device(self.device)  ### from utils.torch_utils import select_device
        half &= device.type != '0'  # 'cpu'# half precision only supported on CUDA

        # Load model
        model = attempt_load(self.weights,
                             map_location=device)  # load FP32 model  from models.experimental import attempt_load
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False  # TODO:bug-3  classify can not set True
        if classify:
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

        # Dataloader
        if webcam:  # self.source.isnumeric() or self.source.endswith('.txt') or
            view_img = check_imshow()  # from utils.general
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=imgsz,
                                  stride=stride)  # loadstreams  return self.sources, img, img0, None
            bs = len(dataset)  # batch_size
            # #### streams = LoadStreams

        else:  # load the images or .mp4
            print('if webcam false')
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
            bs = 1  # batch_size
        # vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference 推理
        if device.type == 'cpu':  # != '0' or 'cpu'
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        start_time = time.time()
        # t0 = time.time()
        count = 0

        # load  the camera's index 载入 摄像头号码
        streams_list = []
        with open('streams.txt', 'r') as file:
            for line in file:
                streams_list.append(line.strip())
        print('streams:', streams_list)

        # dataset = iter(dataset)  ##迭代器 iter 创建了一个迭代器对象，每次调用这个迭代器对象的__next__()方法时，都会调用 object

        while loop_flag:  ##### 采用循环来 检查是否 停止推理
            print('Run while loop')
            print(' while loop self.is_continue', self.is_continue)
            print(' while loop self.jump_out', self.jump_out)

            # change model & device  20230810
            if self.current_weight != self.weights:
                print('self.current_weight != self.weights')
                # Load model
                model = attempt_load(self.weights, map_location=device)  # load FP32 model
                stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(imgsz, s=stride)  # check image size
                names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                if half:
                    model.half()  # to FP16将模型的权重参数和激活值的数据类型转换为半精度浮点数格式。
                    ### 这种转换可以减少模型的内存占用和计算开销，从而提高模型在 GPU 上的运行效率
                # Run inference
                if device.type == 'cpu':  # 'cpu' or GPU '0' '1''2'
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                self.current_weight = self.weights

            # load  streams
            # _pred_flag = True

            if self.is_continue:
                global results, ngCounter, okCounter, loopCounter, emit_frame_flag
                det_flag = None
                #  loadstreams // dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
                for path, img, im0s, self.vid_cap in dataset:  # 由于dataset在RUN中运行 会不断更新，所以此FOR循环 不会穷尽
                    t1 = time_sync()
                    # print(path)
                    # print(len(path), type(img), len(im0s), type(self.vid_cap))
                    # ['0', '1'] <class 'numpy.ndarray'> <class 'list'> <class 'cv2.VideoCapture'>
                    # #for testing : show row image
                    # cv2.imshow('ch0', im0s[0])
                    # cv2.imshow('ch1', im0s[1])
                    # ### img recode
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    # 初始化 判断条件 建立对应字典 {key:value}
                    quantity_dic = {name: 0 for name in names}  # create diction
                    confidence_dic = {name: 0 for name in names}
                    area_dic = {name: 0 for name in names}
                    count += 1  # ### FSP counter
                    if count % 30 == 0 and count >= 30:  # 大循环Loop 执行10的倍数次时，更新FSP
                        loopcycle = int(30 / (time.time() - start_time))  # 大 循环周期
                        self.send_fps.emit('fsp:' + str(loopcycle))
                        start_time = time.time()  # update start-time
                    if self.vid_cap:  # 显示视频进度条（当输入为video文件时）
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    # # todo  建立 pred 预测开关， 2种图像输出方式 ， 原始输出  VS  预测结果后输出，控制变量 = self.pred_flag
                    if not self.pred_flag and self.is_continue:  # if not pred_frag  output raw frame
                        for i, index in enumerate(streams_list):
                            t2 = time_sync()
                            ms = round((t2 - t1), 3)  # frame text: fsp
                            fsp = int(1 / ms) if ms > 0 else 'div0'
                            # print(fsp,t1,t2)
                            label_chanel = str(streams_list[i])
                            # print(i, index, label_chanel)
                            im0 = im0s[i].copy()  # for path, img, im0s, self.vid_cap in dataset
                            cv2.putText(im0, str(f'FPS. {fsp}  CAM. {label_chanel}'), (40, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                            im0 = cv2.resize(im0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                            ## chanel-0  ##### show images
                            if label_chanel == '0' and emit_frame_flag:
                                self.send_img_ch0.emit(im0)  ### 发送图像
                                # print('seng img : ch0')
                            ## chanel-1
                            if label_chanel == '1' and emit_frame_flag:
                                self.send_img_ch1.emit(im0)  ### 发送图像
                                # print('seng img : ch1')
                            # chanel-2
                            if label_chanel == '2' and emit_frame_flag:
                                self.send_img_ch2.emit(im0)  ### 发送图像fi
                                # print('seng img : ch2')
                            ## chanel-3
                            if label_chanel == '3' and emit_frame_flag:
                                self.send_img_ch3.emit(im0)  #### 发送图像
                                # print('seng img : ch3')
                            ## chanel-4
                            if label_chanel == '4' and emit_frame_flag:
                                self.send_img_ch4.emit(im0)  #### 发送图像
                                # print('seng img : ch4')
                            ## chanel-5
                            if label_chanel == '5' and emit_frame_flag:
                                self.send_img_ch5.emit(im0)  #### 发送图像
                                # print('seng img : ch5')

                    # Inference prediction
                    # TODO ： 原来的代码  输出 推理后的 图像  im0 = with box  imc= without box
                    if self.pred_flag and self.is_continue:  # 预测后 再输出图像  add box 为可选项目
                        add_box = mainWin.plot_box_CheckBox.isChecked()

                        # pred = model(img, augment=augment)[0] # 预测  使用loadWebcam是 加载的model
                        # print('pred_flag = true pred')
                        pred = model(img,
                                     augment=augment,
                                     visualize=increment_path(save_dir / Path(path).stem,
                                                              mkdir=True) if visualize else False)[0]

                        # Apply NMS
                        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms,
                                                   max_det=max_det)

                        # Apply Classifier
                        if classify:  # classify = False
                            pred = apply_classifier(pred, modelc, img, im0s)
                            print(f'type pred:', type(pred), len(pred))

                        # emit frame  & Process detections
                        for i, det in enumerate(pred):  # pred = []
                            # print(f'i: {i}')
                            if i == len(pred) - 1:  # the last one
                                loopCounter += 1
                                # print(f'main loop: {loopCounter} ')
                                if det_flag:
                                    ngCounter += 1
                                    det_flag = False  # Reset det_flag
                                else:
                                    det_flag = False  # Reset det_flag
                            # #label_index 方法1  ↓ ###label_chanel 依据 list det的 元素
                            # #label_chanel = str(i)
                            # ##label_index 方法2  ↓  依据 streams.txt camera号码
                            if len(pred) <= len(streams_list):
                                label_chanel = str(streams_list[i])
                                # print(f'len(pred) : {len(pred)} ')

                            else:
                                print(f'streams : {len(pred)} camera quantity : {len(streams_list)}')
                                break
                            # print(type(label_chanel),'img chanel=', label_chanel)

                            if webcam:  # streams  #webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith
                                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                            else:  # ##image
                                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                            p = Path(p)  # to Path
                            # save_path = str(save_dir / p.name)  # img.jpg
                            # txt_path = str(save_dir / 'labels' / p.stem) + (dtxt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  #
                            txt_path = r'.\auto_save\txt\results'  # .txt 结果保存路径 r'.\auto_save'
                            s += '%gx%g ' % img.shape[2:]  # print string
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            imc = im0.copy()  # if save_crop else im0  # for save NG frame
                            if len(det):  # if trigger    detection per image
                                # #counter of ng judgement
                                det_flag = True
                                # if det_flag and i == len(pred) - 1:
                                #     ngCounter += 1
                                #     det_flag = False
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                                # Print results
                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                                # function: save_txt / plot one box /save image
                                for *xyxy, conf, cls in reversed(det):
                                    if save_txt:  # save_txt=False,  # save results to *.txt
                                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                            -1).tolist()  # normalized xywh
                                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                        with open(txt_path + '.txt', 'a') as f:
                                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                            # print(txt_path + '.txt')
                                    # plot_one_box here
                                    # if save_img or save_crop or view_img:  # Add bbox to image
                                    # if self.pred_flag:  # Add bbox to image
                                    # print('plot_one_box', save_img, save_crop, view_img)
                                    # save_img = not nosave and not self.source.endswith('.txt')  # save inference images
                                    # save_crop=False,  # save cropped prediction boxes
                                    # view_img =check_inshow() # Check if environment supports image displays
                                    # print(f'Line 317 save_img {save_img},save_crop {save_crop},view_img {view_img}')
                                    c = int(cls)  # index of class
                                    quantity_dic[names[c]] += 1  # 统计匹配目标的个数
                                    # print('quantity_dic-2',quantity_dic) # 输出示例 statisstic_dic-2 {'block': 0, 'scratch': 0, 'edge': 0, 'fibre': 0, 'spot': 3}
                                    label = None if hide_labels else (
                                        names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    # print(f'label: {label}', type(label)) # 输出示例 label: spot 0.97 <class 'str'>
                                    # 使用 split 方法按空格拆分 label 变量
                                    parts = label.split()
                                    # 分别取出类别和置信度，并赋值给新变量
                                    det_name = parts[0]
                                    det_confidence = float(parts[1])  # 将置信度转换为浮点数
                                    confidence_dic[det_name] = det_confidence  # 更新字典中的对应键值 置信度
                                    values = [float(tensor.item()) for tensor in xyxy]  # 提取数值并添加到新的列表中
                                    w = int(values[2] - values[0])
                                    h = int(values[3] - values[1])
                                    area_dic[det_name] = w * h  # 更新字典中的对应键值 像素面积
                                    # print('wh:', w, h)
                                    # print('area_dic', area_dic)

                                    plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                                 line_thickness=line_thickness)

                                    # save NG image  here ↓
                                    # function auto_save  Write results  save NG image in floder jpg

                                    if self.save_folder:  #### when checkbox: autosave is  setcheck, self.save_folder = true
                                        os.makedirs(self.save_folder, exist_ok=True)
                                        if len(det):
                                            if det_name != 'spot':  # 限定保存类型，特定名称
                                                # if names[c]:  # 不限定保存类型，非空即保存
                                                save_path = os.path.join(self.save_folder,
                                                                         f'{det_name}_' + time.strftime(
                                                                             '%Y_%m_%d_%H_%M_%S',
                                                                             time.localtime()) + f'_Cam{label_chanel}_' + f'{det_confidence}_' + '.jpg')
                                                # todo function auto_save : 选择SAVE图像类型 box
                                                add_box = mainWin.plot_box_CheckBox.isChecked()
                                                # cv2.imwrite(save_path, im0)  # im0 = im0s.copy()  with box
                                                # cv2.imwrite(save_path, imc)  # imc = no box
                                                im = imc if not add_box else im0
                                                cv2.imwrite(save_path, im)  # save image
                                                # print('plot_box_CheckBox', myWin.plot_box_CheckBox.isChecked())
                                                print(
                                                    str(f'save as .jpg im{i} , CAM = {label_chanel},save_path={save_path}'))  # & str(save_path))
                                                print('CheckBox_autoSave', mainWin.CheckBox_autoSave.isChecked())

                                # print('detection is running')

                                # if 'sql_is_open' is true, write data to SQL
                                if SQL_is_open:
                                    if feedback_data_D3 == '0110000c000281cb':  # '0105330BFF00F2BC'
                                        feedbacksql = 'Output successful'
                                    else:
                                        feedbacksql = 'Output failed'
                                    sqltime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 写入时间
                                    writesql(sqltime, names[c], feedbacksql)

                            t2 = time_sync()
                            fsp = int(1 / (t2 - t1)) if (t2 - t1) > 0 else 0  # frame text:
                            # print(f'{s}Done. ({t2 - t1:.3f}s fsp ={fsp})')
                            # precition end #######################################################################

                            # emit frame  Stream results

                            # if self.is_continue: # ##### send image in loop @  for i, det in enumerate(pred):
                            cv2.putText(im0, str(f'FSP. {fsp}  CAM. {label_chanel}'), (40, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                            im0 = cv2.resize(im0, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
                            # chanel-0  ##### show images

                            if label_chanel == '0' and emit_frame_flag:
                                self.send_img_ch0.emit(im0)  ### 发送图像
                                # print('seng img : ch0')
                            # chanel-1
                            if label_chanel == '1' and emit_frame_flag:
                                self.send_img_ch1.emit(im0)  ### 发送图像
                                # print('seng img : ch1')
                            # chanel-2
                            if label_chanel == '2' and emit_frame_flag:
                                self.send_img_ch2.emit(im0)  ### 发送图像fi
                                # print('seng img : ch2')
                            # chanel-3
                            if label_chanel == '3' and emit_frame_flag:
                                self.send_img_ch3.emit(im0)  #### 发送图像
                                # print('seng img : ch3')
                            # chanel-4
                            if label_chanel == '4' and emit_frame_flag:
                                self.send_img_ch4.emit(im0)  #### 发送图像
                                # print('seng img : ch4')
                            # chanel-5
                            if label_chanel == '5' and emit_frame_flag:
                                self.send_img_ch5.emit(im0)  #### 发送图像
                                # print('seng img : ch5')
                            # ##send the detected result
                            # self.send_statistic.emit(quantity_dic)  #发送 检测结果 quantity_dic name:数量
                            # self.send_statistic.emit(confidence_dic)  # 发送 检测结果 confidence_dic  name：置信度
                            self.send_statistic.emit(area_dic)  # 发送 检测结果 confidence_dic  name：置信度
                    # #end line  if pred_flag __________________________________________________________
                    '''
                    if self.save_folder:  #### when checkbox: autosave is  setcheck
                        # save as mp4
                        if self.vid_cap is None:  ####save as .mp4
                            # else: ### self.vid_cap is cv2capture save as .mp4
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_folder, time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                                       time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                            print(str(f'save as .mp4  CAM = {label_chanel}'))  # & str(save_path))
                    '''
                    if self.rate_check:
                        time.sleep(1 / self.rate)  # self.det_thread.rate =  self.rateSpinBox.setValue(x)  * 10

                    # im0 = annotator.result()
                    # Write results

                    if self.jump_out:
                        print('jump_out push-2', self.jump_out)
                        global cam_stop
                        cam_stop = True
                        if streams_list:  # 摄像头启动列表非空时 方可加载
                            stopcam = LoadStreams()
                            stopcam.stop_cam()  # 使用非streams.txt时存在bug，默认Stop参数为streams.txt
                        # if self.vid_cap.isOpened():
                        #     self.vid_cap.release()  # todo bug-2  无法释放摄像头  未解决
                        #     logging.info("vid_cap.release...")
                        #     time.sleep(2)
                        #     continue

                        self.send_percent.emit(0)
                        self.send_msg.emit('Stop')
                        # if hasattr(self, 'out'):
                        #     self.out.release()
                        #     print('self.out.release')

                self.is_continue = False  # exit inner loop
                print('self.is_continue set False')
                # 暂停时 重置输出 IO  将字典 area_dic 每个元素的值 都设置为0
                for key in area_dic:
                    area_dic[key] = 0
                self.send_statistic.emit(area_dic)
                print('reset output area_dic = 0')
            else:
                print('is_continue break', self.is_continue)

            if not self.vid_cap.isOpened():
                loop_flag = False  # exit main loop  # todo bug 此处退出会卡死

        if update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)

        #### 生成结果文件夹
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     print(f"Results saved to {save_dir}{s}")

        # except Exception as e:
        #     self.send_msg.emit('%s' % e)

# ↓ Class Main Window 主窗口
class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # self.LoadStreams_thread = None
        self.setupUi(self)
        self.mouse_flag = False

        # style 1: window can be stretched ↓
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
        # style 2: window can not be stretched ↓
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # Transparency of window

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_sensor_data)  # sensor data update
        self.timer.start(1000)

        # search models automatically
        self.comboBox_model.clear()  ### clear model
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))
        self.comboBox_model.clear()
        self.comboBox_model.addItems(self.pt_list)

        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox_model.currentText()  # get model from combobox
        self.device_type = self.comboBox_device.currentText()  # get device type from combobox
        self.source_type = self.comboBox_source.currentText()  # get device type from combobox
        self.port_type = self.comboBox_port.currentText()  # get port type from combobox
        self.det_thread.weights = "./pt/%s" % self.model_type  # difined
        self.det_thread.device = self.device_type  # difined  device
        self.det_thread.source = self.source_type  # get origin source index
        self.det_thread.percent_length = self.progressBar.maximum()
        # the connect function transform to  def run_or_continue(self):
        # tab0-mutil
        self.det_thread.send_img_ch0.connect(lambda x: self.show_image(x, self.video_label_ch0))
        self.det_thread.send_img_ch1.connect(lambda x: self.show_image(x, self.video_label_ch1))
        self.det_thread.send_img_ch2.connect(lambda x: self.show_image(x, self.video_label_ch2))
        self.det_thread.send_img_ch3.connect(lambda x: self.show_image(x, self.video_label_ch3))
        self.det_thread.send_img_ch4.connect(lambda x: self.show_image(x, self.video_label_ch11))
        self.det_thread.send_img_ch5.connect(lambda x: self.show_image(x, self.video_label_ch12))
        # tab-1
        self.det_thread.send_img_ch0.connect(lambda x: self.show_image(x, self.video_label_ch4))
        # tab-2
        self.det_thread.send_img_ch1.connect(lambda x: self.show_image(x, self.video_label_ch5))
        # tab-3
        self.det_thread.send_img_ch2.connect(lambda x: self.show_image(x, self.video_label_ch6))
        # tab-4
        self.det_thread.send_img_ch3.connect(lambda x: self.show_image(x, self.video_label_ch7))
        # tab-5
        self.det_thread.send_img_ch4.connect(lambda x: self.show_image(x, self.video_label_ch8))
        # tab-6
        self.det_thread.send_img_ch5.connect(lambda x: self.show_image(x, self.video_label_ch9))

        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        # self.runButton_modbus.clicked.connect(self.modbus_on_off) # modbusRTU mode
        self.runButton_modbus.clicked.connect(self.modbustcp_on_off) #modbusTCP mode
        self.stopButton.clicked.connect(self.stop)

        self.comboBox_model.currentTextChanged.connect(self.change_model)
        self.comboBox_device.currentTextChanged.connect(self.change_device)
        self.comboBox_source.currentTextChanged.connect(self.change_source)
        self.comboBox_port.currentTextChanged.connect(self.change_port)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))
        self.checkBox_latency.clicked.connect(self.latency_check)
        self.CheckBox_autoSave.clicked.connect(self.auto_save_folder)
        self.pred_CheckBox.clicked.connect(self.pred_run)
        self.load_parameters()  # loading setting
        self.actionGeneral.triggered.connect(self.setting_page_show)  # 设置按键跳转至设置UI
        self.Button_setting.clicked.connect(self.setting_page_show)  # 设置按键跳转至设置UI

        self.dateTimeEdit.setDateTime(QDateTime.currentDateTime())  # emit dateTime to UI

    def read_sensor(self):  ### 检查触发开关
        global ser2
        # print("in read sensor")
        # ser2 = serial.Serial('com9', 38400, 8, 'N', 1, 0.3)    #将串口设置为全局变量可有效降低通讯延时，def内延时0.3~4S不等，全局变量0.3S
        # modbus_rtu model ↓----------------------------------
        if not ser2 is None:
            sensor = modbus_rtu.writedata(ser2, '01 02 00 00 00 01 B9 CA')
            if sensor == '010201016048':
                return True
            else:
                return False
        return False

    def update_sensor_data(self):  ### 光纤开关信号reflash
        global output_box_list
        if sensor_is_open:
            sensor_data = self.read_sensor()

        if sensor_is_open:
            global current_state
            # print(current_state)
            new_state = sensor_data
            if new_state != current_state:
                if sensor_data:
                    print("sensor start")
                    self.runButton.setChecked(True)
                    self.runButton.setText('PAUSE')
                    self.det_thread.is_continue = True
                    if not self.det_thread.isRunning():
                        self.run_or_continue()
                else:
                    print("sensor stop")
                    # reset output
                    self.checkBox_2.setChecked(False)
                    self.checkBox_3.setChecked(False)
                    self.checkBox_4.setChecked(False)
                    self.checkBox_5.setChecked(False)
                    self.checkBox_6.setChecked(False)
                    self.checkBox_7.setChecked(False)
                    self.checkBox_8.setChecked(False)
                    self.checkBox_9.setChecked(False)
                    output_box_list = [0, 0]
                    # if len(output_box_list):
                    #     for i in len(output_box_list):
                    #         output_box_list[i] = 0
                    print('reset output')
                    time.sleep(0.5)  # wait the checkbox set false
                    self.det_thread.is_continue = False
                    self.runButton.setChecked(False)
                    self.runButton.setText('RUN')

                current_state = new_state

    def run_or_continue(self):  # runButton.clicked.connect
        # self.det_thread.source = 'streams.txt'
        self.det_thread.jump_out = False
        # print('runbutton is check', self.runButton.isChecked())
        if self.runButton.isChecked():
            self.runButton.setText('PAUSE')
            self.det_thread.is_continue = True
            self.det_thread.pred_flag = self.pred_CheckBox.isChecked()
            if not self.det_thread.isRunning():
                self.det_thread.start()
            device = os.path.basename(self.det_thread.device)  ### only for display
            source = os.path.basename(self.det_thread.source)  ### 引用 det_thread类的 self.source
            source = str(source) if source.isnumeric() else source  ### source为 int时 转换为 str
            self.statistic_msg('Detecting >> model：{}，device: {}, source：{}'.
                               format(os.path.basename(self.det_thread.weights), device,
                                      source))
            print('self.det_thread.is_continue', self.det_thread.is_continue)

        else:
            self.det_thread.is_continue = False
            self.runButton.setText('RUN')
            self.statistic_msg('Pause')

            # print('self.det_thread.is_continue', self.det_thread.is_continue)

    def calculate_crc(self, raw_data):  # 计算CRC校验码
        # 参数raw_data 全部采用十进制格式列表： [站号 , 功能码, 软元件地址 , 读写位数/数据] 示例raw_data = [1, 6, 10, 111]
        # 将 raw_data（DEC格式） 转换为 hex_data
        hex_data = [format(x, 'X').zfill(4) for x in raw_data]
        string = hex_data[0]
        hex_data[0] = string[-2:]
        string = hex_data[1]
        hex_data[1] = string[-2:]
        try:
            hex_data[4]  # 确认是否有第五位→多寄存器读写
        except:
            print("calculate_crc: no hex 4")
        else:
            string = hex_data[4]
            hex_data[4] = string[-2:]  # 多寄存器位数信息
            string = hex_data[5]
            hex_data[5] = string.zfill(8)  # 写入2个寄存器/补充至8位16进制数
            string = hex_data[5]
            string1 = string[-4:]  # 串口写入高低位与PLC高低位逻辑不一致，需要高低4位互换
            string2 = string[:4]
            hex_data[5] = ''.join([string1, string2])
        # 将 hex_data 转换为 str_data
        str_data = ' '.join([x[i:i + 2] for x in hex_data for i in range(0, len(x), 2)])
        # 将字符串转换为十六进制数组
        data_array = [int(x, 16) for x in str_data.split(' ')]

        # 计算CRC校验码
        crc = 0xFFFF
        for i in range(len(data_array)):
            crc ^= data_array[i]
            for j in range(8):
                if crc & 1:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        # 将CRC校验码添加到原始数据后面
        crc_code = str_data + ' ' + format(crc & 0xFF, '02X') + ' ' + format((crc >> 8) & 0xFF, '02X')
        return crc_code  # return str  crc_data: 01 06 00 0A 00 6F E9 E4

    def thread_mudbus_run(
            self):  ###  PC → PLC  通讯线程 ↓  CRC校验计算 [站号(DEC) , 功能码(DEC), 软元件地址（DEC) , 读写位数/数据(DEC)] 示例raw_data = [1, 6, 10, 111]
        global modbus_flag, okCounter, ngCounter, output_box_list
        modbus_flag = True

        self.port_type = self.comboBox_port.currentText()
        print(type(self.port_type), self.port_type)

        if self.ret:  ### openport sucessfully
            # feedback_data = modbus_rtu.writedata(self.ser, DO_ALL_OFF)  ###OUT1-4  OFF  全部继电器关闭  初始化
            self.runButton_modbus.setChecked(True)
            print('thread_mudbus_run modbus_flag = True')

            while self.runButton_modbus.isChecked() and modbus_flag:
                start = time.time()
                modbus_tcp.modbustcp_write_registers(10, 2, ngCounter)  # D10 写入数据 NG次数
                modbus_tcp.modbustcp_write_registers(12, 2, loopCounter)  # 写入D12 检查次数

                for i, n in enumerate(output_box_list):
                    if len(output_box_list) >= 1:
                        if i == 0:
                            # modbus_rtu.writedata(self.ser, DO0_ON) if n else modbus_rtu.writedata(self.ser, DO0_OFF)
                            modbus_tcp.modbustcp_write_registers(20, 1,
                                                                 1) if n else modbus_tcp.modbustcp_write_registers(
                                20,
                                1, 0)
                            # modbus_tcp.modbustcp_write_coil()
                            # time.sleep(0.1)
                        if i == 1:
                            # modbus_rtu.writedata(self.ser, DO1_ON) if n else modbus_rtu.writedata(self.ser, DO1_OFF)
                            modbus_tcp.modbustcp_write_registers(22, 1,
                                                                 1) if n else modbus_tcp.modbustcp_write_registers(
                                22,
                                1, 0)
                            # time.sleep(0.1)
                        if i == 2:
                            # modbus_rtu.writedata(self.ser, DO2_ON) if n else modbus_rtu.writedata(self.ser, DO2_OFF)
                            modbus_tcp.modbustcp_write_registers(24, 1,
                                                                 1) if n else modbus_tcp.modbustcp_write_registers(
                                24,
                                1, 0)
                        if i == 3:
                            # modbus_rtu.writedata(self.ser, DO3_ON) if n else modbus_rtu.writedata(self.ser, DO3_OFF)
                            modbus_tcp.modbustcp_write_registers(26, 1,
                                                                 1) if n else modbus_tcp.modbustcp_write_registers(
                                26,
                                1, 0)
                            # time.sleep(0.1)
                        if i == 4:
                            # modbus_rtu.writedata(self.ser, DO4_ON) if n else modbus_rtu.writedata(self.ser, DO4_OFF)
                            modbus_tcp.modbustcp_write_registers(28, 1,
                                                                 1) if n else modbus_tcp.modbustcp_write_registers(
                                28,
                                1, 0)
                        if i == 5:
                            # modbus_rtu.writedata(self.ser, DO5_ON) if n else modbus_rtu.writedata(self.ser, DO5_OFF)
                            modbus_tcp.modbustcp_write_registers(30, 1,
                                                                 1) if n else modbus_tcp.modbustcp_write_registers(
                                30,
                                1, 0)
                        if i == 6:
                            # modbus_rtu.writedata(self.ser, DO6_ON) if n else modbus_rtu.writedata(self.ser, DO6_OFF)
                            modbus_tcp.modbustcp_write_registers(32, 1,
                                                                 1) if n else modbus_tcp.modbustcp_write_registers(
                                32,
                                1, 0)
                        if i == 7:
                            # modbus_rtu.writedata(self.ser, DO7_ON) if n else modbus_rtu.writedata(self.ser, DO7_OFF)
                            modbus_tcp.modbustcp_write_registers(34, 1,
                                                                 1) if n else modbus_tcp.modbustcp_write_registers(
                                34,
                                1, 0)

                stop = time.time()
                freq = int(1 / (stop - start))
                self.label_modbus.setText(str(freq))
            else:
                modbus_flag = False
                print('modbus shut off')
                time.sleep(0.1)
                try:
                    self.ser.close()
                    self.client.close()
                except Exception as e:
                    print('ser.close error ', e)

    def modbus_on_off(self):  # modbus rtu 控制开关 open port ↓
        global modbus_flag
        # if not modbus_flag:
        if self.runButton_modbus.isChecked():
            print('runButton_modbus.isChecked')
            modbus_flag = True
            print('set  modbus_flag = True')
            try:
                self.ser, self.ret, error = modbus_rtu.openport(self.port_type, 9600, 5)  # 打开端口
            except Exception as e:
                print('openport erro -1', e)
                self.statistic_msg(str(e))

            if not self.ret:
                self.runButton_modbus.setChecked(False)
                # self.runButton_modbus.setStyleSheet('background-color:rgb(220,0,0)') ### background = red
                MessageBox(
                    self.closeButton, title='Error', text='Connection Error: ' + str(error), time=2000,
                    auto=True).exec_()
                print('port did not open')
            else:  # self.ret is  True
                self.runButton_modbus.setChecked(True)
                _thread.start_new_thread(mainWin.thread_mudbus_run, ())  # 启动检测 信号 循环
        else:  # shut down modbus
            print('runButton_modbus.is unChecked')
            modbus_flag = False
            self.runButton_modbus.setChecked(False)
            print('shut down modbus_flag = False')  ####  ###

    def modbustcp_on_off(self):  # modbus_tcp 控制开关 open port ↓
        global modbus_flag , modbus_tcp_ip, modbus_tcp_port
        if self.runButton_modbus.isChecked():
            modbus_flag = True
            error = None
            try:
                self.client, self.ret, error = modbus_tcp.modbustcp_open_port(modbus_tcp_ip, modbus_tcp_port)  # 打开端口
            except Exception as e:
                print('tcp openport erro -1', e)
                self.statistic_msg(str(e))
                self.runButton_modbus.setChecked(False)
                MessageBox(
                    self.closeButton, title='Error', text='ModbusTCP Failed to connect !: ' + str(error), time=2000,
                    auto=True).exec_()
                print('ModbusTCP Failed to connect !')

            else:  # self.ret is  True
                print('modbus_tcp connected successful')
                self.runButton_modbus.setChecked(True)
                _thread.start_new_thread(mainWin.thread_mudbus_run, ())  # 启动检测 信号 循环
        else:  # shut down modbus
            print('runButton_modbus.is unChecked')
            modbus_flag = False
            self.runButton_modbus.setChecked(False)
            print('shut down modbus_flag = False')  #

    def stop(self):  # connect stopButton  主窗口停止按键
        if not self.det_thread.jump_out:
            self.det_thread.jump_out = True


    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def auto_save_folder(self):  ###auto_save folder
        if self.CheckBox_autoSave.isChecked():
            self.det_thread.save_folder = r'.\auto_save\jpg\pt_protrusion'  ### save result as .mp4
        else:
            self.det_thread.save_folder = None

    def pred_run(self):
        if self.pred_CheckBox.isChecked():
            self.det_thread.pred_flag = True
        else:
            self.det_thread.pred_flag = False

    def latency_check(self):  #####latency checkbox
        if self.checkBox_latency.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading rtsp stream', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):  # UI bottun 'cameraButton'
        try:
            self.stop()  # stop running thread
            print('chose_cam run')
            MessageBox(
                self.closeButton, title='Enumerate Cameras', text='Loading camera', time=2000,
                auto=True).exec_()  # self.closeButton, title='Enumerate Cameras', text='Loading camera', time=2000, auto=True).exec_()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            print('enum_camera:', cams)
            self.statistic_msg('enum camera：{}'.format(cams))
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)
            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()  # choose source
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x / 100)
            self.det_thread.conf_thres = x / 100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x / 100)
            self.det_thread.iou_thres = x / 100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        print(msg)

    def show_msg(self, msg):
        self.statistic_msg(msg)

        if msg == "Finished":
            print(' msg == Finished')

    def change_model(self, x):
        self.model_type = self.comboBox_model.currentText()  # comboBox
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def change_device(self, x):
        self.device_type = self.comboBox_device.currentText()
        self.det_thread.device = self.device_type
        self.statistic_msg('Change device to %s' % x)

    def change_source(self, x):  # while the comboBox_source has changed
        self.source_type = self.comboBox_source.currentText()
        self.det_thread.source = self.source_type
        self.statistic_msg('Change source to %s' % x)

    def change_port(self, x):
        self.port_type = self.comboBox_port.currentText()
        self.statistic_msg('Change port to %s' % x)

    def open_file(self):
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        if name:  # 打开对象属于指定类型的文件时 ↓
            self.det_thread.source = name
            self.textBrowser.setText(name)
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def max_or_restore(self):  ### window size control
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.mouse_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.mouse_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.mouse_flag = False

    def setting_page_show(self):  # 菜单栏设定 genaral
        self.trg_setting_ui = setting_page()
        self.trg_setting_ui.show()

    @staticmethod
    def show_image(img_src, label):  ### input img_src  output to pyqt label
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):  ### predicttion  output  resultWidget
        global results, okCounter, ngCounter, output_box_list
        try:
            self.dateTimeEdit.setDateTime(QDateTime.currentDateTime())  # emit dateTime to UI
            self.resultWidget.clear()
            # print('statistic_dic:', statistic_dic) # NG项目位置固定
            dic2list = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)

            dic2list = [i for i in dic2list if i[1] > 0]  # append to List  while the value greater than 0
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in dic2list]  # reform the list
            # print('statistic result_list:',  results)  # NG项目排列在前面
            self.resultWidget.addItems(results)
            self.label_okCounter.setText(str(loopCounter))

            for index, (key, value) in enumerate(statistic_dic.items()):
                # print(f"Position: {index}, Key: {key}, Value: {value}")
                global area_min
                if index == 0:
                    self.checkBox_2.setChecked(True) if value > int(area_min) else self.checkBox_2.setChecked(False)
                    self.checkBox_2.setText(key)
                if index == 1:
                    self.checkBox_3.setChecked(True) if value > int(area_min) else self.checkBox_3.setChecked(False)
                    self.checkBox_3.setText(key)
                if index == 2:
                    self.checkBox_4.setChecked(True) if value > int(area_min) else self.checkBox_4.setChecked(False)
                    self.checkBox_4.setText(key)
                if index == 3:
                    self.checkBox_5.setChecked(True) if value > int(area_min) else self.checkBox_5.setChecked(False)
                    self.checkBox_5.setText(key)
                if index == 4:
                    self.checkBox_6.setChecked(True) if value > int(area_min) else self.checkBox_6.setChecked(False)
                    self.checkBox_6.setText(key)
                if index == 5:
                    self.checkBox_7.setChecked(True) if value > int(area_min) else self.checkBox_7.setChecked(False)
                    self.checkBox_7.setText(key)
                if index == 6:
                    self.checkBox_8.setChecked(True) if value > int(area_min) else self.checkBox_8.setChecked(False)
                    self.checkBox_8.setText(key)
                if index == 7:
                    self.checkBox_9.setChecked(True) if value > int(area_min) else self.checkBox_9.setChecked(False)
                    self.checkBox_9.setText(key)
                ###  更新全局变量 output_box_list
                output_box_list = [self.checkBox_2.isChecked(), self.checkBox_3.isChecked(),
                                   self.checkBox_4.isChecked(), self.checkBox_5.isChecked(),
                                   self.checkBox_6.isChecked()]

            if len(results):
                # ngCounter += 1
                self.label_ngCounter.setText(str(ngCounter))
                # print(f'ngCounter: {ngCounter}')
                self.pushButton_okng.setText(f"NG: {len(results)}")
                self.pushButton_okng.setStyleSheet('''QPushButton{
                        font-size: 20px;
                        font-family: "Microsoft YaHei";
                        font-weight: bold;
                        border-radius: 4px;
                        background-color: rgb(240,20,30);
                        color: rgb(255, 255, 255);
                        }''')
            else:
                self.pushButton_okng.setText(f"OK")
                self.pushButton_okng.setStyleSheet('''QPushButton{
                        font-size: 20px;
                        font-family: "Microsoft YaHei";
                        font-weight: bold;
                        border-radius: 4px;
                        background-color: rgb(0,220,127);
                        color: rgb(255, 255, 255);
                        }''')

        except Exception as e:
            print(repr(e))

    def load_parameters(self):  ### laoding mainwindow object...'
        print(' run load_parameters() at mainwindows')
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            latency = False
            auto_save = False
            device = 0
            port = 5
            source = 0
            model = 0
            add_box = False
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "latency": latency,
                          "auto_save": auto_save,
                          "device": device,
                          "port": port,
                          "source": source,
                          "model": model,
                          "add_box": add_box,
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            # print('load config:', type(config), config)
            if len(config) < 8:  ### 参数不足时  补充参数
                iou = 0.26
                conf = 0.33
                rate = 10
                latency = False
                auto_save = False
                device = 0
                port = 0
                source = 0
                model = 0
                add_box = False
            else:
                print('loading completed...', config_file, config)
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                latency = config['latency']
                auto_save = config['auto_save']
                device = config['device']  ## index number
                port = config['port']  ## index number
                source = config['source']
                model = config['model']
                add_box = config['add_box']

        # 依据存储的json文件 更新 UI 显示
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        self.rateSpinBox.setValue(rate)
        self.checkBox_latency.setCheckState(latency)
        self.det_thread.rate_check = latency
        self.CheckBox_autoSave.setCheckState(auto_save)
        self.auto_save_folder()  # creat path of  auto_save img
        self.comboBox_device.setCurrentIndex(device)  #
        self.comboBox_port.setCurrentIndex(port)  #
        self.comboBox_source.setCurrentIndex(source)  #
        self.comboBox_model.setCurrentIndex(model)  #
        self.plot_box_CheckBox.setCheckState(add_box)  #

    def save_mainwin_settings(self):
        #### read current paraments
        config_path = 'config/setting.json'
        config = dict()
        config['iou'] = self.iouSpinBox.value()
        config['conf'] = self.confSpinBox.value()  # self.confSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['latency'] = self.checkBox_latency.checkState()  # Latency function .checkState()
        config['auto_save'] = self.CheckBox_autoSave.checkState()  # Auto Save check box .checkState()
        config['device'] = self.comboBox_device.currentIndex()  # 获取当前索引号
        config['port'] = self.comboBox_port.currentIndex()  # 获取当前索引号
        config['source'] = self.comboBox_source.currentIndex()  # 获取当前索引号
        config['model'] = self.comboBox_model.currentIndex()  # 获取当前索引号 20240403
        config['add_box'] = self.plot_box_CheckBox.checkState()
        # config['On/Off State'] =
        ####新增参数 请在此处添加↑ ， 运行UI后 点击关闭按钮 后保存为 json文件 地址= ./config/setting.json
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        print('content config_json:', config_json)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_json)
            print('confi_json write')

    def closeEvent(self, event):  ###点击关闭开关按钮执行 以下
        print('execute : closeEvent of main window')
        global modbus_flag, sensor_is_open, SQL_is_open, ser2
        modbus_flag = False
        self.det_thread.jump_out = True
        self.det_thread.is_continue = False
        # if not ser2 == None:
        if sensor_is_open:
            try:
                ser2.close()  # 240704
                sensor_is_open = False
                ser2 = None
            except:
                ser2 = None
        if SQL_is_open:
            closesql()
            SQL_is_open = False
        self.save_mainwin_settings()
        MessageBox(
            self.closeButton, title='Tips', text='Terminate Program.', time=2000, auto=True).exec_()
        sys.exit(0)

# Class child Window子窗口  参数设置页面 created by PyQt
class setting_page(QMainWindow, Ui_setting):
    def __init__(self):
        super().__init__()
        self.server = None
        self.SQL_switch = None
        self.sensor_switch = None
        self.setupUi(self)
        # function connecting ↓
        self.saveConfig2_Button.clicked.connect(self.save_settings)
        self.sql_connect_Button.clicked.connect(self.run_sql)
        self.SensorPort_connect_Button.clicked.connect(self.sensor_on_off)
        config = self.load_setting()

    def load_setting(self):  # sql config setting.json'
        global sql_server_ip, config, area_min
        config_file = 'config/setting2.json'
        if not os.path.exists(config_file):  # 如果.json文件不存在则创建文件 ↓
            sensor_switch = 0
            SQL_switch = 2
            server = sql_server_ip
            database = 'PE_DataBase'
            username = 'TRG-PE'
            password = '705705'
            new_config = {"sensor_switch": sensor_switch,
                          "SQL_switch": SQL_switch,
                          "server": server,
                          "database": database,
                          "username": username,
                          "password": password,
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            print('setting_page loading config:', config_file, config)
            if len(config) < 6:  #  参数不足时  补充参数,否则无法启动
                print("len", len(config))
                self.sensor_switch = 0
                self.SQL_switch = 2
                self.server = sql_server_ip  # 不在一个域内的账户 只能使用IP 连接SQL服务器
                self.database = 'PE_DataBase'
                self.username = 'TRG-PE'
                self.password = '705705'
                area_min = '1000'
                self.area_max = '50000'
                self.ngcounter_low = '1'
                self.ngcounter_high = '5'
            else:  # 更新UI参数 ↓
                self.sensor_switch = config['sensor_switch']
                self.sensor_port = config['sensor_port']
                self.SQL_switch = config['SQL_switch']
                self.server = config['server']
                self.database = config['database']
                self.username = config['username']
                self.password = config['password']
                area_min = config['area_min']
                self.area_max = config['area_max']
                self.ngcounter_low = config['ng_counter_low_limit']
                self.ngcounter_high = config['ng_counter_higt_limit']

        # 依据存储的json文件 更新 ui参数
        self.checkBox_sensor.setCheckState(config['sensor_switch'])
        self.comboBox_sensor_port.setCurrentIndex(config['sensor_port'])
        self.checkBox_sql.setCheckState(config['SQL_switch'])
        self.lineEdit_server.setText(config['server'])
        self.lineEdit_db.setText(config['database'])
        self.lineEdit_user.setText(config['username'])
        self.lineEdit_pw.setText(config['password'])

        self.checkBox_plc_enable.setCheckState(config['plc_switch'])
        self.comboBox_rtu_port.setCurrentIndex(config['modbus_rtu_port'])
        self.lineEdit_tcp_server.setText(config['modbus_tcp_server'])
        self.lineEdit_tcp_port.setText(config['modbus_tcp_port'])
        self.lineEdit_ngcounter_low.setText(config['ng_counter_low_limit'])
        self.lineEdit_ngcounter_high.setText(config['ng_counter_higt_limit'])
        self.lineEdit_area_max.setText(config['area_max'])
        self.lineEdit_area_min.setText(config['area_min'])


        return config

    def update_settings(self):
        global area_min
        # update area_min
        area_min = self.lineEdit_area_min.text()

    def save_settings(self):
        print("Execute save setting at setting page")
        config_path = 'config/setting2.json'
        config = dict()
        config['sensor_switch'] = self.checkBox_sensor.checkState()  # 保存开关勾选状态
        config['sensor_port'] = self.comboBox_sensor_port.currentIndex()  # 获取当前索引号
        config['SQL_switch'] = self.checkBox_sql.checkState()  # 保存开关勾选状态
        config['server'] = self.lineEdit_server.text()
        config['database'] = self.lineEdit_db.text()
        config['username'] = self.lineEdit_user.text()
        config['password'] = self.lineEdit_pw.text()

        config['plc_switch'] = self.checkBox_plc_enable.checkState()
        config['modbus_rtu_port'] = self.comboBox_rtu_port.currentIndex()
        config['modbus_tcp_server'] = self.lineEdit_tcp_server.text()
        config['modbus_tcp_port'] = self.lineEdit_tcp_port.text()
        config['ng_counter_low_limit'] = self.lineEdit_ngcounter_low.text()
        config['ng_counter_higt_limit'] = self.lineEdit_ngcounter_high.text()
        config['area_max'] = self.lineEdit_area_max.text()
        config['area_min'] = self.lineEdit_area_min.text()

        # 新增参数 请在此处添加↑ ， 运行UI后 点击 save按钮，保存为 json文件 地址= ./config/setting2.json
        try:
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_json)
                print('confi_json2 write', config_path, config)
        except Exception as e:
            QMessageBox.critical(
                self,  # parent window
                'Save Parameters Error',  # title
                f'Failed to write confi_json2:\n{str(e)}',  # message
                QMessageBox.Ok
            )
        else:
            QMessageBox.information(
                self,  # parent window
                'Save Parameters Successfully',  # title
                f'write confi_json2 Successfully:\n{config_json}',  # message
                QMessageBox.Ok
            )

    def closeEvent(self, event):
        print("Execute close event()")
        # 保存设置窗口参数
        self.update_settings()
        event.accept()

    def run_sql(self): # sql_connect_Button
        print("Execute function run_sql")
        print("checkbox_sql", self.checkBox_sql.isChecked())
        global SQL_is_open
        if self.checkBox_sql.isChecked():
            try:
                SQL_is_open = SQL_write.opensql(self.server, self.database, self.username, self.password)  # 打开SQL
            except Exception as e:
                SQL_is_open = False
                QMessageBox.critical(
                    self,  # parent window
                    'Open Port Error',  # title
                    f'Failed to connect sql server:\n{str(e)}',  # message
                    QMessageBox.Ok
                )
        if not self.checkBox_sql.isChecked():
            try:
                SQL_write.closesql()
                SQL_is_open = False
            except Exception as e:
                print("Do not use SQL server", e)
                QMessageBox.critical(
                    self,  # parent window
                    'SQL Error',  # title
                    f'Failed to disconnect sql server:\n{str(e)}',  # message
                    QMessageBox.Ok
                )
        if not SQL_is_open:
            QMessageBox.critical(
                self,  # parent window
                'SQL Error',  # title
                f'Failed to connect sql server:',  # message
                QMessageBox.Ok
            ) ##

    def sensor_on_off(self): # SensorPort_connect_Button
        global ser2, ret2, sensor_is_open
        print('Execute sensor_on_off', self.checkBox_sensor.isChecked())
        sensor_port1 = mainWin.comboBox_port.currentText()
        print('parameters: port1', sensor_port1)
        sensor_port2 = self.comboBox_sensor_port.currentText()
        print('parameters: sensor_port2',sensor_port2)
        if self.checkBox_sensor.isChecked():
            try:
                ser2 = serial.Serial(sensor_port2, 38400, 8, 'N', 1, 0.3)
                print("Try to connect sensor_port2")
            except Exception as e:
                print('openport error-2', e)
                sensor_is_open = False
                QMessageBox.critical(
                    self,  # parent window
                    'Open Sensor Port Error',  # title
                    f'Failed to open sensor port:\n{str(e)}',  # message
                    QMessageBox.Ok
                )
            else:
                print("sensor connect successfully")
                sensor_is_open = True
        if not self.checkBox_sensor.isChecked():
            sensor_is_open = False
            if ser2 :  # sensor port turn on
                try: # try to close the port
                    ser2.close()
                    print("sensor is close")
                    ser2 = None
                except Exception as e:
                    print('close port error-2', e)
                    # Error popup for closing port
                    QMessageBox.critical(
                        self,  # parent window
                        'Close Sensor Port Error',  # title
                        f'Failed to close sensor port:\n{str(e)}',  # message
                        QMessageBox.Ok  # button
                    )
                    self.statistic_msg(str(e))
        if not sensor_is_open:
            QMessageBox.critical(
                self,  # parent window
                'Open Sensor Port Error',  # title
                f'Failed to open sensor port',  # message
                QMessageBox.Ok
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()  # 实例化
    mainWin.show()
    # 加载 子窗口参数 ↓
    _setting_page = setting_page()
    config = _setting_page.load_setting()
    print('setting page config:',config)
    if config['sensor_switch']:
        print('sensor_switch on', config['sensor_switch'])
        _setting_page.sensor_on_off()
    if config['SQL_switch']:
        print('SQL_switch on', config['SQL_switch'])
        _setting_page.run_sql()
    # update parameters
    modbus_tcp_ip = config['modbus_tcp_server']
    modbus_tcp_port = config['modbus_tcp_port']

    mainWin.runButton_modbus.setChecked(True)
    mainWin.modbustcp_on_off()  # start modbus tcp

    # time.sleep(1)
    # print('thread_mudbus_run start')
    # _thread.start_new_thread(myWin.thread_mudbus_run, ())  #### 启动检测 信号 循环
    # 单独输出 调试模式 ↓
    # det_thread.send_img_ch0.connect(lambda x: cvshow_image(x))

    # myWin.showMaximized()
    sys.exit(app.exec_())
