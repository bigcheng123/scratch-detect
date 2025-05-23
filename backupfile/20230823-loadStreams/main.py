from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
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

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam, LoadStreams
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.plots import Annotator, colors, save_one_box,plot_one_box

from utils.torch_utils import select_device,time_sync,load_classifier
from utils.capnums import Camera
from dialog.rtsp_win import Window


class DetThread(QThread): ###继承 QThread
    send_img_ch0 = pyqtSignal(np.ndarray)  ### CH0 output image
    send_img_ch1 = pyqtSignal(np.ndarray)  ### CH1 output image
    send_img_ch2 = pyqtSignal(np.ndarray)  ### CH1 output image
    send_img_ch3 = pyqtSignal(np.ndarray)  ### CH1 output image
    send_statistic = pyqtSignal(dict)  ###
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.device = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold =  None ####'./result'

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            # self.source = '0'
            # self.device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img = False ,  # show results
            save_txt=False,  # save results to *.txt
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
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

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
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model  from models.experimental import attempt_load
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

        # Dataloader
        if webcam: ###self.source.isnumeric() or self.source.endswith('.txt') or
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)  #### loadstreams  return self.sources, img, img0, None
            print('dataset type', type(dataset), dataset)
            bs = len(dataset)  # batch_size
            print('len(bs)=', bs)
            # #### streams = LoadStreams

        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs


        # Run inference 推理
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        start_time = time.time()
        # t0 = time.time()
        count = 0
        # dataset = iter(dataset)  ##迭代器 iter 创建了一个迭代器对象，每次调用这个迭代器对象的__next__()方法时，都会调用 object

        while True: ##### 采用循环来 检查是否 停止推理
            if self.jump_out:
                self.vid_cap.release()
                self.send_percent.emit(0)
                self.send_msg.emit('Stop')
                if hasattr(self, 'out'):
                    self.out.release()
                break

            # change model & device  20230810
            if self.current_weight != self.weights:
                # Load model
                model = attempt_load(self.weights, map_location = device)  # load FP32 model
                stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(imgsz, s=stride)  # check image size
                names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                if half:
                    model.half()  # to FP16
                # Run inference
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                self.current_weight = self.weights

            # load  streams
            if self.is_continue:
                # ### 使用 loadstreams  dataset = ： self.sources, img, img0, None
                for path, img, im0s, self.vid_cap in dataset:
                    # im0s = im0s[0]
                    # cv2.imshow('im0s[0]', im0s[0])  ##### show raw images
                    # # im1s = im0s[1]
                    # cv2.imshow('im0s[1]', im0s[1])  ##### show raw images
                    # print(type(path), path, type(img), type(im0s), im0s[0], type(self.vid_cap))
                    #### img recode
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    statistic_dic = {name: 0 for name in names}
                    count += 1  #### FSP counter
                    if  count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    # Inference
                    t1 = time_sync()
                    # pred = model(img, augment=augment)[0] #### 预测

                    pred = model(img,
                                 augment=augment,
                                 visualize=increment_path(save_dir / Path(path).stem,
                                                          mkdir=True) if visualize else False)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    t2 = time_sync()

                    # Apply Classifier
                    if classify:
                        pred = apply_classifier(pred, modelc, img, im0s)

                    # Process detectionsprint('n',str(n))
                    for i, det in enumerate(pred):  # detections per image
                        if webcam:  # batch_size >= 1     get the frame
                            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                            label_chanel = int(i)
                            # cv2.imshow(str(p), im0)  ##### show images
                            ### print(p,s,frame)
                        else: ### image
                            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        # save_path = str(save_dir / p.name)  # img.jpg
                        # txt_path = str(save_dir / 'labels' / p.stem) + (dtxt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  #
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                                 line_thickness=line_thickness)
                                    if save_crop:
                                        print('save_one_box')
                                        # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        # Print time (inference + NMS)
                        FSP = int(1 / (t2 - t1))
                        print(f'{s}Done. ({t2 - t1:.3f}s FSP={FSP})')
                        # print(f'FSP={FSP}')

                        # Stream results   emit frame
                        if view_img:
                            cv2.putText(im0, str(f'FSP={FSP}'), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                            # res = cv2.resize(im0, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            ## chanel-0  ##### show images
                            if label_chanel == 0:
                                self.send_img_ch0.emit(im0)  ### 发送处理后图像
                            ## chanel-1
                            if label_chanel == 1:
                                self.send_img_ch1.emit(im0)  #### 发送原始图像
                            ## chanel-2
                            if label_chanel == 2:
                                self.send_img_ch2.emit(im0)  #### 发送原始图像
                            ## chanel-2
                            if label_chanel == 3:
                                self.send_img_ch3.emit(im0)  #### 发送原始图像
                            ### ## 发送声明
                            self.send_statistic.emit(statistic_dic)
                            # cv2.imshow('im0s', im0s)  ##### show images
                            # cv2.waitKey(1)  # 1 millisecond

                if self.rate_check:
                    time.sleep(1/self.rate)

                # im0 = annotator.result()

                # Write results
                if self.save_fold:
                    os.makedirs(self.save_fold, exist_ok=True)
                    if self.vid_cap is None:
                        save_path = os.path.join(self.save_fold,
                                                 time.strftime('%Y_%m_%d_%H_%M_%S',
                                                               time.localtime()) + '.jpg')
                        cv2.imwrite(save_path, im0)
                    else:
                        if count == 1:
                            ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                            if ori_fps == 0:
                                ori_fps = 25
                            # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            width, height = im0.shape[1], im0.shape[0]
                            save_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                            self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                       (width, height))
                        self.out.write(im0)
                if percent == self.percent_length:
                    print(count)
                    self.send_percent.emit(0)
                    self.send_msg.emit('finished')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        if update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)

        ##except Exception as e:
        ##    self.send_msg.emit('%s' % e)



class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # style 1: window can be stretched
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
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

        # search models automatically
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)


        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)


        # yolov5 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()  ### get model from combobox
        self.device_type = self.comboBox_2.currentText()  ###  get device type from combobox
        self.det_thread.weights = "./pt/%s" % self.model_type  # difined
        self.det_thread.device = self.device_type # difined  device
        self.det_thread.source = '0'
        self.det_thread.percent_length = self.progressBar.maximum()
        #### the connect funtion transform to  def run_or_continue(self):
        self.det_thread.send_img_ch0.connect(lambda x: self.show_image(x, self.video_label_ch0))
        self.det_thread.send_img_ch1.connect(lambda x: self.show_image(x, self.video_label_ch1))
        self.det_thread.send_img_ch2.connect(lambda x: self.show_image(x, self.video_label_ch2))
        self.det_thread.send_img_ch3.connect(lambda x: self.show_image(x, self.video_label_ch3))

        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        # self.runButton_2.clicked.connect(self.run_or_continue_2)

        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.comboBox_2.currentTextChanged.connect(self.change_device)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()

    def run_or_continue(self):
        # self.det_thread.source = 'streams.txt'
        # self.det_thread.send_img_ch0.connect(lambda x: self.show_image(x, self.video_label_ch0))
        # self.det_thread.send_img_ch1.connect(lambda x: self.show_image(x, self.video_label_ch1))
        # self.det_thread.send_img_ch2.connect(lambda x: self.show_image(x, self.video_label_ch2))
        # self.det_thread.send_img_ch3.connect(lambda x: self.show_image(x, self.video_label_ch3))
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            device = os.path.basename(self.det_thread.device)
            source = os.path.basename(self.det_thread.source)
            source = str(source) if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，device: {}, source：{}'.
                               format(os.path.basename(self.det_thread.weights),device,
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    # def run_or_continue_2(self):
    #     # self.det_thread.source = '1'
    #     self.det_thread.send_img.connect(lambda x: self.show_image(x, self.video_label_ch1))  ####img_src = x   label = self.raw_video
    #     self.det_thread.jump_out = False
    #     if self.runButton_2.isChecked():
    #         self.saveCheckBox.setEnabled(False)
    #         self.det_thread.is_continue = True
    #         if not self.det_thread.isRunning():
    #             self.det_thread.start()
    #         source = os.path.basename(self.det_thread.source)
    #         source = 'camera' if source.isnumeric() else source
    #         self.statistic_msg('Detecting >> model：{}，file：{}'.
    #                            format(os.path.basename(self.det_thread.weights),
    #                                   source))
    #     else:
    #         self.det_thread.is_continue = False
    #         self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)


    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'  ### save result as .mp4
        else:
            self.det_thread.save_fold = None

    def checkrate(self):
        if self.checkBox.isChecked():
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

    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading camera', time=2000, auto=True).exec_()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
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
                self.det_thread.source = action.text()  ##### choose source
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
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
        # self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.runButton_2.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()  #comboBox_2
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def change_device(self, x):
        self.device_type = self.comboBox_2.currentText()
        self.det_thread.device = self.device_type
        self.statistic_msg('Change device to %s' % x)


    def open_file(self):
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()


    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):  ### input img_src  output to pyqt label
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
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

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)
####  测试用   ↓ ##################################################
def cvshow_image(img):  ### input img_src  output to pyqt label
    try:
        cv2.imshow('Image', img)
    except Exception as e:
        print(repr(e))
    # # 等待键盘输入
    # key = cv2.waitKey(1) & 0xFF
    # # 如果按下ESC键，退出循环
    # if key == 27:
    #     break
    # # 关闭窗口
    # cv2.destroyAllWindows()

if __name__ == "__main__":

    app = QApplication(sys.argv)
    myWin = MainWindow() #### 实例化
    myWin.show()
    print('prameters load completed')

    # det_thread = DetThread() #### 实例化
    # det_thread.weights = "pt/yolov5s.pt"
    # det_thread.device = '0'
    # det_thread.source = 'streams.txt'
    # det_thread.start()   ###
    #
    # ##### connect UI  输出到 UI  ↓
    # det_thread.send_img_ch0.connect(lambda x: myWin.show_image(x, myWin.video_label_ch0))
    # det_thread.send_img_ch1.connect(lambda x: myWin.show_image(x, myWin.video_label_ch1))
    # det_thread.send_img_ch2.connect(lambda x: myWin.show_image(x, myWin.video_label_ch2))
    # det_thread.send_img_ch3.connect(lambda x: myWin.show_image(x, myWin.video_label_ch3))


    ## 单独输出 调试模式 ↓
    # det_thread.send_img.connect(lambda x: cvshow_image(x))

    # myWin.showMaximized()
    sys.exit(app.exec_())


