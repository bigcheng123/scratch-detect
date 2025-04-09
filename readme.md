### Readme English [https://gitee.com/trgtokai/scratch-detect/blob/master/readme.md](https://gitee.com/trgtokai/scratch-detect/blob/master/readme.md)
### Readme Chinese [https://gitee.com/trgtokai/scratch-detect/blob/master/readme-CN.md](https://gitee.com/trgtokai/scratch-detect/blob/master/readme-CN.md)
### Readme Japanese [https://gitee.com/trgtokai/scratch-detect/blob/master/readme-JP.md](https://gitee.com/trgtokai/scratch-detect/blob/master/readme-JP.md)
…………………………………………………………………………………………………………………………………………………………………………………………………………

## Project name: 【Scratch-detect】

### 1. Tips <br>

This program is based on [YOLOv5 v6.1](https://github.com/ultralytics/yolov5/tree/v6.1)

**Hardware platform:** <br>
Camera: Hikvision MV-CU050-90UC<br>
PLC: Mitsubishi FX3S + modbus module<br>
Touch screen: Weiluntong MT8072IP<br>
Light source: Hikvision ring light source MV-LRSS-H-80-W<br>
Light source controller: [Digital 8-channel light source controller](https://detail.tmall.com/item.htm?abbucket=1&id=656543446110&rn=21d65f2d271defe4d3b29e10ced9b2a5&spm=a1z10.5-b.w4011-23573612475.52.201646d6ZWIsQh&skuId=4738283905874)<br>

**Host recommended configuration Hardware:**<br>
1.CPU: i7 13700k and above<br>
2. Graphics card: RTX3050 and above (only supports NVIDIA graphics cards)<br>
3. Memory: 16G or above is recommended<br>
4. Hard disk: 1TB or above is recommended<br>

**Software platform Software:**<br>
1. System: win10 x64 <br>
2. Driver: [Hik MVS](https://www.hikrobotics.com/cn2/source/support/software/MVS_STD_4.3.2_240529.zip), [Weinview EBPRO](https://www.weinview.cn/Admin/Others/DownloadsPage.aspx?nid=3&id=10917&tag=0&ref=download&t=a4ff8b5703a191fe), [NVIDIA driver](https://cn.download.nvidia.com/Windows/555.99/555.99-desktop-win10-win11-64bit-international-nsd-dch-whql.exe), [Mitsubishi GXWORKS2](https://www.mitsubishielectric-fa.cn/site/file-software-detail?id=18), etc.<br>
3. Python environment: [anaconda](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Windows-x86_64.exe)<br>
4. Python IDE: [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC)<br>
5. Python version: Python3.8<br>
6. Version control: [Git](https://git-scm.com/download/win)

**AI detection program Code:**<br>
1. Clone/download all files of this project to your local computer
2. Use Pycharm or other IDE software to open this project
3. Configure the environment and ensure that all dependent packages are successfully installed (list = requirement-trg.txt)
4. Run the main program main.py
5. If you want to run your own detection model, please put the trained model file into the pt folder, and then select the corresponding PT file in the UI interface

### 2. Demo video Demo
China region Bilibili Demo ↓
[https://www.bilibili.com/video/BV1nz421S7KR](https://www.bilibili.com/video/BV1nz421S7KR)

YouTube Demo outside China ↓
[https://youtu.be/mEYHFr3ZQhM](https://youtu.be/mEYHFr3ZQhM)

### 3. Installation

1. Before cloning the repository code, you need to install tools such as anaconda, PyCharm, and Git<br>
2. Use the following code to clone the code to the local, create a Python environment, and install dependent packages

```bash
git clone https://gitee.com/trgtokai/scratch-detect.git
cd scratch-detect
conda create -n yolov5_pyqt5 python=3.8
conda activate yolov5_pyqt5
pip install -r requirement-trg.txt
```
3. Torch and torchvision files are large and download slowly. You can download them from domestic sources (Tsinghua source/Ali source, etc.)<br>
Place the downloaded files in the D:\code\scratch-detect\install_torch\ folder. After placing them, you can use the following code to install them
```bash
pip install -r requirement-torch Local Installation.txt
```
If the installation fails, open requirement-torch Local Installation.txt and modify the file name to be consistent with the downloaded file<br>
4. If pycocotool compilation errors occur during the installation process, you need to download and install visual studio C++ build tools<br>
5. After the dependency is installed, run the program using the following code
```bash
python main.py
```

### 4. Functions

1. Support images, videos, multiple cameras and network rtsp streaming as input
2. Drop-down menu to change the training model
3. Drag the button to adjust IoU
4. Drag the button to adjust the confidence
5. Set the delay
6. Start, pause, stop functions (stop function BUG to be fixed)
7. Results statistics and real-time display in the Weiluntong touch screen
8. Automatically save after identifying the target image
9. Use ModbusRTU protocol to communicate with PLC
10. Trigger three-color lights and alarms when abnormal targets are identified
11. Use touch screen to control the start of the detection program
12. Automatic restart check when the program exits abnormally (PLC coil signal required)

**Run interface:**
![Enter image description](imgs/%E7%BA%BF%E4%B8%8A%E6%A3%80%E6%9F%A5%5B00_10_57%5D%5B20240605-174147%5D.png)

### 5. File composition Files

1. ./data/ —— Training script
2. ./pt/ —— Model file storage location
3. ./plc/ —— PLC project files and touch screen project files
4. ./runs/ —— operation result storage location
5. ./ui_files/ —— GUI visualization interface source code storage location
6. ./main.py —— main program
7. ./train.py —— model training program
8. ./requirement-trg.txt —— dependency package list
9. ./requirement-torch Local Installation.txt —— local dependency package list

### 6. Program structure Network

**The program structure is shown in the figure below:**
![Program structure diagram](imgs/%E7%A8%8B%E5%BA%8F%E7%BB%93%E6%9E%84%E5%9B%BE.png)

### 7. Model training
1. Use [labelImg](https://blog.csdn.net/klaus_x/article/details/106854136) to label the image
2. Modify the training set location of the training script in data, for details [click here](https://blog.csdn.net/qq_45945548/article/details/121701492)
3. Train a model file suitable for your own use, put the model file in ./pt and select it when the main program main.py is running

### 8. Contact Us
If there are any bugs or other suggestions, please send a private message to  @li-chey /  @Alex_Kwan 