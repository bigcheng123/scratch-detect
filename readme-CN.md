### Readme English [https://gitee.com/trgtokai/scratch-detect/blob/master/readme.md](https://gitee.com/trgtokai/scratch-detect/blob/master/readme.md)
### 中文 Chinese [https://gitee.com/trgtokai/scratch-detect/blob/master/readme-CN.md](https://gitee.com/trgtokai/scratch-detect/blob/master/readme-CN.md)
### 日本語 Japanese [https://gitee.com/trgtokai/scratch-detect/blob/master/readme-JP.md](https://gitee.com/trgtokai/scratch-detect/blob/master/readme-JP.md)
…………………………………………………………………………………………………………………………………………………………………………………………………………
## 项目名称： 【Scratch-detect】伤痕检测  

### 一、提示 Tips <br>

这个程序基于[YOLOv5 v6.1](https://github.com/ultralytics/yolov5/tree/v6.1)

**硬件平台：** <br>
摄像头：海康MV-CU050-90UC<br>
PLC:三菱FX3S＋modbus模块<br>
触摸屏：威纶通MT8072IP<br>
光源：海康环形光源MV-LRSS-H-80-W<br>
光源控制器：[数字8通道光源控制器](https://detail.tmall.com/item.htm?abbucket=1&id=656543446110&rn=21d65f2d271defe4d3b29e10ced9b2a5&spm=a1z10.5-b.w4011-23573612475.52.201646d6ZWIsQh&skuId=4738283905874)<br>

**主机建议配置 Hardware：**<br>
1.CPU：i7 13700k及以上<br>
2.显卡：RTX3050及以上(只支持NVIDIA显卡)<br>
3.内存：建议16G以上<br>
4.硬盘：建议1TB及以上<br>

**软件平台 Software：**<br>
1.系统：win10 x64 <br>
2.驱动：[海康MVS](https://www.hikrobotics.com/cn2/source/support/software/MVS_STD_4.3.2_240529.zip)、[威纶通EBPRO](https://www.weinview.cn/Admin/Others/DownloadsPage.aspx?nid=3&id=10917&tag=0&ref=download&t=a4ff8b5703a191fe)、[NVIDIA显卡驱动](https://cn.download.nvidia.com/Windows/555.99/555.99-desktop-win10-win11-64bit-international-nsd-dch-whql.exe)、[三菱GXWORKS2](https://www.mitsubishielectric-fa.cn/site/file-software-detail?id=18)等<br>
3.Python环境：[anaconda](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Windows-x86_64.exe)<br>
4.PythonIDE：[PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC)<br>
5.Python版本：Python3.8<br>
6.版本控制：[Git](https://git-scm.com/download/win)

**AI检测程序 Code：**<br>
1. 克隆/下载 此项目全部文件到本地电脑
2. 使用 Pycharm 或其他 IDE软件 打开此项目
3. 配置环境 并确保所有依赖包都成功安装（清单=requirement-trg.txt)
4. 运行主程序 main.py
5. 如果希望运行你自己的检测模型，请把训练好的模型文件 放入 pt文件夹，然后在UI界面中选择对应的PT文件

### 二、演示视频 Demo
中国地区 Bilibili Demo ↓
[https://www.bilibili.com/video/BV1nz421S7KR](https://www.bilibili.com/video/BV1nz421S7KR)

中国以外地区 YouTube Demo ↓
[https://youtu.be/mEYHFr3ZQhM](https://youtu.be/mEYHFr3ZQhM)

### 三、安装方式 Installation

1.在克隆仓库代码前，需要安装好anaconda、PyCharm、Git等工具<br>
2.使用以下代码，将代码克隆至本地，创建Python环境，安装依赖包

```bash
git clone https://gitee.com/trgtokai/scratch-detect.git
cd scratch-detect
conda create -n yolov5_pyqt5 python=3.8
conda activate yolov5_pyqt5
pip install -r requirement-trg.txt
```
3.torch及torchvision文件较大，下载较缓慢，可通过国内源（清华源/阿里源等）进行下载<br>
并将下载好的文件放置在D:\code\scratch-detect\install_torch\文件夹内、放入后可使用以下代码进行安装
```bash
pip install -r requirement-torch Local Installation.txt
```
如果出现安装失败，打开requirement-torch Local Installation.txt修改文件名与下载的文件一致即可<br>
4.安装过程中出现pycocotool编译报错则需要下载并安装visual studio C++ build tools<br>
5.依赖安装完成后，使用以下代码运行程序
```bash
python main.py
```

### 四、功能 Functions

1. 支持图片、视频、多摄像头及网络rtsp串流作为输入
2. 下拉菜单更换训练模型
3. 拖动按钮调节 IoU
4. 拖动按钮调节 置信度
5. 设置延时
6. 启动、暂停、停止功能（停止功能BUG待修复）
7. 结果统计并实时显示在威纶通触摸屏中
8. 识别到目标图片后自动保存
9. 使用ModbusRTU协议与PLC进行通讯 
10. 识别到异常目标时触发三色灯及警报
11. 使用触摸屏控制检测程序启动
12. 程序异常退出自动重启检查（需PLC线圈信号）

**运行界面：**
![输入图片说明](imgs/%E7%BA%BF%E4%B8%8A%E6%A3%80%E6%9F%A5%5B00_10_57%5D%5B20240605-174147%5D.png)

### 五、文件构成 Files

1. ./data/ —— 训练脚本
2. ./pt/ —— 模型文件储存位置
3. ./plc/ —— PLC工程文件及触摸屏工程文件
4. ./runs/ —— 运行结果储存位置
5. ./ui_files/ —— GUI可视化界面源代码储存位置
6. ./main.py —— 主程序
7. ./train.py ——模型训练程序
8. ./requirement-trg.txt —— 依赖包清单
9. ./requirement-torch Local Installation.txt —— 本地依赖包清单

### 六、程序结构 Network

 **程序结构见下图：** 
![程序结构图](imgs/%E7%A8%8B%E5%BA%8F%E7%BB%93%E6%9E%84%E5%9B%BE.png)

### 七、模型训练 Trains
1. 使用[labelImg](https://blog.csdn.net/klaus_x/article/details/106854136)对图片进行标注
2. 修改data中的训练脚本的训练集位置，详细[点击这里](https://blog.csdn.net/qq_45945548/article/details/121701492)
3. 训练适合自己使用的模型文件，将模型文件放入./pt中即可在主程序main.py运行时选取使用

### 八、联系我们 
如果程序有BUG或其他建议，可通过本站私信@li-chey /  @Alex_Kwan 