# -*- coding:utf-8 -*-
# Author:
# Windows7&Python3.7

import serial  ###pip install pyserial
import time
import cv2
import threading
import random

STRGLO =  "" #读取的数据
serflag = True  #读取标志位
#读数代码本体实现
def ReadData(ser):
    global STRGLO , serflag
    # 循环接收数据，此为死循环，可用线程实现
    while serflag:
        if ser.in_waiting:
            STRGLO = ser.read(ser.in_waiting).hex() ###读取到的数据是 二进制  转换位hex() 16进制
            # print(STRGLO)
#打开串口
# 端口，GNU / Linux上的/ dev / ttyUSB0 等 或 Windows上的 COM3 等
# 波特率，标准值之一：50,75,110,134,150,200,300,600,1200,1800,2400,4800,9600,19200,38400,57600,115200
# 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）

def openport(port,baudrate,timeout):  ### 降低PORT口的 数据缓存设定 可以提高通信质量
    ret = False
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)   # 选择串口，并设置波特率
        if not ser.is_open:
            ser.open()
            # threading.Thread(target=ReadData, args=(ser,)).start()
    except Exception as error:
        print("---OPEN PORT ERROR---：", error)
        ser = None
        ret = False
        return ser, ret, error
    else:
        ret = True
        error = "PORT SUCCESSFULLY"
        return ser, ret, error

#关闭串口
def DColsePort(ser):
    global serflag
    serflag = False
    ser.close()

#写数据
def DWritePort(ser,text):
    result = ser.write(bytes.fromhex(text))  # 写数据  使用 HEX TO  bytes    16进制 转 二进制
    return result   #### 返回代码

#读数据
def DReadPort():
    global STRGLO
    str = STRGLO
    STRGLO = ""  #清空 当次  读取
    return str

    # serial .isOpen()

def writedata(ser, hexcode):
    # lock= threading.Lock()
    # openport()
    # if (ser.is_open):
    # lock.acquire()
    str_return_data = ""
# try:
    # time.sleep(0.2)
    # time.sleep(random.random()*0.5) # 每次write port要 分隔开  防止同时进行 写操作 间隔越大
    send_data = bytes.fromhex(hexcode)    # HEX码 转换 bytes 字节码     发送数据转换为b'\xff\x01\x00U\x00\x00V'
    ser.write(send_data)   # 发送命令
    time.sleep(0.02)        # 延时，否则len_return_data将返回0，此处易忽视！！！ 延迟低于 0.01无法接收数据
    len_return_data = ser.inWaiting()  # 获取缓冲数据（接收数据）长度
    # if len_return_data:
    return_data = ser.read(len_return_data)  # 读取缓冲数据
    # bytes(2进制)转换为hex(16进制)，应注意Python3.7与Python2.7此处转换的不同，并转为字符串后截取所需数据字段，再转为10进制
    str_return_data = str(return_data.hex())   # bytes(2进制)转换为hex(16进制
    # feedback_data = int(str_return_data[-6:-2], 16) ### j截取所需字段
    # feedback_data = int(str_return_data[-6:-2], 16)
    # print(feedback_data)
    # print(str_return_data)
# except Exception as e:
#     print('writedata error',e)       
# else:
#     pass
#     global str_return_data
    return(str_return_data)  ##返回数据

    # lock.release()
# else:
    #     print("open failed")


# def closeport(ser):
#     ser.colse()
#     print("port close success")
def str2bool(feedback_data):
    # hexcode = 'FE 02 00 01 00 01 FC 05' ### READ IN2 
    # feedback_data = openport(hexcode)
    # print('feedback_data',feedback_data)
    BOOL =  False
    try:
        time.sleep(random.random()*0.5) #加入随机延迟 错开信号
        if feedback_data:
            str_result = int(feedback_data[6:8], 10)  ###取目标字符   ##返回的代码已经时10进制 010201016048
            # print('feedback_data',str_result)
            if str_result == 1:
                BOOL = True
            else:
                BOOL = False 
            # ser.close()  ### 关闭串口
            # print("port close success")  
    except Exception as e:
        print('readdata error',e)  
    else:
        pass

    return (BOOL)


if __name__ == '__main__':
    #### hexcode  ######
    IN0_READ = '01 02 00 00 00 01 B9 CA'
    IN1_READ = '01 02 00 01 00 01 E8 0A'
    IN2_READ = '01 02 00 02 00 01 18 0A'
    IN3_READ = '01 02 00 03 00 01 49 CA'
    DO0_ON = '01 05 00 00 FF 00 8C 3A'
    DO0_OFF = '01 05 00 00 00 00 CD CA'
    DO1_ON = '01 05 00 01 FF 00 DD FA'
    DO1_OFF = '01 05 00 01 00 00 9C 0A'
    DO2_ON = '01 05 00 02 FF 00 2D FA'
    DO2_OFF = '01 05 00 02 00 00 6C 0A'
    DO3_ON = '01 05 00 03 FF 00 7C 3A'
    DO3_OFF = '01 05 00 03 00 00 3D CA'

    DO_ALL_ON = '01 0F 00 00 00 04 01 FF 7E D6'
    DO_ALL_OFF = '01 0F 00 00 00 04 01 00 3E 96'  ##OUT1-4  OFF  全部继电器关闭  初始化

    # read_in1()
    ser, ret, _ = openport(port='COM5', baudrate=9600, timeout=5) #打开端口port,baudrate,timeout
    n=10
    str_result=''

    while  n:
        # t = threading.Thread(target= writedata,args=(ser,'01 02 00 00 00 01 B9 CA'))
        # print(t)
        # t.start()
        feedback_data_IN1 = writedata(ser,IN0_READ)  #### 检查IN1 触发 返回01020100a188
        DAM4040_IN1 = feedback_data_IN1[0:8] ##读取字符
        print('IN1 feedback_data',feedback_data_IN1) 
        print('IN1', DAM4040_IN1)
        if DAM4040_IN1 == '01020101':  ####010201016048
            # 改变指示灯------------------------------------
            # # self.radioButton_ready.setChecked(False) #ready off
            # self.radioButton_run.setChecked(True) #  run  on
            # # self.radioButton_stop.setChecked(False) #stop  off
            # self.label_image1.setText('checking')
            # self.label_image2.setText('checking')
            # self.label_image3.setText('checking')
            on_run = writedata(ser,DO2_ON) ###2号继电器打开  运行中
        if not feedback_data_IN1:  ##如果接收到的数据位 None  则输出异常信号
            no_feedback = writedata(ser,DO3_ON) ###3号继电器打开   控制器无返回数据

        feedback_data = writedata(ser,DO1_ON)  ###1号继电器打开  运行准备
        # feedback_data = mymodbus.writedata(self.ser,'01 05 00 01 00 00 9C 0A')  ###2号继电器关闭  运行中信号关闭
            # self.lineEdit_result.setText('停止')
        
        feedback_data_IN2 = writedata(ser,IN2_READ)  ### IN2 读取
        DAM4040_IN2 =  feedback_data_IN2[0:8] ##读取BOOL值  返回代码
        print('IN2 feedback_data',feedback_data_IN2)
        print('IN2',DAM4040_IN2)
        if DAM4040_IN2 == '01020101':  #010201016048
            print('in2_on')
            # 改变指示灯------------------------------------

        if not feedback_data_IN2:  ##如果接收到的数据位 None  则输出异常信号
            no_feedback = writedata(ser,DO3_ON) ###3号继电器打开   控制器无返回数据


        if cv2.waitKey(1) == ord('q'):
            print('quit')
            break
        n-=1
        print(n)
    # colse_all_coil = writedata(ser,'01 0F 00 00 00 04 01 00 3E 96')
    open_all_coil = writedata(ser, DO_ALL_OFF)

    ser.close()
    print(('sel.close'))


####    功能码  IN 读取操作
# FE 02 00 00 00 04 6D C6 ## 读取4路IN
# FE 02 01 01 50 5C ##  IN1_ON
# FE 02 01 02 10 5D ##  IN2_ON
# FE 02 01 04 90 5F ##  IN3_ON
# FE 02 01 08 90 5A ##  IN4_ON


# FE 02 00 00 00 01 AD C5  ## READ IN1
# FE 02 01 00 91 9C  ## IN1_OFF
# FE 02 01 01 50 5C  ## IN1_ON

# FE 02 00 01 00 01 FC 05 ### READ IN2
# FE 02 01 00 91 9C  ### IN2_OFF
# FE 02 01 01 50 5C  ### IN2_ON

# FE 02 00 02 00 01 0C 05  ### READ IN3
# FE 02 01 00 91 9C ## IN3_OFF
# FE 02 01 01 50 5C  ## IN3_ON

# FE 02 00 03 00 01 5D C5 ## READ IN4
# FE 02 01 00 91 9C ## IN4_OFF
# FE 02 01 01 50 5C  ## IN4_ON


# #####  线圈写操作
# 01 0F 00 00 00 04 01 00 3E 96  # 全部COIL关闭
# #返回码 01 0F 00 00 00 04 54 08 
# 01 0F 00 00 00 04 01 FF 7E D6  # 全部COIL打开
# #返回码 01 0F 00 00 00 04 54 08