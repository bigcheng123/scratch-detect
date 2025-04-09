import serial
import time


# raw_data 十进制格式列表:[站号,功能码,软元件地址,读写位数/数据]
def calculate_crc(raw_data):
    # 将 raw_data 转换为 hex_data
    hex_data = [format(x, 'X').zfill(4) for x in raw_data]
    string = hex_data[0]
    hex_data[0] = string[-2:]
    string = hex_data[1]
    hex_data[1] = string[-2:]
    try:
        hex_data[4]   # 确认是否有第五位→多寄存器读写
    except:
        print("no hex 4")
    else:
        string = hex_data[4]
        hex_data[4] = string[-2:]   # 多寄存器位数信息
        string = hex_data[5]
        hex_data[5] = string.zfill(8)   # 写入2个寄存器/补充至8位16进制数
        string = hex_data[5]
        string1 = string[-4:]     # 串口写入高低位与PLC高低位逻辑不一致，需要高低4位互换
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


def writedata(ser, hexcode):
    send_data = bytes.fromhex(hexcode)    # HEX码 转换 bytes 字节码     发送数据转换为b'\xff\x01\x00U\x00\x00V'
    ser.write(send_data)   # 发送命令
    time.sleep(0.02)        # 延时，否则len_return_data将返回0，此处易忽视！！！ 延迟低于 0.01无法接收数据
    len_return_data = ser.inWaiting()  # 获取缓冲数据（接收数据）长度
    return_data = ser.read(len_return_data)  # 读取缓冲数据
    # bytes(2进制)转换为hex(16进制)，应注意Python3.7与Python2.7此处转换的不同，并转为字符串后截取所需数据字段，再转为10进制
    str_return_data = str(return_data.hex())   # bytes(2进制)转换为hex(16进制
    # 返回数据
    return str_return_data

# 写两个寄存器，calculate自动高低位切换
print(calculate_crc([1, 16, 10, 2, 4, 888888]))


print(calculate_crc([1, 5, 13067, 0]))

# 打开串口COM4,根据自己的PLC修改COM口及波特率参数
ser2 = serial.Serial('COM4', 38400, 8, 'N', 1, 0.3)

# Y13线圈写入闭合
writedata(ser2, calculate_crc([1, 5, 13067, 65280]))
# 相当于 writedata(ser2, '01 05 33 0B FF 00 F2 BC')

# Y13线圈写入断开
writedata(ser2, calculate_crc([1, 5, 13067, 0]))
# 相当于 writedata(ser2, '01 05 33 0B 00 00 B3 4C')
