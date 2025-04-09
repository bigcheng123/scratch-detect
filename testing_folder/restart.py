import time

import modbus_rtu
import os
import sys

class restart():
    def __init__(self):
        super(restart, self).__init__()
        self.ser, self.ret, error = modbus_rtu.openport(port='COM8', baudrate=9600, timeout=2)
        # self.thread_mudbus_run()
    def calculate_crc(self, raw_data):  # raw_data 十进制格式列表： [站号 , 功能码, 软元件地址 , 读写位数/数据] 示例raw_data = [1, 6, 10, 111]
        # 将 raw_data 转换为 hex_data
        hex_data = [format(x, 'X').zfill(4) for x in raw_data]
        string = hex_data[0]
        hex_data[0] = string[-2:]
        string = hex_data[1]
        hex_data[1] = string[-2:]
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


    def thread_mudbus_run(self):
        global modbus_flag
        modbus_flag = True

        if self.ret: ### openport sucessfully
            read_m21 = self.calculate_crc([1, 1, 21, 1])  # 预留触摸屏开关用，读取M11线圈闭合状态，
            while modbus_flag:
                # print("in loop")
                # print(read_m21)
                time.sleep(0.2)
                m21_result = modbus_rtu.writedata(self.ser, read_m21)  # 如果返回值为：'01 01 0B 01 8C 08'，启动检查；为'01 01 0B 00 8C 08'停止检查
                # print(m21_result)
                if m21_result == '010101019048':
                    # print("true bat")
                    # start_dire = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\StartUp\autorun_main.bat"
                    os.system("start cmd.exe /K autorun_main.bat")
                    sys.exit()
                    # os.system('start cmd.exe /K C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\StartUp\\autorun_main.bat')
                    break
                else:
                    sys.exit()
                    break

if __name__ == "__main__":
    restarts = restart()
    # modbus_rtu.openport(port='COM8', baudrate=9600, timeout=2)
    restarts.thread_mudbus_run()