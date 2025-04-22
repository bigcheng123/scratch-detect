import time

import modbus_rtu
import os
import sys
import serial

ser2 = serial.Serial('com9', 38400, 8, 'N', 1, 0.3)

class restart():
    def __init__(self):
        super(restart, self).__init__()

    def read_sensor(self):  ### 检查触发开关
        global ser2
        # modbus_rtu model ↓----------------------------------
        if not ser2 is None:
            sensor = modbus_rtu.writedata(ser2, '01 02 00 00 00 01 B9 CA')
            if sensor == '010201016048':
                os.system("start cmd.exe /K autorun_main.bat")
                ser2.close()
                sys.exit()
            else:
                ser2.close()
                sys.exit()

if __name__ == "__main__":
    restarts = restart()
    restarts.read_sensor()