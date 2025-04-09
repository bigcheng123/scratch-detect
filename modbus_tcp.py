# 导入pymodbus模块，使用前需要pip install pymodbus
import time

from pymodbus.client import ModbusTcpClient


# 通过IP及端口号开启连接
def modbustcp_open_port(ip, port):
    global client
    try:
        # 创建 Modbus TCP 客户端
        client = ModbusTcpClient(ip, port)
        # 连接到服务器
        connection = client.connect()

        if not connection:
            print("无法连接到 Modbus 服务器。")
            return client
    except Exception as e:
        print(f"连接发生错误：{e}")
        client = None
        return client
    else:
        ret = True  # ret:modbus连接成功标识，当ret为True时，表示modbus已连接，具备通讯能力
        error = "modbus-tcp connect success"
        return client, ret, error


def modbustcp_write_coil(address, write_value):
    try:
        # 写入线圈
        result = client.write_coil(address, write_value)
        print(write_value, type(write_value))
        if result.isError():
            print(f"写入失败：{result}")
        else:
            print(f"成功写入线圈 {address}，值：{value}")
    except Exception as e:
        print(f"写线圈发生错误：{e}")
    # finally:
    #     # 关闭连接
    #     connect_plc.client.close()


def modbustcp_read_coil(address, coil_number):
    result = client.read_coils(address, coil_number)
    return result


def modbustcp_write_registers(address, n, decimal_value):  # 起始地址， 写入寄存器个数，写入值
    # 确保输入值在32位范围内（0~2^32-1）
    if not (0 <= decimal_value < 2 ** 32):
        raise ValueError("Input value must be a 32-bit unsigned integer")
    # 获取低16位和高16位
    low_16_bits = decimal_value & 0xFFFF  # 低16位
    high_16_bits = (decimal_value >> 16) & 0xFFFF  # 高16位
    # 将两个16位数分别存入寄存器
    if n >= 2:
        write_value = [low_16_bits, high_16_bits]  # [寄存器1（低16位）  寄存器2（高16位）]
        # print('write_value = [low_16_bits, high_16_bits]')
        try:
            result = client.write_registers(address, write_value)
            # print(result)
            if result.isError():
                print(f"写入失败：{result}")
        except Exception as e:
            print(f"写寄存器发生错误：{e}")
    if n == 1:
        write_value = [low_16_bits]
        # print('write_value = [low_16_bits, high_16_bits]')
        try:
            result = client.write_register(address, write_value[0])
            # print(result)
            if result.isError():
                print(f"写入失败：{result}")
        except Exception as e:
            print(f"写寄存器发生错误：{e}")
        # else:
        #     print(f"成功写入寄存器 {address}，值：{write_value}")



def modbustcp_read_register(address, decimal_value):
    print('read the register form PLC')



def reset_register(address, zero_value):
    try:
        result = client.write_registers(address, zero_value)
        if result.isError():
            print(f"置0失败：{result}")
    except Exception as e:
        print(f"重置寄存器发生错误：{e}")


def modbustcp_client_close():
    try:
        result = client.close()
        if not result:
            print("成功关闭通讯")
        else:
            print(f"无法关闭modbus。")

    except Exception as e:
        print(f"连接发生错误：{e}")


if __name__ == "__main__":
    # 配置 Modbus 服务器地址和端口
    modbus_ip = "192.168.3.110"  # 替换为您的 Modbus 服务器 IP
    modbus_port = 502  # 默认 Modbus TCP 端口

    # 配置要写入的线圈地址和值
    coil_address = 0  # 线圈地址
    coil_value = True  # 要写入的值（True 或 False）

    register_address = 2
    value = 0
    # register_1, register_2 = split_into_registers(decimal_value)
    # register_value = [register_1, register_2]

    # 写入线圈
    modbustcp_open_port(modbus_ip, modbus_port)
    modbustcp_write_coil(8310, True)
    time.sleep(0.2)
    modbustcp_write_coil(8310, False)
    # modbustcp_write_register(register_address, value)
    # result = modbustcp_read_coil(8212, 1)
    # print(result)
    modbustcp_client_close()
