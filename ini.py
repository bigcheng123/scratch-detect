

import configparser

# 创建一个ConfigParser对象
config = configparser.ConfigParser()

# 添加一个section（类似分组的概念）
config['DEFAULT'] = {'debug': 'False', 'log_level': 'INFO'}
config['Database'] = {'host': 'localhost', 'port': '5432', 'user': 'admin', 'password': '123456'}

# 将配置写入到文件中，这里假设保存为config.ini
with open('./config/config.ini', 'w') as configfile:
    config.write(configfile)

# 从文件中读取配置
config.read('config.ini')
# 获取特定section下的参数
print(config['Database']['host'])
print(config['DEFAULT']['log_level'])