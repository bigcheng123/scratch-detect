import pyodbc

# def connectsql(database):
# 连接数据库参数
server = 'TRG-327-PC'  # 替换为你的SQL Server服务器名或IP地址  DESKTOP-QGKNIRA\SQLEXPRESS
database = 'PE_DataBase'      # 数据库名
username = 'TRG-PE'           # 登录名
password = '705705'          # 密码


# def opensql():
# 构建连接字符串
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# 尝试连接数据库
try:
    conn = pyodbc.connect(conn_str)
    print("Successfully connected to the database.")
    sql_connection = True   # 输出SQL连接成功信号

except pyodbc.Error as e:
    print(f"Error connecting to database: {e}")
    sql_connection = False  # 输出SQL连接成功信号




# values1 = input("请输入时间")
# values2 = input("请输入异常类型")
# values3 = input("请输入输出状态")
def writesql(values1=None, values2=None, values3=None,  values4=None, values5=None, values6=None, values7=None,):

    # 确认SQL连接状态
    # if sql_connection:



    try:
        # conn = pyodbc.connect(conn_str)
        # print("Successfully connected to the database.")

        # 创建游标
        cursor = conn.cursor()

        # 执行插入示例
        cursor.execute("INSERT INTO AI_check_machine (时间, NG类型, 输出状态, 批次号, 作业员, 品番, 图片保存) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (f'{values1}', f'{values2}', f'{values3}', f'{values4}',
                        f'{values5}', f'{values6}', f'{values7}'))
        print()
        conn.commit()
        print("Data successfully inserted.")

        # 关闭游标和连接
        cursor.close()
        # conn.close()

    except pyodbc.Error as e:
        print(f"Error connecting to database: {e}")



def closesql():
    # cursor.close()
    conn.close()
    print("Disconnected from the database.")


def readsql(read_line = 10):
    try:
        # 创建游标
        cursor = conn.cursor()

        # 执行查询示例
        cursor.execute(f'SELECT TOP {read_line} * FROM [dbo].[AI_check_machine]')  # 替换为你的表名和查询语句

        # 获取查询结果
        rows = cursor.fetchall()
        for row in rows:
            print(row)

        # 关闭游标和连接
        cursor.close()

    except pyodbc.Error as e:
        print(f"Error connecting to database: {e}")

#
# read_line = input("请输入读取行数")
# readsql(read_line)

