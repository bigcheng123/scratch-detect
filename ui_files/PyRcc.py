
from PyQt5 import QtCore
import subprocess
# 编译.rcc文件
qrc_path = "apprcc.qrc"
output_path = "apprcc_rc.py"
# QtCore.qrcc_compile(qrc_path , output_path )

# 编译.rcc文件
subprocess.call(["pyrcc5", qrc_path, "-o", output_path])
