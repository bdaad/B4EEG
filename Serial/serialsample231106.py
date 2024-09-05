# 230426
# arduinoでシリアル通信テスト
import re
import serial
import serial.tools.list_ports
import time

# 
bitRate = 115200

def list_com():
    ports = serial.tools.list_ports.comports()
    print("利用可能なCOMポート:")
    for port in ports:
        print(port.device)


def input_com():
    com = input("COMポートを入力してください(例: COM3): ")
    return com




# COMポート一覧を表示
list_com()

while True:
    try:
        com = input_com()
        ser = serial.Serial(com, bitRate, timeout=None)
        break
    except serial.SerialException:
        print("COMポートが開けませんでした。再度入力してください。")





# ser = serial.Serial(COM, bitRate, timeout=None)

while True:
    result = ser.readline() # 改行コードまで読み込む
    if result:
        result = re.sub(rb'\r\n$', b'', result) # 改行コードを削除
        print(result.decode()) # バイト列を文字列に変換

ser.close()