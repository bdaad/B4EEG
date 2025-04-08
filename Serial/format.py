import serial
import serial.tools.list_ports
import re
import time

# シリアル通信のビットレート
bitRate = 115200

# COMポート一覧を表示する関数
def list_com():
    ports = serial.tools.list_ports.comports()
    print("利用可能なCOMポート:")
    for port in ports:
        print(port.device)

# COMポートを入力させる関数
def input_com():
    com = input("COMポートを入力してください(例: COM3): ")
    return com

# COMポート一覧を表示
list_com()

# シリアルポートを開く
while True:
    try:
        com = input_com()
        ser = serial.Serial(com, bitRate, timeout=None)
        break
    except serial.SerialException:
        print("COMポートが開けませんでした。再度入力してください。")

# 「数値,数値,数値」形式(マイナス含む可能性)を表す正規表現パターン
pattern = re.compile(r'^-?\d+,-?\d+,-?\d+$')

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # パターンにマッチする場合はValid、しない場合はInvalidとして出力
            if pattern.match(line):
                # print(f"Valid Data  : {line}")
                None
            else:
                print(f"Invalid Data: {line}")
                time.sleep(1)
                

except KeyboardInterrupt:
    # Ctrl + C押下などでプログラムを終了するときの処理
    ser.close()
    print("終了しました。")
