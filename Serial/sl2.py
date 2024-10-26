import serial
import time
from threading import Thread
import serial.tools.list_ports
# 通信速度
# 
# bitRate = 115200
bitRate = 230400

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
# データカウント用の変数
data_count = 0
sampling_rate = 0

# スレッドを使用してサンプリングレートを1秒ごとに計算・出力
def sampling_rate_calculator():
    global data_count, sampling_rate
    next_call = time.time()
    while True:
        # Schedule the next call time
        next_call += 1
        # Sleep until the next scheduled time
        time.sleep(max(0, next_call - time.time()))
        sampling_rate = data_count
        data_count = 0
        print(f"Sampling Rate: {sampling_rate} Hz")


# スレッドの開始
thread = Thread(target=sampling_rate_calculator)
thread.daemon = True
thread.start()

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # データを受信したらカウントを増やす
            data_count += 1
            # 取得したデータを表示
            # print(f"Received Data: {line}")
except KeyboardInterrupt:
    # 終了処理
    ser.close()
    print("終了しました。")
