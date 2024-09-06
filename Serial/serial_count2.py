import re
import serial
import serial.tools.list_ports
import time

# 通信速度
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

# データ受信回数のカウンター
count = 0
start_time = time.perf_counter()  # 高精度なタイマーに変更

while True:
    result = ser.readline()  # 改行コードまで読み込む
    if result:
        result = re.sub(rb'\r\n$', b'', result)  # 改行コードを削除
        result.decode()  # バイト列を文字列に変換
        count += 1  # カウントを増やす

    # 一秒ごとにカウントを出力
    current_time = time.perf_counter()  # 高精度なタイマーに変更
    if current_time - start_time >= 600:
        sampling_rate = count / (current_time - start_time)  # サンプリングレート計算
        print(f"1秒間に受信したデータ数: {count}回, サンプリングレート: {sampling_rate} サンプル/秒")
        count = 0  # カウンターをリセット
        start_time = current_time  # タイマーをリセット

ser.close()
