import re
import serial
import serial.tools.list_ports
import time

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

# データ受信カウンターとバッファ
count = 0
buffer = []

start_time = time.time()

while True:
    result = ser.readline()  # 改行コードまで読み込む
    if result:
        result = re.sub(rb'\r\n$', b'', result)  # 改行コードを削除
        buffer.append(result.decode())  # バッファに追加
        count += 1

    # 100個のデータが集まったら処理を行う
    if count >= 100:
        # ここで100データを処理する（例: 表示）
        for data in buffer[:100]:
            print(data)
        buffer = buffer[100:]  # 使用済みのデータをバッファから削除
        count = len(buffer)  # バッファに残ったデータの数でカウントをリセット

    # 一秒ごとに進行を確認する
    current_time = time.time()
    if current_time - start_time >= 1:
        if count < 100:
            print(f"1秒間で100データ未満が受信されました: {count}データ")
        start_time = current_time  # タイマーをリセット
        count = 0  # カウンターをリセット
        buffer.clear()  # バッファをクリア

ser.close()
