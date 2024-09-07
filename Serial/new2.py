# この方式がいいかもwindowsの場合は..

import re
import serial
import serial.tools.list_ports
import time

# シリアル通信のボーレート
bitRate = 115200

# COMポートの一覧表示
def list_com():
    ports = serial.tools.list_ports.comports()
    print("利用可能なCOMポート:")
    for port in ports:
        print(port.device)

# COMポートの入力
def input_com():
    com = input("COMポートを入力してください(例: COM3): ")
    return com

# 1000Hzでデータ要求を送信し、受信も行い、データの数をカウントする関数
def communicate_and_count(ser):
    interval = 1.0 / 1000  # 1000Hz
    next_time = time.perf_counter()  # 高精度タイマーの現在時刻を取得
    start_time = time.perf_counter()  # 計測開始時間
    data_count = 0  # データのカウント
    t = 1

    while True:
        current_time = time.perf_counter()  # 現在のタイムスタンプを取得

        # 10秒経過したらループを終了
        if current_time - start_time >= 1 * t:
            # 10秒間で受信したデータの数を表示
            print(f"100秒間で受信したデータの数: {data_count}")
            data_count = 0
            t = t + 1


        # 1000Hzでデータ要求を送信
        if current_time >= next_time:
            ser.write(b"req\n")  # Arduinoにデータ要求コマンドを送信
            next_time += interval  # 次の送信時間を設定

        # データを受信しカウント
        if ser.in_waiting > 0:  # 受信データがあるか確認
            result = ser.readline()  # 改行コードまで読み込む
            if result:
                data_count += 1  # データをカウント
                result = re.sub(rb'\r\n$', b'', result)  # 改行コードを削除
                # print(result.decode())  # バイト列を文字列に変換

        # 次のタイムスタンプまでの残り時間を計算
        sleep_time = next_time - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)  # 必要な場合のみスリープ

    

# COMポート一覧を表示
list_com()

# COMポート接続の初期化
while True:
    try:
        com = input_com()
        ser = serial.Serial(com, bitRate, timeout=None)
        break
    except serial.SerialException:
        print("COMポートが開けませんでした。再度入力してください。")

# データ要求を1000Hzで送信し、10秒間で受信したデータ数を計測
communicate_and_count(ser)

ser.close()
