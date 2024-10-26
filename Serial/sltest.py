import serial
import serial.tools.list_ports
import time
BAUD_RATE = 115200    # ボーレートをマイコン側と一致させる

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
# シリアルポートの設定
while True:
    try:
        com = input_com()
        ser = serial.Serial(com, BAUD_RATE, timeout=None)
        break
    except serial.SerialException:
        print("COMポートが開けませんでした。再度入力してください。")




# データを格納するリスト
data_list = []
timestamp_list = []

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # 現在時刻を取得
            timestamp = time.time()
            # データとタイムスタンプをリストに追加
            data_list.append(line)
            timestamp_list.append(timestamp)
            # 取得したデータを表示
            print(f"{timestamp}: {line}")

            # サンプリングレートの計算
            if len(timestamp_list) >= 2:
                # サンプリング間隔を計算
                interval = timestamp_list[-1] - timestamp_list[-2]
                # サンプリングレートを計算
                sampling_rate = 1 / interval
                print(f"Sampling Rate: {sampling_rate:.2f} Hz")
except KeyboardInterrupt:
    # 終了処理
    ser.close()
    print("終了しました。")
