import cv2
import time

# ビデオキャプチャの初期化（カメラがデフォルトで0）
cap = cv2.VideoCapture(0)

# フレームカウントを初期化
frame_count = 0

# 計測開始
prev_time = time.time()

while frame_count < 100:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームカウントを増やす
    frame_count += 1

    # フレームを表示
    cv2.imshow('frame', frame)

    # エスケープキーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了時刻を記録
end_time = time.time()

# フレームレートを計算
elapsed_time = end_time - prev_time

# elapsed_time が 0 になっていないか確認
if elapsed_time > 0:
    fps = frame_count / elapsed_time
    print(f"計測したFPS（フレームレート）: {fps:.2f}")
else:
    print("エラー: 経過時間が 0 です。")

cap.release()
cv2.destroyAllWindows()
