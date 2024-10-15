
import cv2
import time

# ビデオキャプチャの初期化（画面キャプチャの場合、通常はカメラがデフォルトで0）
cap = cv2.VideoCapture(0)

# 計測開始
prev_time = time.time()
frame_count = 0

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
fps = frame_count / elapsed_time

print(f"計測したFPS（フレームレート）: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()
