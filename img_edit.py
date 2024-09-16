import os
from PIL import Image

# フォルダのパスを指定
folder_path = './解析/img_file/'

# 背景色を指定（白色: (255, 255, 255)、黒色: (0, 0, 0)など）
background_color = (255, 255, 255)  # 例: 白色

# フォルダ内の全てのPNG画像に対して処理を行う
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):  # PNGファイルのみ対象
        file_path = os.path.join(folder_path, filename)
        print(f"Processing {file_path}")

        # 画像を開く
        image = Image.open(file_path)

        if image.mode == 'RGBA':  # RGBAモード（アルファチャンネルあり）の場合
            # アルファチャンネルを背景色で埋める
            background = Image.new("RGB", image.size, background_color)
            background.paste(image, mask=image.split()[3])  # アルファチャンネルを使って貼り付け

            # 変換後の画像を保存（同じファイルに上書き保存）
            background.save(file_path)

            print(f"アルファチャンネルを削除し、{file_path} に保存しました。")
        else:
            print(f"{file_path} は既にRGB形式です。")

print("すべてのPNG画像の処理が完了しました。")
