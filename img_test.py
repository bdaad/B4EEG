from PIL import Image

# 画像を開く
# image = Image.open('./解析/img_file/a_off.png')
image = Image.open('./解析/img_file/circle_2.png')

# 画像の基本情報を表示
print(f"Format: {image.format}")
print(f"Size: {image.size}")
print(f"Mode: {image.mode}")

# アルファチャンネルがあるかどうか確認
if image.mode == 'RGBA':
    print("This image has an alpha channel.")
else:
    print("This image does not have an alpha channel.")
