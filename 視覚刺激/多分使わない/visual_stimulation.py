

import tkinter as tk

def toggle_circle():
    """円の表示と非表示を切り替える"""
    global is_visible
    if is_visible:
        canvas.itemconfig(circle, state='hidden')
    else:
        canvas.itemconfig(circle, state='normal')
    is_visible = not is_visible

def start_blinking():
    """10Hzで2秒間点滅させる"""
    for i in range(20):  # 10Hzで2秒間 = 20回点滅
        root.after(i * 100, toggle_circle)
    # root.after(2000, lambda: root.destroy())  # 2秒後にウィンドウを閉じる

# ウィンドウの初期化
root = tk.Tk()

# 全画面表示
root.attributes("-fullscreen", True)

# 画面の幅と高さを取得
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# キャンバスの作成
canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack()

# 円の半径
radius = 100

# 円を画面の中心に描画
x_center = screen_width // 2
y_center = screen_height // 2
circle = canvas.create_oval(x_center - radius, y_center - radius, x_center + radius, y_center + radius, fill="blue")

# 円の表示状態を管理する変数
is_visible = True

# 点滅を開始
start_blinking()

# メインループの開始
root.mainloop()
