from datetime import datetime

def save_2d_array_to_file(data):
    # 現在の日時を取得してファイル名に使用
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"./data_{current_datetime}.txt"
    
    # ファイルを作成して二次元配列データを保存
    with open(file_name, "w") as file:
        for row in data:
            file.write(",".join(row) + "\n")
    
    print(f"{file_name} に二次元配列データを保存しました。")
    return file_name  # 保存したファイル名を返す

# 使用例
data = [
    ["apple", "banana", "cherry"],
    ["dog", "elephant", "fox"],
    ["grape", "honeydew", "iceberg"]
]

save_2d_array_to_file(data)
