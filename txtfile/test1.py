import time
from datetime import datetime

# リストの名前
list_name = "my_list"

# 現在の日時をファイル名に追加
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"{list_name}_{current_time}.txt"

# 5つのデータ配列
data_list_1 = ["1", "2", "3"]
data_list_2 = ["4", "5", "6"]
data_list_3 = ["7", "8", "9"]
data_list_4 = ["10", "11", "12"]
data_list_5 = ["13", "14", "15"]


def numappend_data_to_file(num):
    with open(file_name, 'a') as file:
        # 配列1

        file.write(f"{num}, ")

        file.write("\n")  # 改行


# ファイルにデータを1秒ごとに追加する関数
def append_data_to_file(list):

    with open(file_name, 'a') as file:
        # 配列1
        for data in list:
            file.write(f"{data}, ")
            print(f"{data} をファイルに書き込みました")
            time.sleep(0.1)
        file.write("\n")  # 改行


#行数をカウントする関数
def count_lines(file_name):
    line_count = 0
    with open(file_name, 'r') as file:
        for line in file:
            line_count += 1
    return line_count



# ファイルから先頭の数値を読み込み、配列に格納する関数
def read_first_column_from_file(file_name):
    numbers = []  # 先頭の数値を格納する配列
    with open(file_name, 'r') as file:
        for line in file:
            # カンマで区切って最初の数値を取得
            first_number = int(line.split(',')[0].strip())
            numbers.append(first_number)
    return numbers


# i行目からj行目までの先頭の数値を配列に格納する関数(スタートは０)
def read_first_column_from_range(file_name, i, j):
    numbers = []  # 先頭の数値を格納する配列
    with open(file_name, 'r') as file:
        lines = file.readlines()  # ファイルの全行を読み込む
        # 指定された範囲(i行目からj行目まで)を処理
        for index in range(i, j+1):  # i, jは1から数えるので-1して調整
            line = lines[index]
            first_number = int(line.split(',')[0].strip())  # カンマで区切って最初の数値を取得
            numbers.append(first_number)
    return numbers



# 関数を実行
def func_1():
    append_data_to_file(data_list_1)
    append_data_to_file(data_list_2)
    append_data_to_file(data_list_3)
    append_data_to_file(data_list_4)
    append_data_to_file(data_list_5)
    numappend_data_to_file(100)
    numappend_data_to_file(200)
    numappend_data_to_file(300)
    numappend_data_to_file(400)
    new_list = read_first_column_from_file(file_name)
    print("new_list")
    print(new_list)
    new_list = read_first_column_from_range(file_name, 2, 5)
    print("new_list")
    print(new_list)


def func_2():
    while True:
        line_count = count_lines(file_name)
        print(f"{line_count}行")
        time.sleep(1)


from concurrent.futures import ProcessPoolExecutor
import time


def main():
    with ProcessPoolExecutor(max_workers=2) as e:
        e.submit(func_1)
        e.submit(func_2)

if __name__ == '__main__':
    main()

