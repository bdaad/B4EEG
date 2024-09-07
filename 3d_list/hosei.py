import numpy as np

# サンプルデータ (1, 2, 3, ... のような数値データ)
da = [
    [1, 2, 3, 4, 5, 6],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
    [71, 72, 73, 74, 75, 76, 77, 78],
    [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    [91, 92, 21, 22, 23, 24, 25, 26, 27, 28]
]

# すべての列を10個のデータに揃える処理
def adjust_data_to_10(data):
    # 行数を取得
    row = len(data)
    if row <= 1:
        return data
    else:
        print(len(data[row - 1]))
        # 最後の行の要素数が10未満の場合
        if len(data[row - 1]) < 10:
            num = 10 - len(data[row - 1])
            print("num", num)
            # 最後の行の要素数が10未満の場合、最後の行の要素数を10個にする
            data[row - 1] = data[row - 2][len(data[row - 1]) : 10] + data[row - 1]
            return data[row - 1]
        elif len(data[row - 1]) > 10:
            # 最後の行の要素数が10より大きい場合、最後の行の要素数を10個にする
            remainder = len(data[row - 1]) - 10
            data[row - 1] = data[row - 1][int(remainder/2):int(remainder/2) + 10]
            return data[row - 1]
        else:
            return data[row - 1]





def main():
    # すべての列を10個のデータに揃える
    data = adjust_data_to_10(da)
    print(data)

if __name__ == '__main__':
    main()
