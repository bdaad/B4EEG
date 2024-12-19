import numpy as np

def std__re_cv(data):
    # 平均を計算
    mean = sum(data) / len(data)
    
    # 各データポイントの偏差の二乗を計算
    squared_differences = [(x - mean) ** 2 for x in data]
    
    # 偏差の二乗の平均を計算
    variance = sum(squared_differences) / len(data)
    
    # 標準偏差を計算（分散の平方根）
    std = variance ** 0.5
    if std != 0:
        re_cv =  mean / std
    else:
        re_cv = mean / 0.0000000001 # 0除算を防ぐための処理
    
    return std, re_cv

# テスト用のデータセット
division = 5    #分割数
data_length = 8 #分割データの長さ

data_set = np.zeros((5, 8))
data_set[0] = [1, 12, 2, 23, 16, 23, 21, 16]
data_set[1] = [1, 2, 2, 4, 5, 6, 7, 8]
data_set[2] = [1, 1, 2, 1, 1, 1, 1, 1]
data_set[3] = [1, 2, 2, 4, 5, 6, 7, 8]
data_set[4] = [1, 2, 2.3, 4, 5, 6, 7, 8]

print(data_set)


# 行に対する標準偏差を計算して出力
std_dev = np.zeros(division)
re_cv_dev = np.zeros(division)

for i in range(division):
    std_dev[i], re_cv_dev[i] = std__re_cv(data_set[i])

print(f"標準偏差[行]: {std_dev}")
print(f"1/CV[行]: {re_cv_dev}")

# 列に対する標準偏差を計算して出力
std_dev = np.zeros(data_length)
re_cv_dev = np.zeros(data_length)

for i in range(data_length):
    std_dev[i], re_cv_dev[i] = std__re_cv(data_set[:, i])

print(f"標準偏差[列]: {std_dev}")
print(f"1/CV[列]: {re_cv_dev}")
