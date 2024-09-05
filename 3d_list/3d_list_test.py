

import numpy as np

# (2, 3, 4) の配列を作成
original_array = np.random.rand(2, 3, 4)  # ランダムな値で初期化

print("元の配列:")
print(original_array)

# 新しい行を作成 (2, 1, 4) の形状を持つ新しい行
new_row = np.random.rand(2, 1, 4)  # ランダムな値で初期化

# 新しい行を追加して (2, 4, 4) の形状にする
expanded_array = np.concatenate((original_array, new_row), axis=1)

print("\n拡張された配列:")
print(expanded_array)



