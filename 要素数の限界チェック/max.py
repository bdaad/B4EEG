import sys

try:
    big_list = []
    i = 0
    while True:
        big_list.append(i)
        i += 1
        if i % 1000000 == 0:  # 100万要素ごとにメモリ使用量をチェック
            print(f"Current size: {len(big_list)}, memory usage: {sys.getsizeof(big_list)} bytes")
except MemoryError:
    print(f"MemoryError: Reached the maximum size of the list with {len(big_list)} elements.")
