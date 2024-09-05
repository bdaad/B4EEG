# 2024/09/02 うまく動いていないコード.

# import threading
# import time

# def three_count():
#     for i in range(1, 4):
#         print(i, end=',')
#         time.sleep(1)
    

# def run_three_count_in_threads():
#     while True:
#         thread = threading.Thread(target=three_count)
#         thread.start()
#         time.sleep(1)
#         print()

# if __name__ == "__main__":
#     run_three_count_in_threads()


# 0s         1s         2s         3s         4s         5s         6s         7s         8s         9s         10s .....
# |----------*----------*----------|----------*----------*----------|----------*----------*----------|.....
#            |----------*----------*----------|----------*----------*----------|----------*----------*----------|.....
#                       |----------*----------*----------|----------*----------*----------|----------*----------*----------|......


import threading
import time

colors = ["\033[91m", "\033[92m", "\033[94m"]
reset = "\033[0m"  # リセット

def three_count(color):
    for i in range(1, 4):
        print(f"{color}{i}{reset}", end=',')
        time.sleep(1)
    print()

def run_three_count_in_threads():
    while True:
        threads = []
        for i in range(3):
            thread = threading.Thread(target=three_count, args=(colors[i],))
            threads.append(thread)
            thread.start()
            time.sleep(1)  # 1秒ごとに次のスレッドを開始

        # 全スレッドが終了するのを待たずに、次のサイクルに移る
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    run_three_count_in_threads()
