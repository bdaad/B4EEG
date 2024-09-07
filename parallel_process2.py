import multiprocessing
import time

# 実行したい関数を定義
def worker(name, sleep_time):
    print(f"Worker {name} starting...")
    time.sleep(sleep_time)
    print(f"Worker {name} finished")

if __name__ == '__main__':
    # プロセスを作成
    process1 = multiprocessing.Process(target=worker, args=("A", 2))
    process2 = multiprocessing.Process(target=worker, args=("B", 1))

    # プロセスを開始
    process1.start()
    process2.start()

    # プロセスの終了を待機
    process1.join()  # process1が終了するまで待つ
    process2.join()  # process2が終了するまで待つ

    print("Both processes finished!")
