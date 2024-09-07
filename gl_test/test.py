import multiprocessing
import time

# 1つ目のプロセス：リストに奇数を追加する
def add_odd_numbers(shared_list, lock, n):
    for i in range(1, n, 2):  # 奇数を追加
        with lock:  # ロックを使って排他制御
            shared_list.append(i)
            print(f"Added odd number {i} to the list")
        time.sleep(0.1)  # 遅延を入れて交互に追加

# 2つ目のプロセス：リストに偶数を追加する
def add_even_numbers(shared_list, lock, n):
    for i in range(0, n, 2):  # 偶数を追加
        with lock:  # ロックを使って排他制御
            shared_list.append(i)
            print(f"Added even number {i} to the list")
        time.sleep(0.1)  # 遅延を入れて交互に追加

if __name__ == '__main__':
    # 共有リストとロックを作成
    manager = multiprocessing.Manager()
    shared_list = manager.list()  # 共有リスト
    lock = multiprocessing.Lock()  # ロック

    n = 10  # 追加する数の上限

    # プロセスを作成
    p1 = multiprocessing.Process(target=add_odd_numbers, args=(shared_list, lock, n))
    p2 = multiprocessing.Process(target=add_even_numbers, args=(shared_list, lock, n))

    # プロセスを開始
    p1.start()
    p2.start()

    # 両プロセスが終了するのを待つ
    p1.join()
    p2.join()

    # 最終結果を出力
    print("Final shared list:", list(shared_list))
