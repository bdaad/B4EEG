# マルチプロセス -> 並列処理
# python3.2以降 concurrent.futuresを使う


from concurrent.futures import ProcessPoolExecutor
import time


def func_1():
    for n in range(3):
        time.sleep(2)
        print(f'func_1 - {n}')

def func_2():
    for n in range(3):
        time.sleep(1)
        print(f'func_2 - {n}')

def main():
    with ProcessPoolExecutor(max_workers=2) as e:
        e.submit(func_1)
        e.submit(func_2)

if __name__ == '__main__':
    main()