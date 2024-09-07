from concurrent.futures import ProcessPoolExecutor
import time


def func_1(arg):
    for n in range(3):
        time.sleep(2)
        print(f'func_1 - {n}, arg: {arg}')

def func_2(arg):
    for n in range(3):
        time.sleep(1)
        print(f'func_2 - {n}, arg: {arg}')

def main():
    with ProcessPoolExecutor(max_workers=2) as e:
        e.submit(func_1, 'Argument for func_1')  # 引数を渡す
        e.submit(func_2, 'Argument for func_2')  # 引数を渡す

if __name__ == '__main__':
    main()
