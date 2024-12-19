import time


g = [1,2,3,4,5,6,7,8,9,10]

print(g)

for i in range(5):
    #3番目の要素を0に変更
    g[2] = 0
    print(g)
    time.sleep(3)

print(g)