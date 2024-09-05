flag = True



def main(flag):
    if flag:
        print("Flag is: True")
    else:
        print("Flag is: False")
    flag = False

    if flag:
        print("Flag is: True")
    else:
        print("Flag is: False")

    return flag


if __name__ == "__main__":

    flag = main(flag) #これでフラグが更新される.

    if flag:
        print("Flag is: True")
    else:
        print("Flag is: False")