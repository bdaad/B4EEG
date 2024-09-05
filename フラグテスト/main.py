# main.py

import flag_module

# フラグを操作する
print("Initial flag value:")
flag_module.print_flag()

# フラグをTrueに変更
flag_module.flag = True

print("Flag value after modification:")
flag_module.print_flag()
