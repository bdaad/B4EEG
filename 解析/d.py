def func_chank_all(receive_value, flag_blink_A, flag_blink_B, chank_list_A, chank_list_B, clock_signal_A, clock_signal_B, adjust_chank_list_A, adjust_chank_list_B, lock):
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    flag_state_A = None
    chank_chank_list_1_A = []
    chank_chank_list_2_A = []
    
    flag_state_B = None
    chank_chank_list_1_B = []
    chank_chank_list_2_B = []


    po = 0
    po2 = 0

    while True:
        if po >= 30:
            break
        #計測の最初は、必ずflag_blink_1=Trueのときにデータを受け取る.
        if flag_state_A is None:
            with lock:
                print("flag_blink_A: ", flag_blink_A.value)
                if flag_blink_A.value == True:
                    flag_state_A = True
        else:
            if flag_blink_A.value == True:
                if len(chank_chank_list_2_A) != 0:    
                    with lock:
                        chank_list_A.append(chank_chank_list_2_A)
                    chank_chank_list_2_A = []
                    po = po + 1
                    print("po15: ", po)
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list_A[-3:]))
                        adjust_chank_list_A.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po: ", po)      
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal_A.value == True:
                        chank_chank_list_1_A.append(receive_value[0])
                        clock_signal_A.value = False

            elif flag_blink_A.value == False:
                if len(chank_chank_list_1_A) != 0:    
                    with lock:
                        chank_list_A.append(chank_chank_list_1_A)
                    chank_chank_list_1_A = []
                    po = po + 1
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list_A[-3:]))
                        adjust_chank_list_A.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po: ", po)

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal_A.value == True:
                        chank_chank_list_2_A.append(receive_value[0])
                        clock_signal_A.value = False


        if flag_state_B is None:
            with lock:
                print("flag_blink_B: ", flag_blink_B.value)
                if flag_blink_B.value == True:
                    flag_state_B = True
        else:
            if flag_blink_B.value == True:
                if len(chank_chank_list_2_B) != 0:    
                    with lock:
                        chank_list_B.append(chank_chank_list_2_B)
                    chank_chank_list_2_B = []
                    # po2 = po2 + 1
                    # print("po15: ", po2)
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list_B[-3:]))
                        adjust_chank_list_B.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po2: ", po2)      
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal_B.value == True:
                        chank_chank_list_1_B.append(receive_value[0])
                        clock_signal_B.value = False

            elif flag_blink_B.value == False:
                if len(chank_chank_list_1_B) != 0:    
                    with lock:
                        chank_list_B.append(chank_chank_list_1_B)
                    chank_chank_list_1_B = []
                    # po2 = po2 + 1
                    with lock:
                        chank_list_copy = copy.deepcopy(list(chank_list_B[-3:]))
                        adjust_chank_list_B.append(adjust_data_to_size(chank_list_copy, target_size=667))
                    # print("po2: ", po2)

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal_B.value == True:
                        chank_chank_list_2_B.append(receive_value[0])
                        clock_signal_B.value = False
    # print("chank_list_A: ", chank_list_A)
    # テキストファイルにデータを追記
    # append_data_to_file(receive_data_txt, adjust_chank_list_A)
    print("len of chank_list_A 15Hz: ", len(chank_list_A))               
    # 各行の列数を出力
    for i, row in enumerate(chank_list_A):
        print(f"Row {i+1} length: {len(row)}")  # 各行の列数を出力
    print("adjust_chank_list_A")
    # 各行の列数を出力
    for i, row in enumerate(adjust_chank_list_A):
        print(f"Row {i+1} length: {len(row)}")

