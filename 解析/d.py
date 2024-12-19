def func_chank(priority, receive_value, flag_blink, chank_list, clock_signal, adjust_chank_list, analysis_flag, chank_size, lock, receive_value2, chank_list2, adjust_chank_list2):
    """
    1000data / 3Hz = 333.333data = 334data : 60/3 = 20
    1000data / 5Hz = 200data : 60/5 = 12                       採用(未実験)
    1000data / 6Hz = 166.666data = 167data : 60/6 = 10         採用(未実験)
    1000data / 10Hz = 100data : 60/10 = 6                      採用
    1000data / 12Hz = 83.3333data = 83data : 60/12 = 5         採用(未実験)
    1000data / 15Hz = 66.6666data = 67data : 60/15 = 4         採用(うまくいかなかった感じがする)
    1000data / 20Hz = 50data : 60/20 = 3
    1000data / 30Hz = 33.3333data = 34data : 60/30 = 2

    """
    # とりあえず０ｃｈのデータのみを処理する。受け取るデータはch0, 1,2である..
    p = psutil.Process()
    p.nice(priority)  # psutilで優先順位を設定
    print(f"Process (func_chank) started with priority {priority}")
    flag_state = None
    chank_chank_list_1 = [] #buffer1 (higi q)
    chank_chank_list_2 = [] #buffer2 (higi q)

    chank_chank_list_1_2 = [] #buffer1 (usual q)
    chank_chank_list_2_2 = [] #buffer2 (usual q)

    pretime = time.time()
    current_time = 0;
    # po = 0

    print("chank_size: {chank_size}")
    print("chank_size: {chank_size}")
    print("chank_size: {chank_size}")



    while True:
        if flag_state is None:
            with lock:
                print("first flag_blink: ", flag_blink.value)
                if flag_blink.value == True:
                    flag_state = True
        else:
            if flag_blink.value == True:
                if len(chank_chank_list_2) != 0:
                    current_time = time.time()
                    interval_time = current_time - pretime
                    
                    with lock:
                        chank_list.append(chank_chank_list_2)
                        chank_list2.append(chank_chank_list_2_2)

                    chank_list_copy = copy.deepcopy(list(chank_chank_list_2))
                    chank_list_copy2 = copy.deepcopy(list(chank_chank_list_2_2))
                    with lock:    
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=chank_size))
                        adjust_chank_list2.append(adjust_data_to_size(chank_list_copy2, target_size=chank_size))

                        analysis_flag.value = True
                    chank_chank_list_2 = []
                    chank_chank_list_2_2 = []
                    pretime = current_time
                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_1.append(receive_value[0])
                        chank_chank_list_1_2.append(receive_value2[0])

                        clock_signal.value = False

            elif flag_blink.value == False:
                if len(chank_chank_list_1) != 0:
                    current_time = time.time()
                    interval_time = current_time - pretime

                    with lock:
                        chank_list.append(chank_chank_list_1)
                        chank_list2.append(chank_chank_list_1_2)

                    chank_list_copy = copy.deepcopy(list(chank_chank_list_1))
                    chank_list_copy2 = copy.deepcopy(list(chank_chank_list_1_2))
                    with lock:
                        adjust_chank_list.append(adjust_data_to_size(chank_list_copy, target_size=chank_size))
                        adjust_chank_list2.append(adjust_data_to_size(chank_list_copy2, target_size=chank_size))

                        analysis_flag.value = True
                    chank_chank_list_1 = []
                    chank_chank_list_1_2 = []
                    # print("chank_list len: ", len(chank_list_copy), "interval_time: ", interval_time)  
                    pretime = current_time

                with lock:
                    if isinstance(receive_value, ListProxy) and len(receive_value) > 0 and clock_signal.value == True:
                        chank_chank_list_2.append(receive_value[0])
                        chank_chank_list_2_2.append(receive_value2[0])
                        clock_signal.value = False
