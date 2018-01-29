import numpy as np
import pandas as pd
import os
import mxnet as mx
from multiprocessing import Pool, Manager
from functools import partial
import random
import time

manager = Manager()
queue = manager.Queue()
STEP = 2000


def same_index(lst):
    s = []
    sub_s = []
    for index in lst:
        if len(sub_s) == 0:
            sub_s.append(index)
        elif index - sub_s[-1] == 1:
            sub_s.append(index)
        else:
            s.append(sub_s)
            sub_s = [index]

    return s


def same_day(lst):
    s = []
    sub_s = []
    for index in lst:
        if len(sub_s) == 0:
            sub_s.append(index)
        elif index[:3] == sub_s[-1][:3]:
            sub_s.append(index)
        else:
            s.append(sub_s)
            sub_s = [index]

    return s


def aft_cl_time(lst):
    index_lst = []
    for day in lst:
        if (day[0][0], day[0][1], day[0][2], 15, day[0][4], day[0][5]) in day:
            index_lst.append((day[0][0], day[0][1], day[0][2], 15, day[0][4], day[0][5]))
        elif (day[0][0], day[0][1], day[0][2], 15, day[0][4], day[0][5]) not in day and (
        day[0][0], day[0][1], day[0][2], 14, 59, day[0][5]) in day:
            index_lst.append((day[0][0], day[0][1], day[0][2], 14, 59, day[0][5]))
        elif (day[0][0], day[0][1], day[0][2], 15, day[0][4], day[0][5]) not in day and (
        day[0][0], day[0][1], day[0][2], 14, 59, day[0][5]) not in day and (
        day[0][0], day[0][1], day[0][2], 14, 58, day[0][5]) in day:
            index_lst.append((day[0][0], day[0][1], day[0][2], 14, 58, day[0][5]))
        elif (day[0][0], day[0][1], day[0][2], 15, day[0][4], day[0][5]) not in day and (
        day[0][0], day[0][1], day[0][2], 14, 59, day[0][5]) not in day and (
        day[0][0], day[0][1], day[0][2], 14, 58, day[0][5]) not in day and (
        day[0][0], day[0][1], day[0][2], 14, 57, day[0][5]) in day:
            index_lst.append((day[0][0], day[0][1], day[0][2], 14, 57, day[0][5]))
        else:
            continue
    return index_lst


def get_label_data(path):
    print(path)
    data = pd.read_csv(path, index_col=0, encoding='GB2312')
    data = data.dropna(axis=0, how='any')
    data = (data['total_turnover']) / (data['volume'])
    data = data.dropna(axis=0, how='any')
    data.index = pd.to_datetime(data.index)
    data_lst = []
    data_time = []
    for i in data.index:
        data_year, data_month, data_day, data_hour, data_minute, data_second = i.year, i.month, i.day, i.hour, i.minute, i.second
        data_time.append((data_year, data_month, data_day, data_hour, data_minute, data_second))
    for j in data:
        data_lst.append(j)
    total_day_lst = same_day(data_time)
    aft_time = aft_cl_time(total_day_lst)
    aft_time_index = []
    step = STEP
    for k in aft_time:
        aft_time_index.append(data_time.index(k))
    aft_time_index = list(filter(lambda x: x >= step, aft_time_index))
    return data_lst, aft_time_index


def extract_data(autio_path):
    if autio_path is None:
        queue.put((None, None))
    else:
        queue.put(get_label_data(autio_path))


def parallize_extract_data(audio_list):
    process_num = 12
    assert process_num < len(audio_list)
    pool = Pool(process_num)
    pool.map_async(partial(extract_data), audio_list)
    return pool


def data_record(input_path, train_idx_name, train_rec_name, test_idx_name, test_rec_name):
    if not os.path.isfile(input_path):
        raise FileNotFoundError("Please input right data_list")
    record_id = 0
    test_record_id = 0
    record = mx.recordio.MXIndexedRecordIO(train_idx_name, train_rec_name, 'w')
    test_record = mx.recordio.MXIndexedRecordIO(test_idx_name, test_rec_name, 'w')
    audio_list = open(input_path).readlines()
    audio_list = list(map(lambda x: x.strip(), audio_list))
    audio_list.append(None)
    pool = parallize_extract_data(audio_list)
    audio_num = 0
    is_end = False
    while True:
        if queue.qsize() > 0:
            data_lst, aft_time_index = queue.get()
            if data_lst is None:
                pool.close()
                pool.join()
                is_end = True
                continue
            audio_num += 1
            count = 0
            random_num = [random.randint(50, 3000) for _ in range(10)]
            aft_time_index = list(filter(lambda x: x >= STEP, aft_time_index))
            for index in range(len(aft_time_index) - 1):
                if index in random_num:
                    sparam = data_lst[aft_time_index[index] - STEP: aft_time_index[index]]
                    sparam = np.array(sparam, dtype=np.float32)
                    label = [data_lst[aft_time_index[index + 1]] / data_lst[aft_time_index[index]]]
                    if label[0] >= 1.8:
                        continue
                    elif 1 < label[0] < 1.8:
                        label[0] = 1
                    elif label[0] < 0.7:
                        continue
                    elif 0.7 <= label[0] < 1:
                        label[0] = 0
                    label = np.array(label, dtype=np.float32)

                    header = mx.recordio.IRHeader(0, label, 0, 0)
                    packed = mx.recordio.pack(header, sparam.tobytes())
                    test_record.write_idx(test_record_id, packed)
                    test_record_id += 1
                else:
                    sparam = data_lst[aft_time_index[index] - STEP: aft_time_index[index]]
                    sparam = np.array(sparam, dtype=np.float32)
                    label = [data_lst[aft_time_index[index + 1]] / data_lst[aft_time_index[index]]]
                    if label[0] >= 1.8:
                        continue
                    elif 1 < label[0] < 1.8:
                        label[0] = 1
                    elif label[0] < 0.7:
                        continue
                    elif 0.7 <= label[0] < 1:
                        label[0] = 0
                    label = np.array(label, dtype=np.float32)

                    header = mx.recordio.IRHeader(0, label, 0, 0)
                    packed = mx.recordio.pack(header, sparam.tobytes())
                    record.write_idx(record_id, packed)
                    record_id += 1
                count += 1
        elif queue.qsize() == 0 and is_end:
            break
    else:
        print('wait fft feature')
        time.sleep(0.1)
    record.close()
    print('write total record num: %s' % record_id)


if __name__ == '__main__':
    input_path = './d/project/mlp_data/smlp_data/experiment.lst'
    train_idx_name = './d/project/mlp_data/smlp_data/train_idx.txt'
    train_rec_name = './d/project/mlp_data/smlp_data/train_rec_data.rec'
    test_idx_name = './d/project/mlp_data/smlp_data/test_idx.txt'
    test_rec_name = './d/project/mlp_data/smlp_data/test_rec_data.rec'
    data_record(input_path=input_path, train_idx_name=train_idx_name, train_rec_name=train_rec_name, test_idx_name=test_idx_name,
                test_rec_name=test_rec_name)
