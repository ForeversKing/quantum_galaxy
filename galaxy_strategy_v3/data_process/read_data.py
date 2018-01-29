import numpy as np
import pandas as pd
import os
import mxnet as mx
from multiprocessing import Pool, Manager
from functools import partial
from galaxy_strategy_v3.data_process.config import *
import random
import time

manager = Manager()
queue = manager.Queue()
STEP = 28 * 60 * 2
sample_data_step = 30
sample_label_step = 1


def get_label_data(path):
    data = pd.read_csv(path, index_col=0, encoding='GB2312')
    data = data.dropna(axis=0, how='any')
    op = []
    for k in data[O]:
        if k != 0:
            op.append(k)
            break
        else:
            continue
    data[A] = data[A] / (data[B] * 10)
    data = data[[T, A, B, C, D]]
    data = data.dropna(axis=0, how='any')
    data.index = pd.to_datetime(data[T], format="%Y-%m-%d %H:%M:%S")
    data = data.drop([T], axis=1)
    # data[A] = (data[A] - op[0]) / op[0]
    data[B] = (data[B] - data.loc[data.index[0], B]) / data.loc[data.index[0], B]
    for k in range(len(data[C])):
        if data.loc[data.index[k], C] == kk:
            data.loc[data.index[k], C] = 0
        elif data.loc[data.index[k], C] == kp:
            data.loc[data.index[k], C] = 1
        elif data.loc[data.index[k], C] == dk:
            data.loc[data.index[k], C] = 2
        else:
            data.loc[data.index[k], C] = 3
    for k in range(len(data[D])):
        if data.loc[data.index[k], D] == 'S':
            data.loc[data.index[k], D] = 0
        else:
            data.loc[data.index[k], D] = 1
    assert len(data[A]) == len(data[B]) == len(data[C]) == len(data[D]), 'There is wrong here.'

    data = np.concatenate((np.array(data[A]).T, np.array(data[B]).T, np.array(data[C]).T, np.array(data[D]).T), axis=0)
    data = data.reshape(-1, int(data.shape[0] / 4))
    return op, data


def extract_data(autio_path):
    if autio_path is None:
        queue.put((None, None))
    else:
        queue.put(get_label_data(autio_path))


def parallize_extract_data(audio_list):
    process_num = 1
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
            op, data = queue.get()
            print('&&&&&&&', data.shape)
            if data is None:
                pool.close()
                pool.join()
                is_end = True
                continue
            audio_num += 1
            count = 0
            random_num = [random.randint(STEP, data.shape[1]) for _ in range(50)]
            assert sample_data_step > sample_label_step
            for index in range(STEP, data.shape[1] - 1, sample_data_step):
                if index in random_num:
                    label = data[0][index + sample_label_step] / data[0][index]
                    if label > 1:
                        label = 1
                    else:
                        label = 0
                    label = np.array(label, dtype=np.float32)

                    sparam = data[:, index - STEP:index]
                    sparam[0, :] = (sparam[0, :] - op[0]) / op[0]
                    sparam = np.array(sparam, dtype=np.float32)

                    header = mx.recordio.IRHeader(0, label, 0, 0)
                    packed = mx.recordio.pack(header, sparam.tobytes())
                    test_record.write_idx(test_record_id, packed)
                    test_record_id += 1
                else:
                    label = data[0][index + sample_label_step] / data[0][index]
                    if label > 1:
                        label = 1
                    else:
                        label = 0
                    label = np.array(label, dtype=np.float32)

                    sparam = data[:, index - STEP:index]
                    sparam[0, :] = (sparam[0, :] - op[0]) / op[0]
                    sparam = np.array(sparam, dtype=np.float32)

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
    input_path = './d/project/rnn_data/total.lst'
    train_idx_name = './d/project/rnn_data/train_idx.txt'
    train_rec_name = './d/project/rnn_data/train_rec_data.rec'
    test_idx_name = './d/project/rnn_data/test_idx.txt'
    test_rec_name = './d/project/rnn_data/test_rec_data.rec'
    data_record(input_path=input_path, train_idx_name=train_idx_name, train_rec_name=train_rec_name, test_idx_name=test_idx_name,
                test_rec_name=test_rec_name)
