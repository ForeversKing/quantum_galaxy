import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
from multiprocessing import Pool
from functools import partial


STEP = 2000
FFT_SIZE = 400  # or must be 2^n
STEP_SIZE = 16
THRESH = 5


def overlap(X, window_size, window_step):
    assert window_size % 2 == 0, "Window size must be even!"
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    ws = int(window_size)
    ss = int(window_step)
    valid = len(X) - ws
    nw = valid // ss
    out = np.ndarray((nw, 512), dtype=X.dtype)
    for i in range(nw):
        start = i * ss
        stop = start + ws
        tmp = X[start: stop]
        tmp = np.hamming(ws) * tmp
        sig512 = np.hstack((tmp, np.zeros(512 - len(tmp))))
        out[i] = np.fft.fft(sig512)
    return out[:, :256]


def stft(X, fftsize=128, step_size=16, mean_normalize=True):
    if mean_normalize:
        X = (X - X.mean()) / X.std()
    X = overlap(X, fftsize, step_size)
    return X


def pretty_spectrogram(audio_data, log=True, thresh=5,
                       fft_size=512, step_size=64):
    specgram = np.abs(stft(audio_data, fftsize=fft_size, step_size=step_size))
    if log:
        specgram /= specgram.max()
        specgram = np.log10(specgram)
        specgram[specgram < -thresh] = -thresh
        specgram += thresh
    else:
        specgram[specgram < thresh] = thresh
    return specgram


def voice2image(data, out_dir, label, fft_size=80, step_size=20, thresh=5, color=True):
    fft_size = fft_size
    step_size = step_size
    thresh = thresh
    wav_spectrogram = pretty_spectrogram(data.astype('float64'),
                                         fft_size=fft_size,
                                         step_size=step_size,
                                         log=True,
                                         thresh=thresh)
    wav_spectrogram = wav_spectrogram / np.max(wav_spectrogram) * 255.0
    print('wav_spectrogram: ', wav_spectrogram.shape)
    slice_spectrogram = wav_spectrogram
    spectrogram = slice_spectrogram.astype(np.uint8).T
    spectrogram = np.flip(spectrogram, 0)
    if color:
        new_im = cv2.applyColorMap(spectrogram, cv2.COLORMAP_JET)
        cv2.imwrite('%s/%s.png' % (out_dir, label), new_im)
    else:
        new_im = Image.fromarray(spectrogram)
        new_im.save(out_dir)


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


def get_data(path, out_dir):
    print(path)
    name_class = os.path.basename(path).split('.')
    name_class = '%s_%s' % (name_class[0], name_class[1])
    out_path_dir = os.path.join(out_dir, name_class)
    if not os.path.exists(out_path_dir):
        os.mkdir(out_path_dir)
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
    for k in aft_time:
        aft_time_index.append(data_time.index(k))
    aft_time_index = list(filter(lambda x: x >= STEP, aft_time_index))
    for index in range(len(aft_time_index) - 1):
        sparam = data_lst[aft_time_index[index] - STEP: aft_time_index[index]]
        sparam = np.array(sparam)
        label = data_lst[aft_time_index[index + 1]] / data_lst[aft_time_index[index]]
        if label > 1:
            label = 1
        else:
            label = 0

        name_time = data_time[aft_time_index[index]]
        name_time = '%s-%s-%s %s:%s:%s_%s' % (name_time[0], name_time[1], name_time[2], name_time[3], name_time[4],
                                              name_time[5], label)
        voice2image(data=sparam, out_dir=out_path_dir, label=name_time, fft_size=FFT_SIZE, step_size=STEP_SIZE, thresh=THRESH)


if __name__ == '__main__':
    input_path = './d/project/mlp_data/smlp_data/experiment.lst'
    out_dir = './d/project/image_data/sdata_image'
    total_lst = []
    with open(input_path, 'r') as data_input:
        for i in data_input:
            total_lst.append(i.strip())
    pool = Pool(12)
    rl = pool.map(partial(get_data, out_dir=out_dir), total_lst)
    pool.close()
    pool.join()

