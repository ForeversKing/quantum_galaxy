import random
import time
import numpy as np
import mxnet as mx
from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet import ndarray


class BucketSentenceIter(DataIter):
    def __init__(self, idx_file=None, rec_file=None,
                 record_len=None, feature_len=None,
                 seq_len_range=(150, 200),
                 batch_size=None, gap=200,
                 data_name='data', label_name='softmax_label'):
        super(BucketSentenceIter, self).__init__()
        self.record_len = record_len
        self.gap = gap
        self.read_num = 0
        self.idx_range = [i for i in range(self.record_len)]
        self.batch_size = batch_size
        self.record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = 'float64'
        self.layout = 'NT'
        self.seq_len_range = list(range(*seq_len_range))
        self.default_bucket_key = max(seq_len_range)

        self.provide_data = [DataDesc(
            name=self.data_name, shape=(batch_size, self.default_bucket_key, feature_len),
            layout=self.layout)]
        self.provide_label = [DataDesc(
            name=self.label_name, shape=(batch_size, self.default_bucket_key),
            layout=self.layout)]
        self.cache = {}
        self.reset()

    def reset(self):
        random.shuffle(self.idx_range)
        self.read_num += 1

    def _read_batch(self):
        thresh = 5
        now = time.time()
        seq_len = random.choice(self.seq_len_range)
        feature = []
        label_list = []
        start = self.read_num % (self.record_len - self.batch_size)
        for idx in self.idx_range[start: start + self.batch_size]:
            self.read_num += 1
            if self.read_num % (self.record_len - self.batch_size) == 0:
                raise StopIteration
            if idx in self.cache:
                data, label = self.cache[idx]
            else:
                item = self.record.read_idx(idx)
                header, data = mx.recordio.unpack(item)
                label = header.label
                data = np.frombuffer(data, np.float64)
                data = np.reshape(data, (self.gap, int(data.shape[0] / self.gap)))
                label = np.array(label)
                data = data + 10e-8
                data /= data.max()
                data = np.log10(data)
                data[data < -thresh] = -thresh  # set anything less than the threshold as the threshold
                data += thresh
                data /= thresh
                self.cache[idx] = (data, label)
            start_idx = random.randint(0, self.gap - seq_len)
            data = data[start_idx: start_idx + seq_len]
            label = label[start_idx: start_idx + seq_len]
            feature.append(data)
            label_list.append(label)
        print('read batch cost: ', time.time() - now)
        return np.array(feature), np.array(label_list), seq_len

    def next(self):
        data, label, seq_len = self._read_batch()
        data = mx.nd.array(data)
        label = mx.nd.array(label)
        return DataBatch([data], [label], pad=0,
                         bucket_key=seq_len,
                         provide_data=[DataDesc(
                             name=self.data_name, shape=data.shape,
                             layout=self.layout)],
                         provide_label=[DataDesc(
                             name=self.label_name, shape=label.shape,
                             layout=self.layout)])
