import random
import time
import numpy as np
import mxnet as mx
from mxnet.io import DataIter, DataBatch, DataDesc


class BucketSentenceIter(DataIter):
    def __init__(self, idx_file=None, rec_file=None,
                 record_len=None, feature_len=None,
                 seq_len_range=3360,
                 batch_size=None,
                 data_name='data', label_name='softmax_label'):
        super(BucketSentenceIter, self).__init__()
        self.record_len = record_len
        self.read_num = 0
        self.idx_range = [i for i in range(self.record_len)]
        self.batch_size = batch_size
        self.record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = 'float32'
        self.layout = 'NT'
        self.seq_len_range = seq_len_range
        self.default_bucket_key = seq_len_range

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
        print('*********')
        now = time.time()
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
                data = np.frombuffer(data, np.float32)
                self.cache[idx] = (data, label)
            feature.append(data)
            label_list.append(label)
        print('read batch cost: ', time.time() - now)
        return np.array(feature), np.array(label_list)

    def next(self):
        data, label= self._read_batch()
        data = mx.nd.array(data)
        label = mx.nd.array(label)
        return DataBatch([data], [label], pad=0,
                         bucket_key=self.seq_len_range,
                         provide_data=[DataDesc(
                             name=self.data_name, shape=data.shape,
                             layout=self.layout)],
                         provide_label=[DataDesc(
                             name=self.label_name, shape=label.shape,
                             layout=self.layout)])
