from __future__ import print_function

import bisect
import random
import numpy as np

from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet import ndarray


class BucketSentenceIter(DataIter):
    def __init__(self, sentences, batch_size, buckets=None, invalid_label=-1,
                 data_name='data', label_name='softmax_label', dtype='float32',
                 layout='NT'):
        super(BucketSentenceIter, self).__init__()
        ndiscard = 0
        self.data = [[] for _ in buckets]
        for i, sent in enumerate(sentences):
            buck = bisect.bisect_left(buckets, len(sent))
            if buck == len(buckets):
                ndiscard += 1
                continue
            buff = np.full((buckets[buck],), invalid_label, dtype=dtype)
            buff[:len(sent)] = sent
            self.data[buck].append(buff)

        self.data = [np.asarray(i, dtype=dtype) for i in self.data]
        self.batch_size = batch_size
        self.buckets = buckets
        self.data_name = data_name
        self.label_name = label_name
        self.dtype = dtype
        self.invalid_label = invalid_label
        self.nddata = []
        self.ndlabel = []
        self.major_axis = layout.find('N')
        self.layout = layout
        self.default_bucket_key = max(buckets)
        self.provide_data = [DataDesc(
            name=self.data_name, shape=(batch_size, self.default_bucket_key),
            layout=self.layout)]
        self.provide_label = [DataDesc(
            name=self.label_name, shape=(batch_size, self.default_bucket_key),
            layout=self.layout)]
        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0
        self.reset()

    def reset(self):
        self.curr_idx = 0
        random.shuffle(self.idx)
        for buck in self.data:
            np.random.shuffle(buck)

        self.nddata = []
        self.ndlabel = []
        for buck in self.data:
            label = np.empty_like(buck)
            label[:, :-1] = buck[:, 1:]
            label[:, -1] = self.invalid_label
            self.nddata.append(ndarray.array(buck, dtype=self.dtype))
            self.ndlabel.append(ndarray.array(label, dtype=self.dtype))

    def next(self):
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1
        data = self.nddata[i][j:j + self.batch_size]
        label = self.ndlabel[i][j:j + self.batch_size]
        return DataBatch([data], [label], pad=0,
                         bucket_key=self.buckets[i],
                         provide_data=[DataDesc(
                             name=self.data_name, shape=data.shape,
                             layout=self.layout)],
                         provide_label=[DataDesc(
                             name=self.label_name, shape=label.shape,
                             layout=self.layout)])
