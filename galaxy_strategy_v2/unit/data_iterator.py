import mxnet as mx
import numpy as np


def get_train_data_iter(data_path, data_shape, batch_size, kv_store):
    if kv_store:
        rank, nworker = kv_store.rank, kv_store.num_workers
    else:
        rank, nworker = 0, 1
    return mx.io.ImageRecordIter(path_imgrec=data_path,
                                 data_shape=data_shape,
                                 batch_size=batch_size,
                                 label_width=1,
                                 scale=1.0 / 255,
                                 rand_mirror=True,
                                 rand_crop=False,
                                 shuffle=True,
                                 num_parts=nworker,
                                 part_index=rank,
                                 data_name='data',
                                 label_name='softmax_label')


def get_infer_data_iter(data_path, data_shape, batch_size):
    return mx.io.ImageRecordIter(path_imgrec=data_path,
                                 data_shape=data_shape,
                                 batch_size=batch_size,
                                 scale=1.0 / 255,
                                 rand_mirror=False,
                                 rand_crop=False,
                                 # shuffle=True,
                                 data_name='data',
                                 label_name='softmax_label')


def get_infer_image_iter(data_path, image_root, data_shape, batch_size):
    return mx.image.ImageIter(batch_size=batch_size,
                              data_shape=data_shape,
                              path_imglist=data_path,
                              path_root=image_root,
                              mean=np.array([0, 0, 0]),
                              std=np.array([255, 255, 255]))
