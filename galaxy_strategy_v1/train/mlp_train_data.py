import mxnet as mx
import numpy as np
import logging
import random


def get_symbol(num_class):
    data = mx.sym.Variable('data')
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=512, name='fc1')
    bn1 = mx.sym.BatchNorm(data=fc1, fix_gamma=False, eps=2e-5, momentum=0.99, name='bn1')
    ac1 = mx.sym.Activation(data=bn1, act_type='relu', name='ac1')

    fc2 = mx.sym.FullyConnected(data=ac1, num_hidden=1024, name='fc2')
    bn2 = mx.sym.BatchNorm(data=fc2, fix_gamma=False, eps=2e-5, momentum=0.99, name='bn2')
    ac2 = mx.sym.Activation(data=bn2, act_type='relu', name='ac2')

    fc3 = mx.sym.FullyConnected(data=ac2, num_hidden=2048, name='fc3')
    bn3 = mx.sym.BatchNorm(data=fc3, fix_gamma=False, eps=2e-5, momentum=0.99, name='bn3')
    ac3 = mx.sym.Activation(data=bn3, act_type='relu', name='ac3')

    fc4 = mx.sym.FullyConnected(data=ac3, num_hidden=1024, name='fc4')
    bn4 = mx.sym.BatchNorm(data=fc4, fix_gamma=False, eps=2e-5, momentum=0.99, name='bn4')
    ac4 = mx.sym.Activation(data=bn4, act_type='relu', name='ac4')

    fc5 = mx.sym.FullyConnected(data=ac4, num_hidden=num_class, name='fc5')

    return mx.sym.SoftmaxOutput(data=fc5, name='softmax')


class Param(object):
    def __init__(self):
        self.batch_size = 16
        self.record_len = 721
        self.learn_rate = 0.01
        self.seq_len = 3360
        self.end_lr = 0.00001
        self.pow = 0.6
        self.epoch = 100
        self.wd = 0.00005
        self.moment = 0.9
        self.gpu = 2
        self.frequent = 40
        self.global_step = int(self.record_len / self.batch_size * self.epoch)
        self.save_model_name = './d/project/mlp_data/model/model_name'


if __name__ == '__main__':
    param = Param()
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    idx_file = '/home/yulongwu/d/project/rnn_data/train_idx.txt'
    rec_file = '/home/yulongwu/d/project/rnn_data/train_rec_data.rec'
    batch_size = param.batch_size
    idx_range = param.record_len
    label_list = []
    feature_list = []
    total_lst = []
    seq_len = param.seq_len
    record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')
    for idx in range(idx_range):
        item = record.read_idx(idx)
        header, data = mx.recordio.unpack(item)
        label = np.array(header.label)
        data = np.frombuffer(data, np.float32)
        data = (data - data.mean()) / data.std()
        # data = data[-seq_len:]
        label_add_data = np.concatenate((label, data), axis=0)
        total_lst.append(label_add_data)
        # feature_list.append(data[:seq_len])
        # label_list.append(label)
    random.shuffle(total_lst)
    for i in total_lst:
        feature_list.append(i[1:])
        label_list.append(i[0])
    feature = np.array(feature_list, dtype=np.float32)
    label = np.array(label_list, dtype=np.float32)
    print('read data is over.')
    train_iter = mx.io.NDArrayIter(data=feature, label=label, batch_size=batch_size,
                                   shuffle=True, data_name='data', label_name='softmax_label')
    net = get_symbol(2)
    model = mx.mod.Module(symbol=net, context=[mx.gpu(i) for i in range(2)])

    sgd_opt = mx.optimizer.SGD(learning_rate=param.learn_rate, momentum=param.moment, wd=param.wd,
                               rescale_grad=1 / param.batch_size)

    def lr_callback(cp):
        global_nbatch = cp.epoch * int(param.record_len / param.batch_size) + cp.nbatch
        base_lr = param.learn_rate
        if cp.epoch >= param.epoch / 4:
            base_lr = param.learn_rate / 10
        if cp.epoch >= param.epoch / 1.5:
            base_lr = param.learn_rate / 20
        if global_nbatch < param.global_step:
            sgd_opt.lr = (1 - global_nbatch / param.global_step) ** param.pow * (base_lr - param.end_lr) + param.end_lr
        else:
            sgd_opt.lr = param.end_lr
        if cp.nbatch % 20 == 0:
            logging.info('Epoch[%d] Batch [%d]      learning rate:%f' % (cp.epoch, cp.nbatch, sgd_opt.lr))

    model.fit(train_data=train_iter,
              optimizer=sgd_opt,
              eval_metric=['acc', 'ce'],
              epoch_end_callback=mx.callback.do_checkpoint(param.save_model_name, 5),
              batch_end_callback=[mx.callback.Speedometer(param.batch_size, param.frequent), lr_callback],
              num_epoch=param.epoch)
