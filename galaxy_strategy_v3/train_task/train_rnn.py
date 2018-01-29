import mxnet as mx
import logging
from galaxy_strategy_v3.units.bucket_rnn import BucketSentenceIter
from galaxy_strategy_v3.units.metric import accuracy, CrossEntropy


class Param(object):
    def __init__(self):
        self.batch_size = 32
        self.record_len = 930
        self.gap = 200
        self.sclice_n = 20
        self.feature_len = 512 * self.sclice_n
        self.seq_range = (150, 200)
        self.learn_rate = 0.0005
        self.end_lr = 0.0000001
        self.pow = 0.6
        self.epoch = 100
        self.wd = 0.00005
        self.moment = 0.9
        self.num_layers = 2
        self.num_hidden = 256
        self.gpu = 1
        self.frequent = 20
        self.global_step = int(self.record_len / self.batch_size * self.epoch)
        self.save_model_name = '/home/yulongwu/d/data/voice_data/voice_label/tmp/model_name'
        self.idx_file = '/home/yulongwu/d/data/voice_data/voice_label/tmp/tmp_data.idx'
        self.rec_file = '/home/yulongwu/d/data/voice_data/voice_label/tmp/tmp_data.rec'


if __name__ == '__main__':
    param = Param()

    feature = []
    record = mx.recordio.MXIndexedRecordIO(param.idx_file, param.rec_file, 'r')
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    data_train = BucketSentenceIter(idx_file=param.idx_file, rec_file=param.rec_file, gap=param.gap,
                                    record_len=param.record_len, feature_len=param.feature_len,
                                    seq_len_range=param.seq_range,
                                    batch_size=param.batch_size)
    stack = mx.rnn.SequentialRNNCell()
    for i in range(param.num_layers):
        stack.add(mx.rnn.BidirectionalCell(mx.rnn.LSTMCell(num_hidden=param.num_hidden, prefix='lstm_l%d_' % i)))

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=data, merge_outputs=False)
        print('seq-len:', seq_len)
        pred = mx.sym.Reshape(outputs[-1], shape=(-1, param.num_hidden))
        fc1 = mx.sym.FullyConnected(data=pred, num_hidden=256, name='fc1')
        fc1_bn = mx.sym.BatchNorm(data=fc1, fix_gamma=False, eps=2e-5, momentum=0.9, name="fc1_bn")
        fc1_act = mx.sym.Activation(data=fc1_bn, act_type='relu', name='fc1_act')

        fc21 = mx.sym.FullyConnected(data=fc1_act, num_hidden=2)
        fc22 = mx.sym.FullyConnected(data=fc1_act, num_hidden=3)
        fc23 = mx.sym.FullyConnected(data=fc1_act, num_hidden=2)

        fc2 = mx.sym.Concat(*[fc21, fc22, fc23], dim=0)

        label = mx.sym.Reshape(data=label, target_shape=(0, ))
        pred = mx.sym.SoftmaxOutput(data=fc2, label=label, name='softmax')
        return pred, ('data',), ('softmax_label', )


    contexts = [mx.gpu(int(i)) for i in range(param.gpu)]
    model = mx.mod.BucketingModule(sym_gen=sym_gen,
                                   default_bucket_key=data_train.default_bucket_key,
                                   context=contexts)
    sgd_opt = mx.optimizer.SGD(learning_rate=param.learn_rate, momentum=param.moment, wd=param.wd,
                               rescale_grad=1 / param.batch_size)
    # lr_pol = PolynomialScheduler(epochs=param.epoch, batch_size=param.batch_size,
    #                              examples=param.record_len, start_lr=param.learn_rate, end_lr=param.end_lr,
    #                              pow=param.pow, frequent=param.frequent)


    def lr_callback(cp):
        global_nbatch = cp.epoch * int(param.record_len / param.batch_size) + cp.nbatch
        base_lr = param.learn_rate
        if cp.epoch >= param.epoch / 4:
            base_lr = param.learn_rate / 10
        if cp.epoch >= param.epoch / 1.5:
            base_lr = param.learn_rate / 50
        if global_nbatch < param.global_step:
            sgd_opt.lr = (1 - global_nbatch / param.global_step) ** param.pow * (base_lr - param.end_lr) + param.end_lr
        else:
            sgd_opt.lr = param.end_lr
        if cp.nbatch % 20 == 0:
            logging.info('Epoch[%d] Batch [%d]      learning rate:%f' % (cp.epoch, cp.nbatch, sgd_opt.lr))


    model.fit(
        train_data=data_train,
        eval_metric=[accuracy, CrossEntropy()],
        optimizer=sgd_opt,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        epoch_end_callback=mx.rnn.do_rnn_checkpoint(stack, param.save_model_name, 40),
        num_epoch=param.epoch,
        batch_end_callback=[mx.callback.Speedometer(param.batch_size, param.frequent), lr_callback])
