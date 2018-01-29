import mxnet as mx
import logging
from galaxy_strategy_v2.network.vgg_aft_256_125 import get_symbol
from galaxy_strategy_v2.unit.data_iterator import get_train_data_iter
from galaxy_strategy_v2.unit.utils import Speedometer
import os, time
import sys
import requests

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(filename)s %(message)s',
#                     datefmt='%d %b %Y %H:%M:%S',
#                     filename='./d/voice/voice_image/voice_model/train_vgg/vgg.log',
#                     filemode='a+')


class Param:
    def __init__(self):
        self.model_name = 'train_vgg'
        self.batch_size = 98
        self.num_classes = 2
        self.examples = 345702
        self.imges_shape = (3, 256, 125)
        self.gpu_num = 2
        self.epoch = 50
        self.wd = 0.0002
        self.learning_rate = 0.007
        self.momentum = 0.9
        self.pow = 0.7
        self.begin_epoch = 15
        self.end_lr = 0.0001
        self.kv_store = mx.kv.create('device')
        self.learn_host = 'http://ip:5010/send_learn'
        self.loss_host = 'http://ip:5010/send_loss'
        self.model_save_root = ".d/project/image_data/sdata_image"
        self.data_name = "./d/project/image_data/sdata_image/train_shuffle_rec.rec"
        path = os.path.join(self.model_save_root, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_model_name = path + '/' + 'vgg_model'
        # self.lr_scheduler = RandomFactorScheduler(max_lr=0.0005, min_lr=0.00005, host_name=self.learn_host)
        # self.lr_scheduler = FermiScheduler(self.epoch, num_data=self.examples, batch_size=self.batch_size,
        #                                   start_lr=self.learning_rate, end_lr=0.0022)
        self.global_step = int(self.examples / self.batch_size * self.epoch)
        self.net = get_symbol(self.num_classes)
        self.data_iter = get_train_data_iter(self.data_name, self.imges_shape, self.batch_size, self.kv_store)
        if self.begin_epoch > 0:
            _, self.pre_arg_params, self.pre_aux_params = mx.model.load_checkpoint(self.save_model_name,
                                                                                   self.begin_epoch)
        else:
            self.pre_arg_params, self.pre_aux_params = None, None


param = Param()
checkpoint = mx.callback.do_checkpoint(param.save_model_name, 5)

# from wyl.units import memonger
# test_bs = int(param.batch_size / param.gpu_num)
# data_shape = {'data': (test_bs, 3, param.imges_shape[1], param.imges_shape[1]),
#               'softmax_label': (test_bs, )}
# net_planned = memonger.search_plan(param.net, **data_shape)

lenet_model = mx.mod.Module(symbol=param.net,
                            context=[mx.gpu(i) for i in range(param.gpu_num)],
                            data_names=['data'],
                            label_names=['softmax_label'])

sgd_opt = mx.optimizer.SGD(learning_rate=param.learning_rate,
                           momentum=param.momentum,
                           wd=param.wd,
                           rescale_grad=1 / param.batch_size)


def lr_callback(cp):
    global_nbatch = cp.epoch * int(param.examples / param.batch_size) + cp.nbatch
    base_lr = param.learning_rate
    if cp.epoch >= param.epoch / 2:
        base_lr = param.learning_rate / 5
    if global_nbatch < param.global_step:
        sgd_opt.lr = (1 - global_nbatch / param.global_step) ** param.pow * (base_lr - param.end_lr) + param.end_lr
    else:
        sgd_opt.lr = param.end_lr
    if cp.nbatch % 20 == 0:
        logging.info('Epoch[%d] Batch [%d]      learning rate:%f' % (cp.epoch, cp.nbatch, sgd_opt.lr))


lenet_model.fit(param.data_iter,
                optimizer=sgd_opt,
                initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2),
                eval_metric=['acc', 'ce'],
                arg_params=param.pre_arg_params,
                aux_params=param.pre_aux_params,
                # allow_missing=True,
                begin_epoch=param.begin_epoch,
                epoch_end_callback=checkpoint,
                # kvstore=param.kv_store,
                batch_end_callback=[Speedometer(batch_size=param.batch_size, frequent=20, host_name=param.loss_host),
                                    lr_callback],
                num_epoch=param.epoch)
