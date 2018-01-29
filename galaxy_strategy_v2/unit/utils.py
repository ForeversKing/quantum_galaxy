import mxnet as mx
import logging
import time
import requests
import random
import numpy as np

RATE = 1
LOSS = 0
ID = 34


##########
class PolynomialScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, epochs, batch_size, examples, start_lr, end_lr, pow, frequent=50):
        super(PolynomialScheduler, self).__init__()
        self._global_step = int(examples / batch_size * epochs)
        self._lr = start_lr
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._pow = pow
        self._last_num = 0
        self._frequent = frequent
        # self._current_step = 0

    def __call__(self, num_update):
        if num_update > self._global_step:
            return self._end_lr
        self._lr = (self._start_lr - self._end_lr) * (1 - num_update / self._global_step) ** self._pow + self._end_lr
        if num_update % self._frequent == 0 and num_update != self._last_num:
            logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                         "change in the future", num_update, self._lr)
            self._last_num = num_update
        return self._lr


####################


class FermiScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, epochs, num_data, batch_size, start_lr, end_lr, frequent=50):
        super(FermiScheduler, self).__init__()
        self.examples = epochs * num_data
        self._global_step = int(self.examples / batch_size)
        self.update_f = int((5) * num_data / batch_size)
        self.delta = self.update_f / 7
        self._lr = start_lr
        self._end_lr = end_lr
        self._start_lr = start_lr
        self.correction_param_1 = (8) * num_data
        self._last_num = 0
        self._frequent = frequent
        # self._current_step = 0

    def __call__(self, num_update):
        if num_update > self._global_step:
            return self._end_lr
        self._lr = self._start_lr / (1 + np.exp((num_update - self.update_f) / self.delta)) + (
                self._end_lr / self.correction_param_1) * (self.correction_param_1 - num_update)
        if num_update % self._frequent == 0 and num_update != self._last_num:
            logging.info("Update[%d]: now learning rate arrived at %0.5e", num_update, self._lr)
            self._last_num = num_update
        return self._lr


#####################


class RandomFactorScheduler(mx.lr_scheduler.LRScheduler):
    def __init__(self, host_name, max_lr, min_lr, frequent=50):
        super(RandomFactorScheduler, self).__init__()
        self._max_lr = max_lr
        self._min_lr = min_lr
        self._last_num = 0
        self._frequent = frequent
        self.host_name = host_name
        # self._current_step = 0

    def __call__(self, num_update):
        self._lr = self._min_lr + (self._max_lr - self._min_lr) * random.random()
        if num_update % self._frequent == 0 and num_update != self._last_num:
            logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                         "change in the future", num_update, self._lr)
            # send_learn_info(self.host_name, self._lr)
            self._last_num = num_update
        return self._lr


##################################
class WYLMultiFactorScheduler(mx.lr_scheduler.LRScheduler):
    """Reduce the learning rate by given a list of steps.

    Assume there exists *k* such that::

       step[k] <= num_update and num_update < step[k+1]

    Then calculate the new learning rate by::

       base_lr * pow(factor, k+1)

    Parameters
    ----------
    step: list of int
        The list of steps to schedule a change
    factor: float
        The factor to change the learning rate.
    """

    def __init__(self, step, factor=1, host_name=None):
        super(WYLMultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i - 1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.host_name = host_name
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while self.cur_step_ind <= len(self.step) - 1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
                # requests.post(self.host_name, data={'learn_rate': self.base_lr, 'value_type': RATE,
                # 'model_id': ID})
            else:
                return self.base_lr
        return self.base_lr


#############################

class Speedometer(object):
    def __init__(self, batch_size, frequent=50, host_name=None, id_to_time=0, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.host_name = host_name
        self.time_id = id_to_time
        self.last_count = 0
        self.auto_reset = auto_reset

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                    msg = 'Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec'
                    msg += '\t%s=%f' * len(name_value)
                    logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                    # send_loss_info(self.host_name, name_value)
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


######################
def send_loss_info(host_name, name_value):
    timeout = 5
    now = time.time()
    dic = dict(name_value)
    while time.time() - now < timeout:
        try:

            requests.post(host_name, data={'loss': dic['cross-entropy'],
                                           'accuracy': dic['accuracy'],
                                           'value_type': LOSS, 'model_id': ID})
            break
        except:
            print('connect error')
            time.sleep(1)


def send_learn_info(host_name, learn_rate):
    timeout = 5
    now = time.time()
    while time.time() - now < timeout:
        try:
            # requests.post(host_name, data={'learn_rate': learn_rate, 'value_type': RATE,
            #                                     'model_id': ID})
            pass
        except:
            print('connect error')
            time.sleep(1)
