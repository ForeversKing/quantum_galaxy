import mxnet as mx
import numpy as np


class Accuracy(mx.metric.EvalMetric):
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None):
        super(Accuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis

    def update(self, labels, preds):
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.nd.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int64')
            label = label.asnumpy().astype('int64')
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)


class CrossEntropy(mx.metric.EvalMetric):
    def __init__(self, eps=1e-12, name='ce',
                 output_names=None, label_names=None):
        super(CrossEntropy, self).__init__(
            name, eps=eps,
            output_names=output_names, label_names=label_names)
        self.eps = eps

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()
            label = label.ravel()
            assert label.shape[0] == pred.shape[0]
            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]


def accuracy(label, pred):
    print('%%%%%', label.shape)
    label = label.T.reshape((-1,))
    hit = 0
    total = 0
    for i in range(int(pred.shape[0] / 4)):
        ok = True
        for j in range(4):
            k = i * 4 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total