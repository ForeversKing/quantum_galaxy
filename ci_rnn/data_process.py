import sqlite3
import os, sys
import mxnet as mx
import logging
import json
from ci_rnn.bucket_ci import BucketSentenceIter
from ci_rnn.metric import Accuracy, CrossEntropy

start_token = 'B'
end_token = 'E'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def encode_sentences(sentences, vocab=None, invalid_label=-1, invalid_key='\n', start_label=0):
    idx = start_label
    if vocab is None:
        vocab = {invalid_key: invalid_label}
        new_vocab = True
    else:
        new_vocab = False
    res = []
    for sent in sentences:
        coded = []
        for word in sent:
            if word not in vocab:
                assert new_vocab, "Unknown token %s" % word
                if idx == invalid_label:
                    idx += 1
                vocab[word] = idx
                idx += 1
            coded.append(vocab[word])
        res.append(coded)

    return res, vocab


def tokenize_text(fname, vocab=None, invalid_label=-1, start_label=0):
    if not os.path.isfile(fname):
        raise IOError("Please use get_ptb_data.sh to download requied file ")
    with open(fname, "r", encoding='utf-8') as fi:
        lines = []
        for row in fi:
            row = row.strip().replace(' ', '')
            if '_' in row or '(' in row or '（' in row or '《' in row or '[' in row or \
                    start_token in row or end_token in row:
                continue
            if len(row) < 5 or len(row) > 90:
                continue
            row = start_token + row + end_token
            lines.append(filter(None, row))
    sentences, vocab = encode_sentences(lines, vocab=vocab, invalid_label=invalid_label,
                                        start_label=start_label)
    return sentences, vocab


def from_database_data(database_path, save_path):
    fo = open(save_path, 'w')
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        name = []
        cursor.execute("SELECT * FROM ci limit 21050;")
        df = cursor.fetchall()
        for k in range(21050):
            # if df[k][2] == '柳永':
            #     name.append(df[k])
            # elif df[k][2] == '辛弃疾':
            #     name.append(df[k])
            # elif df[k][2] == '苏轼':
            name.append(df[k])
    for row in name:
        ci_consent = row[3].split('\n')
        fo.write(''.join(ci_consent) + '\n')
    fo.close()


class Param(object):
    def __init__(self):
        self.batch_size = 64
        self.learn_rate = 0.01
        self.end_lr = 0.0000001
        self.example = 43030
        self.pow = 0.6
        self.epoch = 200
        self.wd = 0.00005
        self.moment = 0.9
        self.num_layers = 2
        self.num_hidden = 128
        self.num_embed = 256
        self.gpu = 2
        self.frequent = 20
        self.global_step = int(self.example / self.batch_size * self.epoch)
        self.vocab_path = './d/data/chinese-poetry/model/vocab.json'
        self.save_model_name = './d/data/chinese-poetry/model/model_name'


if __name__ == '__main__':
    param = Param()
    database_path = './d/data/chinese-poetry/ci/ci.db'
    save_path = './/d/data/tensorflow_rnn_ci/data/poems.txt'
    # from_database_data(database_path=database_path, save_path=save_path)
    start_label = 1
    invalid_label = -1
    buckets = [10, 30, 60, 80, 95]
    train_sent, vocab = tokenize_text(save_path, start_label=start_label, invalid_label=invalid_label)
    json.dump(vocab, fp=open(param.vocab_path, 'w'))
    print(len(vocab))
    tmp = []
    for i in train_sent:
        tmp.append(len(i))
    print(max(tmp), min(tmp))
    data_train = BucketSentenceIter(train_sent, batch_size=param.batch_size, buckets=buckets,
                                    invalid_label=invalid_label)
    stack = mx.rnn.SequentialRNNCell()
    for i in range(param.num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=param.num_hidden, prefix='lstm_l%d_' % i))


    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        #embed = mx.sym.Embedding(data=data, input_dim=len(vocab), output_dim=param.num_embed, name='embed')

        stack.reset()
        output, _ = stack.unroll(seq_len, inputs=data, merge_outputs=True)
        print('seq-len:', seq_len)
        pred = mx.sym.Reshape(output, shape=(-1, param.num_hidden))
        fc1 = mx.sym.FullyConnected(data=pred, num_hidden=param.num_hidden, name='fc1')
        fc1_bn = mx.sym.BatchNorm(data=fc1, fix_gamma=False, eps=2e-5, momentum=0.99, name="fc1_bn")
        fc1_act = mx.sym.Activation(data=fc1_bn, act_type='relu', name='fc1_act')

        pred = mx.sym.FullyConnected(data=fc1_act, num_hidden=len(vocab), name='pred')
        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)


    contexts = [mx.gpu(i) for i in range(param.gpu)]

    model = mx.mod.BucketingModule(sym_gen=sym_gen,
                                   default_bucket_key=data_train.default_bucket_key,
                                   context=contexts)

    sgd_opt = mx.optimizer.SGD(learning_rate=param.learn_rate, momentum=param.moment, wd=param.wd,
                               rescale_grad=1 / param.batch_size)

    def lr_callback(cp):
        global_nbatch = cp.epoch * int(param.example / param.batch_size) + cp.nbatch
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
        eval_metric=[Accuracy(), CrossEntropy()],
        optimizer=sgd_opt,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        epoch_end_callback=mx.rnn.do_rnn_checkpoint(stack, param.save_model_name, 5),
        num_epoch=param.epoch,
        batch_end_callback=[mx.callback.Speedometer(param.batch_size, param.frequent), lr_callback])
