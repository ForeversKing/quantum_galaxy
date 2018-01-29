import mxnet as mx
import os, time
import numpy as np
from mxnet.io import DataBatch


class InferParameter():
    def __init__(self):
        self.batch_size = 1
        self.load_path = './d/project/rnn_data/model/model_name'
        self.path_checkpoint = 10
        self.seq_len = 3360
        self.test_idx_file = './d/project/rnn_data/test_idx.txt'
        self.test_rec_file = './d/project/rnn_data/test_rec_data.rec'
        self.data_shapes = (self.batch_size, 4, self.seq_len)
        self.extract_feature_layer_name = 'fc5_output'
        self.test_protocol_type = 'rank'


start = time.time()

infer_param = InferParameter()
sym, arg_params, aux_params = mx.model.load_checkpoint(infer_param.load_path, infer_param.path_checkpoint)
all_layers = sym.get_internals()
fe_sym = all_layers[infer_param.extract_feature_layer_name]
fe_mod = mx.mod.Module(symbol=fe_sym, context=[mx.gpu()], label_names=None)
fe_mod.bind(for_training=False, data_shapes=[('data', infer_param.data_shapes)])
fe_mod.set_params(arg_params, aux_params)
count = 0
total = 0
test_record = mx.recordio.MXIndexedRecordIO(infer_param.test_idx_file, infer_param.test_rec_file, 'r')
for idx in range(4):
    item = test_record.read_idx(idx)
    header, data = mx.recordio.unpack(item)
    label = int(header.label[0])
    data = np.frombuffer(data, np.float32)
    print('*******', data.shape)
    # data = data[-infer_param.seq_len:]
    data = mx.nd.array(data.reshape(1, -1, infer_param.seq_len))
    data = DataBatch([data], pad=None, index=None)
    fe_mod.forward(data, is_train=False)
    output = fe_mod.get_outputs()[0].asnumpy()
    output = output.ravel()
    output = np.exp(output)
    output = output / np.sum(output)
    pre_label = np.argmax(output)
    if int(label) == int(pre_label):
        count += 1
    else:
        print(output)
    total += 1
print(count / total)

print(time.time() - start)
test_record.close()
