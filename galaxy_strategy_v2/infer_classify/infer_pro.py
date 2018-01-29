import mxnet as mx
from galaxy_strategy_v2.unit.data_iterator import get_infer_data_iter
from galaxy_strategy_v2.unit.generate_rec_list import get_feature_id
import numpy as np


class InferParameter():
    def __init__(self):
        self.batch_size = 1
        self.shape = (3, 256, 125)
        self.path = "./d/project/image_data/sdata_image/test_rec.rec"
        self.id_path = "./d/project/image_data/sdata_image/test_rec.lst"
        self.load_path = './d/project/image_data/sdata_image/train_vgg/vgg_model'
        self.path_checkpoint = 50
        self.data_shapes = (self.batch_size, 3, 256, 125)
        self.feature_filepath = './feature_data/voice/test_prob.feature'
        self.data_iter = get_infer_data_iter(self.path, self.shape, self.batch_size)
        self.extract_feature_layer_name = 'fc2_output'
        self.test_protocol_type = 'rank'


infer_param = InferParameter()
sym, arg_params, aux_params = mx.model.load_checkpoint(infer_param.load_path, infer_param.path_checkpoint)
all_layers = sym.get_internals()
fe_sym = all_layers[infer_param.extract_feature_layer_name]
fe_mod = mx.mod.Module(symbol=fe_sym, context=[mx.gpu()], label_names=None)
fe_mod.bind(for_training=False, data_shapes=[('data', infer_param.data_shapes)])
fe_mod.set_params(arg_params, aux_params)

lst = get_feature_id(infer_param.id_path)
label_count = 0
count = 0
while True:
    try:
        test_data = infer_param.data_iter.next()
    except StopIteration:
        break
    fe_mod.forward(test_data, is_train=False)
    output = fe_mod.get_outputs()[0].asnumpy()
    output = np.exp(output)
    output = output / np.sum(output)
    pre_label = np.argmax(output)
    print(pre_label, lst[label_count])
    if int(pre_label) == int(lst[label_count]):
        count += 1
    print(lst[label_count], output)
    label_count += 1

print('result:', count / label_count)