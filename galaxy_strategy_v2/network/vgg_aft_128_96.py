import mxnet as mx


def get_symbol(num_classes, bn_mom=0.99, **kwargs):
    data = mx.sym.Variable("data")
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(7, 7), pad=(1, 1), stride=(2, 2), num_filter=96, name="conv1_1")
    body1 = mx.sym.BatchNorm(data=conv1_1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1_1 = mx.symbol.Activation(data=body1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(data=relu1_1, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool1")

    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(5, 5), pad=(1, 1), stride=(2, 2), num_filter=256, name="conv2_1")
    body2 = mx.sym.BatchNorm(data=conv2_1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn2')
    relu2_1 = mx.symbol.Activation(data=body2, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(3, 3), stride=(2, 2), name="pool2")

    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="conv3_1")
    body3 = mx.sym.BatchNorm(data=conv3_1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn3')
    relu3_1 = mx.symbol.Activation(data=body3, act_type="relu", name="relu3_1")

    conv4_1 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="conv4_1")
    body4 = mx.sym.BatchNorm(data=conv4_1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn4')

    relu4_1 = mx.symbol.Activation(data=body4, act_type="relu", name="relu4_1")

    conv5_1 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="conv5_1")
    body5 = mx.sym.BatchNorm(data=conv5_1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn5')
    relu5_1 = mx.symbol.Activation(data=body5, act_type="relu", name="relu5_1")

    pool3 = mx.symbol.Pooling(
             data=relu5_1, pool_type="avg", kernel=(2, 1), stride=(2, 1), name="pool3")

    fc6 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 1), stride=(1, 1), num_filter=4096, name="conv_fc6")
    body6 = mx.sym.BatchNorm(data=fc6, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn6')

    relu6 = mx.symbol.Activation(data=body6, act_type="relu", name="relu6")
    apool6 = mx.symbol.Pooling(
         data=relu6, pool_type="avg", kernel=(1, 4), stride=(1, 1), name="apool6")

    fla = mx.symbol.Flatten(data=apool6)

    fc7 = mx.sym.FullyConnected(data=fla, num_hidden=1024, name='fc1')
    body7 = mx.sym.BatchNorm(data=fc7, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn7')
    relu7 = mx.symbol.Activation(data=body7, act_type="relu", name="relu7")

    fc8 = mx.sym.FullyConnected(data=relu7, num_hidden=num_classes, name='fc2')

    return mx.sym.SoftmaxOutput(data=fc8, name='softmax')