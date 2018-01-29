import glob
import random
import os


def gen_list(path, num):
    train_lst = []
    test_lst = []
    with open(path, 'r') as input_path:
        for row in input_path:
            tmp_lst = glob.glob(os.path.join(row.strip(), '*.png'))
            tmp_test_lst = random.sample(tmp_lst, num)
            for k in tmp_test_lst:
                tmp_lst.remove(k)
            train_lst.append(tmp_lst)
            test_lst.append(tmp_test_lst)
    train_lst = sum(train_lst, [])
    test_lst = sum(test_lst, [])
    random.shuffle(train_lst)
    return train_lst, test_lst


if __name__ == '__main__':
    path = './d/project/image_data/sdata_image/total.lst'
    train_lst, test_lst = gen_list(path=path, num=5)
    train_rec_path = './d/project/image_data/sdata_image/train_shuffle_rec.lst'
    test_rec_path = './d/project/image_data/sdata_image/test_rec.lst'
    fo = open(train_rec_path, 'w')
    foo = open(test_rec_path, 'w')
    for i in train_lst:
        label = i.split(':')[-1].split('.')[0].split('_')[1]
        fo.write('%s\t%s\t%s\n' % (0, int(label), i))
    for k in test_lst:
        label = k.split(':')[-1].split('.')[0].split('_')[1]
        foo.write('%s\t%s\t%s\n' % (0, int(label), k))
    fo.close()
    foo.close()
