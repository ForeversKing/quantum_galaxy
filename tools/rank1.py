"""plot the roc curve
need: two feature file, one is for gallery data, the other is for probe data
and the feature file expect label id in first column and features in other columns,
rows for each images
eg. python this.py gallery_feature_file probe_feature_file
"""
import numpy as np
import datetime


# this is for error check

def read_id_feature(file_name):
    """
    txt file format:
    id_1 feature_1 feature_2 ...... feature_n
    id_2 feature_1 feature_2 ...... feature_n
    id above means label of image
    """
    data = np.genfromtxt(file_name, dtype=float, delimiter=' ')
    id, feature = np.split(data, [1], axis=1)
    print("read feature over...")
    return id.ravel().astype(int), feature


def find_idx(a, b):
    """
    :param a: like [1, 3, 3, 4, 2]
    :param b: like [2, 3], the element must be unique
    :return: [4, 1, 2]
    """
    ret = []
    for i in b:
        tmp = np.where(a == i)[0]
        if len(tmp) > 0:
            ret.extend(tmp.tolist())
    return ret


def cal(N_file, M_file):
    """
    :param M_file: probe feature matrix file
    :param N_file: gallery feature matrix file
    :return:
    """
    print("start to read features from files...")

    # 读取特征矩阵
    N_id, N_feature = read_id_feature(N_file)
    M_id, M_feature = read_id_feature(M_file)

    print("start to calculate the similarity...")

    print("gallery feature have [%d] samples and [%d] features" % (N_feature.shape[0], N_feature.shape[1]))
    print("probe feature have [%d] samples and [%d] features" % (M_feature.shape[0], M_feature.shape[1]))
    # relu
    N_feature[N_feature < 0.] = 0.
    M_feature[M_feature < 0.] = 0.
    # calculate sqrt
    N_feature = np.sqrt(N_feature)
    M_feature = np.sqrt(M_feature)
    # calculate norm
    M_norm = M_feature / np.sqrt(np.sum(np.square(M_feature), axis=1, keepdims=True))
    N_norm = N_feature / np.sqrt(np.sum(np.square(N_feature), axis=1, keepdims=True))

    similar_mat = np.dot(M_norm, N_norm.T)
    # rows: number of probe images, cols: number of gallery images
    rows, cols = similar_mat.shape
    mask = (np.ones([rows, 1]) * N_id) == (np.ones([cols, 1]) * M_id).T
    max_idx = np.argmax(similar_mat, axis=1)
    rank1 = np.sum(mask[np.arange(rows), max_idx]) / rows
    return rank1


def arg_parser():
    import argparse
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.png'
    parser = argparse.ArgumentParser()
    parser.add_argument('gallery_feature', help='gallery feature path')
    parser.add_argument('probe_feature', help='probe feature path')
    parser.add_argument('--o', type=str, default=now, help='save image file name')
    return parser.parse_args()


def main():
    args = arg_parser()
    rank1 = cal(args.gallery_feature, args.probe_feature)
    print('rank1:', rank1)


if __name__ == "__main__":
    main()
