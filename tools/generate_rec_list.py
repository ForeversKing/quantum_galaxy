import random
import numpy as np


def generate_rec_list(path, create_file_path):
    data_input = open(path, 'r')
    data_output = open(create_file_path, 'w')
    index = 0
    ss = []
    for i in data_input:
        a = i.strip()
        b = i.strip().split('/')
        name = b[-3]
        if len(ss) == 0:
            ss.append(name)
        if name != ss[0]:
            ss[0] = name
            index += 1
        data_output.write('%s\t%s\t%s\n' % (0, index, a))
    data_input.close()
    data_output.close()


def rec_list(path, create_file_path):
    data_input = open(path, 'r')
    data_output = open(create_file_path, 'w')
    index = 0
    for i in data_input:
        a = i.strip()
        data_output.write('%s\t%s\t%s\n' % (0, index, a))
        index += 1
    data_input.close()
    data_output.close()


def random_shuffle(path, create_file_path):
    data_input = open(path, 'r')
    data_output = open(create_file_path, 'w')
    ss = []
    for i in data_input:
        ss.append(i)
    random.shuffle(ss)
    for j in ss:
        data_output.write(j)
    data_input.close()
    data_output.close()


def get_test_image(path, create_file_path):
    data_input = open(path, 'r')
    data_output = open(create_file_path, 'w')
    ss = []
    for i in data_input:
        a = i.strip().split('\t')
        label = a[1]
        index = 1
        if len(ss) == 0:
            ss.append(label)
            data_output.write(i)
        elif label == ss[0]:
            data_output.write(i)
            index += 1
        elif label != ss[0]:
            ss[0] = label
            data_output.write(i)
    data_input.close()
    data_output.close()


def get_test_gallary_image(path, create_file_path):
    data_input = open(path, 'r')
    data_output = open(create_file_path, 'w')
    ss = []
    # for i in data_input:
    #     a = i.strip().split('\t')
    #     label = a[1]
    #     if len(ss) < 1:
    #         ss.append(label)
    #         if a[2].split('/')[-1] == "0.png.jpg":
    #             data_output.write(i)
    #     elif label != ss[0]:
    #         #ss = []
    #         #ss.append(label)
    #         if a[2].split('/')[-1] == "0.png.jpg":
    #             data_output.write(i)
    #             ss = []
    #             ss.append(label)
    for i in data_input:
        a = i.strip().split('\t')
        name_id = a[1]
        if len(ss) == 0:
            print(name_id)
            ss.append(name_id)
            data_output.write(i)
        if name_id not in ss:
            print(name_id)
            ss[0] = name_id
            data_output.write(i)
    data_input.close()
    data_output.close()


def check_file(origin_path, check_file_path):
    fo = open('/home/yulongwu/e/voice_data/voice_label/test/test_file.lst', 'w')
    origin = open(origin_path)
    download = open(check_file_path)
    origin = origin.readlines()
    download = download.readlines()
    origin = set(map(lambda x: x.strip(), origin))
    download = set(map(lambda x: x.strip(), download))
    print(len(origin), len(download))
    left = origin - download
    for l in left:
        fo.write(l + '\n')
    fo.close()


def get_feature_id(path):
    data_input = open(path, 'r')
    ss = []
    for i in data_input:
        ss.append(i.strip().split('\t')[1])
    data_input.close()
    return ss


def relative_test(master_path, input_path, out_path):
    fm = open(master_path)
    fi = open(input_path)
    out_file = open(out_path, 'w')
    ss = []
    for i in fi:
        str_lst = i.strip().split('\t')
        num_id = str_lst[1]
        mp3_scile = str_lst[2].split('/')[-2]
        if len(ss) == 0:
            ss.append(num_id)
            for j in fm:
                mast_str_lst = j.strip().split('\t')
                mast_num_id = mast_str_lst[1]
                mast_mp3_scile = mast_str_lst[2].split('/')[-2]
                if num_id == mast_num_id and mp3_scile == mast_mp3_scile:
                    out_file.write(j)
        if num_id != ss[0]:
            ss[0] = num_id
            fm = open(master_path)
            for j in fm:
                print(j)
                mast_str_lst = j.strip().split('\t')
                mast_num_id = mast_str_lst[1]
                mast_mp3_scile = mast_str_lst[2].split('/')[-2]
                if num_id == mast_num_id and mp3_scile == mast_mp3_scile:
                    out_file.write(j)
            fm.close()

    fm.close()
    fi.close()
    out_file.close()


def t_people(master_path, input_path, out_path):
    fm = open(master_path)
    fi = open(input_path)
    out_file = open(out_path, 'w')
    ss = []
    for i in fi:
        str_lst = i.strip().split('/')
        people_name = str_lst[0]
        mp3_scile = str_lst[1]
        fm.seek(0, 0)
        for j in fm:
            mast_str_lst = j.strip().split('/')
            mast_people_name = mast_str_lst[7]
            mast_mp3_scile = mast_str_lst[8]
            if people_name == mast_people_name and mp3_scile == mast_mp3_scile:
                out_file.write(j)

    fm.close()
    fi.close()
    out_file.close()


def diff_label(path):
    fi = open(path)
    ss = []
    for i in fi:
        ss.append(int(i.strip().split('\t')[1]))
    print(len(set(ss)))


def feature_contract(path, create_file_path):
    fi = open(path)
    fo = open(create_file_path, 'w')
    count = len(fi.readlines())
    fi.seek(0, 0)
    index = 0
    ss = []
    diff_id = []
    mat_s = np.ndarray((count, 1025), dtype=float)
    for i in fi:
        line = i.strip()
        if line == '':
            break
        data = np.array(line.split(' '), dtype=float)
        mat_s[index] = data
        index += 1
    print(mat_s.shape)
    for id_num in range(count):
        diff_id.append(mat_s[id_num][0])
    diff_id = set(diff_id)
    new_row = len(diff_id)
    print(new_row)
    new_mat_s = np.ndarray((new_row, 1025), dtype=float)
    new_index = 0
    for j in range(mat_s.shape[0]):
        if len(ss) == 0:
            sum_1 = 0
            ss.append(mat_s[j])
            diff_id.remove(mat_s[j][0])
            for k in range(mat_s.shape[0]):
                if mat_s[k][0] == mat_s[j][0]:
                    ss.append(mat_s[k])
                    #mat_s = np.delete(mat_s, k, axis=0)
            for feature_num in ss:
                sum_1 += feature_num
            new_mat_s[new_index] = sum_1 / len(ss)
            new_index += 1
        if len(ss) != 0:
            if mat_s[j][0] in diff_id:
                sum_1 = 0
                ss.clear()
                ss.append(mat_s[j])
                diff_id.remove(mat_s[j][0])
                for k in range(mat_s.shape[0]):
                    if mat_s[k][0] == mat_s[j][0]:
                        ss.append(mat_s[k])
                        #mat_s = np.delete(mat_s, k, axis=0)
                for feature_num in ss:
                    sum_1 += feature_num
                new_mat_s[new_index] = sum_1 / len(ss)
                new_index += 1
                print(new_index)
    for i in range(new_row):
        new_mat_s[i][0] = int(new_mat_s[i][0])
    for i in range(new_row):
        feat_str = list(new_mat_s[i].astype(str))
        fo.write("{}\n".format(' '.join(feat_str)))
    fi.close()
    fo.close()




def id_num(master_path, path, create_file_path):
    fi = open(path)
    check_path = open(master_path)
    fo = open(create_file_path, 'w')
    ss = []
    dict_num = {}
    for i in fi:
        a = i.strip().split('\t')
        dict_num[a[0]] = int(a[1])

    print(len(dict_num))
    fi.seek(0, 0)
    for j in check_path:
        name_str = j.strip().split('/')[-2]
        print(dict_num[name_str])
        fo.write('%s\t%s\t%s\n' % (0, dict_num[name_str], j.strip()))
    print(len(dict_num))
    fi.close()
    check_path.close()
    fo.close()
    #diff_num = set(ss)

def diff_id(path):
    fi = open(path)
    ss = []
    for i in fi:
        ss.append(i.strip().split('\t')[1])
    print(len(set(ss)))
    fi.close()


def line_correlation(data_x, data_y):
   return np.corrcoef(data_x, data_y)


def rank_num(path, out_path):
    fi = open(path, 'r')
    fo = open(out_path, 'w')
    ss = []
    tmp = '/home/yulongwu/e/voice_data/voice_label/tmp/tmp/21'
    for i in fi:
        ss.append(int(i.strip().split('/')[-1].split('.')[0]))
    ss.sort()
    for k in range(len(ss)):
        fo.write('%s/%s.png\n' % (tmp, ss[k]))
    fi.close()
    fo.close()


def rec_lst(path, creat_path):
    fi = open(path, 'r')
    fo = open(creat_path, 'w')
    for k in fi:
        tmp = k.strip().split('\t')[1]
        fo.write('%s\t%s\t%s\n' % (0, int(tmp) - 1, k.strip().split('\t')[-1]))
    fi.close()
    fo.close()


if __name__ == "__main__":
    master_path = '/home/yulongwu/Downloads/asr_sundanese/total_image.lst'
    path = '/home/yulongwu/d/data/malware_traindata/mmcc_train_data_byte2image/train_rec_v1.lst'
    create_file_path = '/home/yulongwu/d/data/malware_traindata/mmcc_train_data_byte2image/train_shuffle_rec.lst'
    random_shuffle(path, create_file_path)

