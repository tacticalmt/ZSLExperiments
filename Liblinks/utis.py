import os
import numpy as np


# from scipy import linalg

def data_pack(path_list, class_info,
              seman_dict):  # 输入类名文件夹的地址和类名信息list，model是读取semantic的函数地址，函数内对每个文件夹的内容遍历，抽出jpg格式的样本  seman_dict是语义字典 4
    samp_path = []  # 样本的读取路径
    label_true = []  # 样本类标
    samp_seman = []  # 样本的语义内嵌
    label_train = []
    lenth_info = len(class_info[0])  # 每个样本包含多少个信息
    print(len(path_list))
    # print(path_list)
    for i, class_root in enumerate(path_list):  # i为第i类
        for x in os.scandir(class_root):
            if x.name.endswith(".jpg"):
                samp_path.append(x.path)
                try:
                    float_label = float(class_info[i][0])  # 0为类标
                except:
                    print('第{0}条数据处理失败'.format(i))
                # float_label=float(class_info[i][0])#0为类标
                # print(float_label)
                label_true.append(float_label)
                samp_seman.append(seman_dict[class_info[i][1]])  # 1为类名字符串model[class_info[i][1]]
                label_train.append(i)
        if i == (len(class_info) - 1):
            break
            # for i_info,info in enumerate(class_info[i]):
            #   a.append(info)
    return samp_path, label_true, samp_seman, label_train


def get_path_info(name_list, name_dict, root,
                  cate='trainingset'):  # namelist是类名的列表， name是字典，每个键值都是一个包含类标和类名的list  tar_path为文件夹地址 3  训练文件夹trainingset,测试集文件夹testingset
    path_dir = []
    cl_info = []
    done_list = []
    # print(len(path_dir))
    # print(name_list)
    tar_path = os.path.join(root, cate)
    for i, word in enumerate(name_list):
        # [path_dir.append(x.path) for x in os.scandir(tar_path) if x.name.endswith(word)]
        for x in os.scandir(tar_path):
            if x.name.endswith(word) and word not in done_list:
                path_dir.append(x.path)
                done_list.append(word)
        cl_info.append(name_dict[word])
        # print(i)
        # print(word)
        if i == (len(name_list) - 1):
            break
        # print(path_dir)
    # print(name_list)
    # print(len(name_list))
    print(len(path_dir))
    # print(path_dir)

    return path_dir, cl_info  # path_dir每个类的样本的文件夹路径，cl_info是对应的真实类标


def get_name_list(root, cls='trainclasses'):  # 在指定目录下，找指定文件名的地址    1 #cls可以是trainclasses 或者 testclasses或者gzslclasses
    if cls == 'trainclasses' or cls == 'testclasses' or cls == 'gzslclasses':
        print(cls)
    else:
        print('error type')
        return -1

    label_path = [x.path for x in os.scandir(root) if cls in x.path]  # cls指定的文件名目录里面找类名 搜索根目录下带有类标名的txt文档
    word = []
    for i, file_label in enumerate(label_path):
        f = open(file_label, 'r')
        while True:
            temp_w = f.readline().strip()
            if not temp_w:
                break
            word.append(temp_w)
        f.close()
    return word  # 返回不同的name list给get_path_info函数来控制gzsl还是zsl


def get_name_list_fea(root, cls='trainvalclasses'):
    if cls == 'trainvalclasses' or cls == 'testclasses' or cls == 'gzslclasses':
        print(cls)
    else:
        print('error type')
        return -1
    label_path = [x.path for x in os.scandir(root) if cls in x.path]  # cls指定的文件名目录里面找类名 搜索根目录下带有类标名的txt文档
    word = []
    for i, file_label in enumerate(label_path):
        f = open(file_label, 'r')
        while True:
            temp_w = f.readline().strip()
            if not temp_w:
                break
            word.append(temp_w)
        f.close()
    return word  # 返回不同的name list给get_path_info函数来控制gzsl还是zsl


def get_dict_name(root):  # 获取name字典  从包含类标和类名的文件获取  root为文件夹路径 每个词条里包含一个列表，这个列表包含了真实类标和对应的自然语言名字 2
    # 返回name
    label_path = os.path.join(root, 'classes.txt')
    dict_name = {}
    wordl_l = []
    cl_f = open(label_path, 'r')

    while True:
        temp_cl = cl_f.readline().strip()
        if not temp_cl:
            break

        wordl_l.append(temp_cl)
    cl_f.close()
    temp_label = []

    for key_word in wordl_l:
        # print(key_word)
        v = key_word.split('\t', 1)
        # print(v)
        # print(len(v))
        temp_label.append(v)

    # else:
    #   v=[key_word]
    #   print(v)

    for i_dict in temp_label:
        dict_name[i_dict[1]] = i_dict

    # print(temp_label)
    # print(label_dict)

    return dict_name


def zsl_el(S, S_tesamp, label, true_label_index, p=5):  # S_tesamp是测试样本(一行一个样本)，跟每个类的semantic算cosine距离,label是真实类标向量
    n = 0
    # print(S.shape)
    # print(S_tesamp.shape)
    dist_cos = np.dot(S_tesamp, S.T) / (np.linalg.norm(S) * np.linalg.norm(S_tesamp))
    # print(dist_cos)
    hit_mat = np.zeros([S_tesamp.shape[0], p])  # 前p命中矩阵
    test_label_mat = np.zeros([S_tesamp.shape[0], p])
    sort_mat = np.argsort(-dist_cos, axis=1)
    for i in range(S_tesamp.shape[0]):
        hit_mat[i, :] = sort_mat[i, 0:p]  # 对应现在训练类标的索引，需要转化成对应的真实类标
        hit_mat = hit_mat.astype(int)
    # print(hit_mat)
    for i in range(S_tesamp.shape[0]):
        for l_i in range(p):
            test_label_mat[i, l_i] = true_label_index[hit_mat[i, l_i]]  # true label是tensor
    test_label_mat = test_label_mat.astype(int)
    # print(test_label_mat)
    # print(label)
    for i in range(S_tesamp.shape[0]):
        # print(test_label_mat[i,:])
        # print(label[i])
        if label[i] in test_label_mat[i, :]:
            n = n + 1
            # print(n)

    acc = n / S_tesamp.shape[0]
    return acc, n, S_tesamp.shape[0]


def get_semantic(word_list, name_dict, semantic_dict):  # 给出对应的类标名字，和语义字典，输出语义的列表
    temp = []
    name_list = []
    for word in word_list:
        temp.append(semantic_dict[word])
        temp_name = int(name_dict[word][0])
        name_list.append(temp_name)
    semantic_mat = np.array(temp)
    true_label_index = np.array(name_list)
    return semantic_mat, true_label_index


def get_attri_label(root):
    a_path = os.path.join(root, 'predicate-matrix-binary.txt')
    word_att = []
    att = open(a_path, 'r')  # 打开attribute文件
    while True:
        temp_att = att.readline().strip()
        if not temp_att:
            break
        word_att.append(temp_att)
    att.close()
    divided_att = []
    for key_word in word_att:
        v = key_word.split(' ')
        divided_att.append(v)
    divided_np = np.array(divided_att)
    att_int = divided_np.astype(int)
    return att_int


def create_name_to_attri(name_list, attri_list):  # name list要包含全部类名的lists,返回一个包含全部类的attribute的字典
    attri_dict = {}
    for i, word_i in enumerate(name_list):
        attri_dict[word_i] = attri_list[i]
    return attri_dict


def data_pack_attr(path_list, class_info, attri_dict):
    # 获取属性类标
    samp_path = []  # 样本的读取路径
    label_true = []  # 样本类标
    attri_label = []  # 训练时用的类标
    lenth_info = len(class_info[0])  # 每个样本包含多少个信息
    label_train = []
    # print(len(path_list))
    for i, class_root in enumerate(path_list):  # i为第i类
        for x in os.scandir(class_root):
            if x.name.endswith(".jpg"):
                samp_path.append(x.path)
                try:
                    float_label = float(class_info[i][0])  # 0为类标
                except:
                    print('第{0}条数据处理失败'.format(i))
                # float_label=float(class_info[i][0])#0为类标
                # print(float_label)
                # print(class_info[i][0])
                print(float_label)
                label_true.append(float_label)
                # print(attri_dict[class_info[i][1]])
                attri_label.append(attri_dict[class_info[i][1]])  # 1为类名字符串model[class_info[i][1]]
                label_train.append(i)  # softmax层用的类标,embed层也要用
        if i == (len(class_info) - 1):
            break
    return samp_path, label_true, attri_label, label_train


def max_singular_value(W, u=None, Ip=1):  # IP为迭代次数   W为tensor
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")

    W_np = W.detach().cpu().numpy()
    if u is None:
        u = np.random.normal(size=(1, W.shape[0])).astype(np.float32)
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(np.dot(_u, W_np), eps=1e-12)
        _u = _l2normalize(np.dot(_v, W_np.T), eps=1e-12)

    sigma = np.dot(_u, np.dot(W_np, _v.T)).sum()
    return sigma, _u, _v


def _l2normalize(vect, eps=1e-12):  # 1e-12=1*10^(-12)
    l2norm = np.linalg.norm(vect)
    return vect / (l2norm + eps)


def get_sample_label(name_dict, tar_list):  # name_dict存储类标与类名对应关系，tar_list存储对应训练集或者测试集类名，返回类标数组
    obj_label_list = []

    for key_word in tar_list:
        temp = int(name_dict[key_word][0])
        obj_label_list.append(temp)

    return obj_label_list


def get_tar_set(tar_class_index, all_labels, all_samples, seman_list, seman_index):  # sample都传tensor
    tar_label = []
    tar_data = []
    tar_seman = []
    for i_label, label in enumerate(all_labels):
        if label in tar_class_index:
            index_sm = np.argwhere(seman_index == label)  # 用于找内嵌的索引 np.argwhere
            tar_seman.append(seman_list[index_sm[0][0]])
            tar_label.append(label)
            tar_data.append((all_samples[i_label, :]))  # 取出对应位置的样本
    # np_tar_data = np.array(tar_data)
    np_tar_seman = np.array(tar_seman)
    np_tar_label = np.array(tar_label)
    return tar_data, np_tar_seman, np_tar_label
