import sys
import math
import copy
import time

start = time.clock()

dict_w = {}
dict_l = {}
dict_w_2 = {}

def read_feature(train_file):
    with open(train_file) as tsvfile:
        ii = 0
        jj = 0
        idx = 0
        train_data = []
        train_seq_idx = []
        for line in tsvfile:
            if line != '\n':
                a, b = line.strip().split('\t')
                data_i = [a, b]
                train_data.append(data_i)
                if a not in dict_w:
                    dict_w[a] = ii
                    ii += 1
                if b not in dict_l:
                    dict_l[b] = jj
                    jj += 1
                idx = idx + 1
            else:
                train_seq_idx.append(idx)
    train_seq_idx.append(-5)
    train_seq_idx.append(-6)
    return train_data, train_seq_idx


def parse_feature(train_data, train_seq_idx):
    M = len(dict_w)
    data_train_mode_2 = []
    idx = 0
    tmp = []
    for i in range(len(train_data)):
        if idx == 0:
            tmp.append(1)
            tmp.append(dict_w[train_data[i][0]] + 3 + M)
            idx += 1
        if i+1 <> train_seq_idx[0] and i + 1 <> len(train_data):
            tmp.append(dict_w[train_data[i+1][0]] + 3 + 2*M)
            new_tmp = copy.copy(tmp)
            data_train_mode_2.append(new_tmp)
            tmp.pop(0)
            tmp[0] = tmp[0] - M
            tmp[1] = tmp[1] - M
        else:
            train_seq_idx.pop(0)
            tmp.append(2)
            new_tmp = copy.copy(tmp)
            data_train_mode_2.append(new_tmp)
            tmp[:] = []
            idx = 0
    return data_train_mode_2

def get_data(file):
    with open(file) as tsvfile:
        data = []
        seq_idx = []
        i = 0
        for line in tsvfile:
            if line != '\n':
                a, b = line.strip().split('\t')
                data_i = [a, b]
                data.append(data_i)
                i = i + 1
            else:
                seq_idx.append(i)
    seq_idx.append(-5)
    seq_idx.append(-6)
    return data, seq_idx


def get_exp_theta_k_mul_x_i(theta_k, i_x):
    return math.exp(theta_k[0] + theta_k[i_x+1])

def get_exp_theta_k_mul_x_i_mode2(theta_k, is_x):
    tmp_theta_sum = theta_k[0]
    for i in range(len(is_x)):
        tmp_theta_sum += theta_k[is_x[i]]
    return math.exp(tmp_theta_sum)

def get_sum_exp_theta_x(i_x):
    sum_exp = 0
    for j in range(K):
        sum_exp += get_exp_theta_k_mul_x_i(theta[j], i_x)
    return sum_exp

def get_sum_exp_theta_x_mode2(is_x):
    sum_exp = 0
    for j in range(K):
        sum_exp += get_exp_theta_k_mul_x_i_mode2(theta[j], is_x)
    return sum_exp


def sgd(i_x, y_i):
    sum_exp = get_sum_exp_theta_x(i_x)

    deriv_theta = []
    for k in range(K):
        exp_theta_k_x_i = get_exp_theta_k_mul_x_i(theta[k], i_x)
        factor =  -((1 if k == y_i else 0) - exp_theta_k_x_i / sum_exp)
        deriv_theta.append(factor)

    for k in range(K):
        theta[k][0] = theta[k][0] - deriv_theta[k] * eta
        theta[k][i_x+1] = theta[k][i_x+1] - deriv_theta[k] * eta

def sgd_mode2(is_x, y_i):
    sum_exp = get_sum_exp_theta_x_mode2(is_x)

    deriv_theta = []
    for k in range(K):
        exp_theta_k_x_i = get_exp_theta_k_mul_x_i_mode2(theta[k], is_x)
        factor = -((1 if k == y_i else 0) - exp_theta_k_x_i / sum_exp)
        deriv_theta.append(factor)

    for k in range(K):
        theta[k][0] = theta[k][0] - eta * deriv_theta[k]
        for i in range(len(is_x)):
            theta[k][is_x[i]] = theta[k][is_x[i]] - deriv_theta[k] * eta

def get_cur_llh(data):
    n = len(data)
    J_theta = 0
    for i in range(n):
        k = dict_l[data[i][1]]
        i_x = dict_w[data[i][0]]
        J_theta += math.log(get_exp_theta_k_mul_x_i(theta[k], i_x) / get_sum_exp_theta_x(i_x))
    avr_J = - J_theta / n
    return avr_J

def get_cur_llh_mode2(data_mode_2, data):
    n = len(data)
    J_theta = 0
    for i in range(n):
        k = dict_l[data[i][1]]
        is_x = data_mode_2[i]
        J_theta += math.log(get_exp_theta_k_mul_x_i_mode2(theta[k], is_x) / get_sum_exp_theta_x_mode2(is_x))
    avr_J = - (J_theta / n)
    return avr_J

def print_llh(epc, avr_llg, metrc_out, data_catg):
    f = open(metrc_out, 'a')
    f.write('Epoch=%d likelihood %s: %.6f\n' % (epc+1, data_catg, avr_llg));
    f.close()

def get_prt_err(data, out_label_file, seq_idx):
    f = open(out_label_file, 'w')
    err_prt_cnt = 0
    for idx in range(len(data)):
        if idx == seq_idx[0]:
            seq_idx.pop(0)
            f.write('\n')
        a, b = data[idx]
        label = dict_l[b]
        max_P = -1
        max_i = -1
        for label_name, label_num in dict_l.iteritems():
            cur_P = get_exp_theta_k_mul_x_i(theta[label_num], dict_w[a])
            if cur_P > max_P:
                maxLabel = label_name
                max_P = cur_P
                max_i = label_num
            if cur_P == max_P:
                if cmp(maxLabel, label_name) > 0:
                    maxLabel = label_name
                    max_P = cur_P
                    max_i = label_num
        prt_label_name = maxLabel
        prt_label = max_i
        f.write('%s\n' % (prt_label_name))
        if label != prt_label:
            err_prt_cnt += 1
    f.write('\n')
    f.close()
    return err_prt_cnt / float(len(data))

def get_prt_err_mode2(data_mode_2, data, out_label_file, seq_idx):
    f = open(out_label_file, 'w')
    err_prt_cnt = 0
    for idx in range(len(data)):
        if idx == seq_idx[0]:
            seq_idx.pop(0)
            f.write('\n')
        is_x = data_mode_2[idx]
        y_i = data[idx][1]
        label = dict_l[y_i]
        max_P = -1
        max_i = -1
        for label_name, label_num in dict_l.iteritems():
            cur_P = get_exp_theta_k_mul_x_i_mode2(theta[label_num], is_x)
            if cur_P > max_P:
                maxLabel = label_name
                max_P = cur_P
                max_i = label_num
            if cur_P == max_P:
                if cmp(maxLabel, label_name) > 0:
                    maxLabel = label_name
                    max_P = cur_P
                    max_i = label_num
        prt_label_name = maxLabel
        prt_label = max_i
        f.write('%s\n' % (prt_label_name))
        if label != prt_label:
            err_prt_cnt += 1
    f.write('\n')
    f.close()
    return err_prt_cnt / float(len(data))

def print_err(err, data_catg):
    f = open(metrc_out, 'a')
    f.write('error(%s): %.6f\n' % (data_catg, err) )

# get parameters from command line
train_file = sys.argv[1]
vldat_file = sys.argv[2]
test_file = sys.argv[3]
train_out = sys.argv[4]
test_out = sys.argv[5]
metrc_out = sys.argv[6]
num_epoch = int(sys.argv[7])
featr_flag = int(sys.argv[8])

# choose feature model
train_seq_idx = []
if featr_flag == 1:
    data_train, train_seq_idx = read_feature(train_file)
elif featr_flag == 2:
    data_train, train_seq_idx = read_feature(train_file)
    train_seq_idx_cpy = copy.copy(train_seq_idx)
    data_train_mode_2 = parse_feature(data_train, train_seq_idx_cpy)

# initialize parameters
data_vldat, vldat_seq_idx = get_data(vldat_file)
data_test, test_seq_idx = get_data(test_file)
if featr_flag == 2:
    test_seq_idx_cpy = copy.copy(test_seq_idx)
    vldat_seq_idx_cpy = copy.copy(vldat_seq_idx)
    data_test_mode_2 = parse_feature(data_test, test_seq_idx_cpy)
    data_vldat_mode_2 = parse_feature(data_vldat, vldat_seq_idx_cpy)

K = len(dict_l)
M = len(dict_w) + 1
eta = 0.5
if featr_flag == 1:
    theta = [[0 for row in range(M)] for col in range(K)]
elif featr_flag == 2:
    theta = [[0 for row in range(3*M)] for col in range(K)]

# SGD
for epc in range(num_epoch):
    for idx in range(len(data_train)):
        if featr_flag == 1:
            sgd(dict_w[data_train[idx][0]], dict_l[data_train[idx][1]])
        elif featr_flag == 2:
            sgd_mode2(data_train_mode_2[idx], dict_l[data_train[idx][1]])
    if featr_flag == 1:
        avr_llh_train = get_cur_llh(data_train)
        avr_llh_vldat = get_cur_llh(data_vldat)
    elif featr_flag == 2:
        avr_llh_train = get_cur_llh_mode2(data_train_mode_2, data_train)
        avr_llh_vldat = get_cur_llh_mode2(data_vldat_mode_2, data_vldat)
    print_llh(epc, avr_llh_train, metrc_out, 'train')
    print_llh(epc, avr_llh_vldat, metrc_out, 'validation')
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
    start = time.clock()

if featr_flag == 1:
    train_err = get_prt_err(data_train, train_out, train_seq_idx)
    test_err = get_prt_err(data_test, test_out, test_seq_idx)
elif featr_flag == 2:
    train_err = get_prt_err_mode2(data_train_mode_2, data_train, train_out, train_seq_idx)
    test_err = get_prt_err_mode2(data_test_mode_2, data_test, test_out, test_seq_idx)
print_err(train_err, 'train')
print_err(test_err, 'test')

elapsed = (time.clock() - start)
print("Time used:", elapsed)