import sys
import math
import numpy as np

dict_w = {}
dict_l = {}

def read_feature_1(train_file):
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
    return train_data, train_seq_idx

def read_feature_2(train_file):
   pass


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
    return data, seq_idx

def get_exp_theta_k_mul_x_i(theta_k, i_x):
    return math.exp(theta_k[0] + theta_k[i_x+1])

def get_sum_exp_theta_x(i_x):
    sum_exp = 0
    for j in range(K):
        sum_exp += get_exp_theta_k_mul_x_i(theta[j], i_x)
    return sum_exp


def sgd(i_x, y_i):
    sum_exp = get_sum_exp_theta_x(i_x)

    deriv_theta = []
    for k in range(K):
        exp_theta_k_x_i = get_exp_theta_k_mul_x_i(theta[k], i_x)
        factor =  -((1 if k == y_i else 0) - exp_theta_k_x_i / sum_exp)
        deriv_theta_k = [0 for col in range(M)]
        deriv_theta_k[0] = factor
        deriv_theta_k[i_x+1] = factor
        deriv_theta.append(deriv_theta_k)

    eta_v = [eta for col in range(M)]
    for k in range(K):
        theta[k] = theta[k] - np.multiply(eta_v, deriv_theta[k])

def get_cur_llh(data):
    n = len(data)
    J_theta = 0
    for i in range(n):
        k = dict_l[data[i][1]]
        i_x = dict_w[data[i][0]]
        J_theta += math.log(get_exp_theta_k_mul_x_i(theta[k], i_x) / get_sum_exp_theta_x(i_x))
    avr_J = - J_theta / n
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
        P = []
        for k in range(K):
            P.append(get_exp_theta_k_mul_x_i(theta[k], dict_w[a]))
        prt_label = P.index(max(P))
        for label_name, label_num in dict_l.iteritems():
            if label_num == prt_label:
                prt_label_name = label_name
        f.write('%s\n' % (prt_label_name))
        if label != prt_label:
            err_prt_cnt += 1
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
    data_train, train_seq_idx = read_feature_1(train_file)
elif featr_flag == 2:
    data_train, train_seq_idx = read_feature_2(train_file)

# initialize parameters
data_vldat, vldat_seq_idx = get_data(vldat_file)
data_test, test_seq_idx = get_data(test_file)
K = len(dict_l)
M = len(dict_w) + 1
eta = 0.5
if featr_flag == 1:
    theta = [[0 for row in range(M)] for col in range(K)]
elif featr_flag == 2:
    pass

# SGD
for epc in range(num_epoch):
    for idx in range(len(data_train)):
        if featr_flag == 1:
            sgd(dict_w[data_train[idx][0]], dict_l[data_train[idx][1]])
        elif featr_flag == 2:
            pass
    avr_llh_train = get_cur_llh(data_train)
    print_llh(epc, avr_llh_train, metrc_out, 'train')
    avr_llh_vldat = get_cur_llh(data_vldat)
    print_llh(epc, avr_llh_vldat, metrc_out, 'validation')

train_err = get_prt_err(data_train, train_out, train_seq_idx)
print_err(train_err, 'train')
test_err = get_prt_err(data_test, test_out, test_seq_idx)
print_err(test_err, 'test')