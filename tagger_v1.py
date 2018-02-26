import sys
import math
import numpy as np

dict_w = {}
dict_l = {}

def read_feature_1(train_file):
    with open(train_file) as tsvfile:
        ii = 0
        jj = 0
        for line in tsvfile:
            if line != '\n':
                a, b = line.strip().split('\t')
                if a not in dict_w:
                    dict_w[a] = ii
                    ii += 1
                if b not in dict_l:
                    dict_l[b] = jj
                    jj += 1

def read_feature_2(train_file):
    pass

def get_data(file):
    with open(file) as tsvfile:
        data = []
        for line in tsvfile:
            if line != '\n':
                a, b = line.strip().split('\t')
                data_i = [a, b]
                data.append(data_i)
    return data

def get_data_rep_1(a):
    w_i = [0 for col in range(M)]
    w_i[0] = 1
    w_i[dict_w[a]+1] = 1
    return w_i

def get_exp_theta_k_mul_x_i(theta_k, x_i):
    return math.exp(np.dot(theta_k, x_i));

def get_sum_exp_theta_x(x_i):
    sum_exp = 0
    for j in range(K):
        sum_exp += get_exp_theta_k_mul_x_i(theta[j], x_i)
    return sum_exp


def sgd(x_i, y_i):
    sum_exp = get_sum_exp_theta_x(x_i)

    deriv_theta = []
    for k in range(K):
        exp_theta_k_x_i = get_exp_theta_k_mul_x_i(theta[k], x_i)
        factor =  -((1 if k == y_i else 0) - exp_theta_k_x_i / sum_exp)
        deriv_theta_k = [x_ii * factor for x_ii in x_i]
        deriv_theta.append(deriv_theta_k)

    eta_v = [eta for col in range(M)]
    for k in range(K):
        theta[k] = theta[k] - np.multiply(eta_v, deriv_theta[k])

def get_cur_llh(data):
    n = len(data)
    J_theta = 0
    for i in range(n):
        k = dict_l[data[i][1]]
        x_i = get_data_rep_1(data[i][0])
        J_theta += math.log(get_exp_theta_k_mul_x_i(theta[k], x_i) / get_sum_exp_theta_x(x_i))
    avr_J = - J_theta / n
    return avr_J

def print_llh(epc, avr_llg, metrc_out, data_catg):
    f = open(metrc_out, 'a')
    f.write('Epoch=%d likelihood %s: %.6f\n' % (epc+1, data_catg, avr_llg));
    f.close()

def get_prt_err(testfile, out_label_file):
    f = open(out_label_file, 'w')
    err_prt_cnt = 0
    total_cnt = 0
    with open(testfile) as tsvfile:
        for line in tsvfile:
            if line != '\n':
                total_cnt += 1
                a, b = line.strip().split('\t')
                data_i = get_data_rep_1(a)
                label = dict_l[b]
                P = []
                for k in range(K):
                    P.append(get_exp_theta_k_mul_x_i(theta[k], data_i))
                prt_label = P.index(max(P))
                for label_name, label_num in dict_l.iteritems():
                    if label_num == prt_label:
                        prt_label_name = label_name
                f.write('%s\n' % (prt_label_name))
                if label != prt_label:
                    err_prt_cnt += 1
            else:
                f.write('\n')
    return err_prt_cnt / float(total_cnt)

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
if featr_flag == 1:
    read_feature_1(train_file)
elif featr_flag == 2:
    read_feature_2(train_file)

# initialize parameters
data_train = get_data(train_file)
data_vldat = get_data(vldat_file)
data_test = get_data(test_file)
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
            sgd(get_data_rep_1(data_train[idx][0]), dict_l[data_train[idx][1]])
        elif featr_flag == 2:
            pass
    avr_llh_train = get_cur_llh(data_train)
    print_llh(epc, avr_llh_train, metrc_out, 'train')
    avr_llh_vldat = get_cur_llh(data_vldat)
    print_llh(epc, avr_llh_vldat, metrc_out, 'validation')

train_err = get_prt_err(train_file, train_out)
print_err(train_err, 'train')
test_err = get_prt_err(test_file, test_out)
print_err(test_err, 'test')