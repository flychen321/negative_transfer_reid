import os
import numpy as np

######################################################################
# This file will execute training, testing and evaluating automatically
# --------------------------------------------------------------------
if not os.path.exists('log'):
    os.mkdir('log')

datasets = ['duke']
# use ResNet-50 as backbone
domain_orders = [5]
i = 0
gpu_id = '0'
for dataset_one in datasets:
    for domain_order in domain_orders:
        print('i = %.3f' % i)
        log_name = 'log/' + 'log_' + str(i)
        print('log name = %s' % log_name)

        cmd = 'python train.py  --gpu  ' + gpu_id + '  --data_dir ' + dataset_one + ' --batchsize  4  --net_loss_model ' + str(
            i) + ' --domain_num ' + str(domain_order) + ' >> ' + log_name
        print('cmd = %s' % cmd)
        os.system(cmd)

        os.system(
            'python test.py   --gpu  ' + gpu_id + ' --test_dir ' + dataset_one + ' --which_epoch best --net_loss_model ' + str(
                i) + ' --domain_num ' + str(domain_order) + ' >>  ' + log_name)
        os.system('python evaluate.py   --gpu  ' + gpu_id + ' >> ' + log_name)

        os.system(
            'python test.py   --gpu  ' + gpu_id + ' --test_dir ' + dataset_one + ' --which_epoch last --net_loss_model ' + str(
                i) + ' --domain_num ' + str(domain_order) + ' >>  ' + log_name)
        os.system('python evaluate.py   --gpu  ' + gpu_id + ' >> ' + log_name)

        i += 1
