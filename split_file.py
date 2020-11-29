import numpy as np
import os
import shutil

path_src = 'data/ins_dataset/VIPeR'
path_dst = 'data/viper'
path_src2 = 'data/ins_dataset/PRID2011/single_shot'
path_dst2 = 'data/prid'
if os.path.exists(path_dst):
    shutil.rmtree(path_dst)
os.makedirs(path_dst)

path_train = os.path.join(path_dst, 'bounding_box_train')
path_qurey = os.path.join(path_dst, 'query')
path_test = os.path.join(path_dst, 'bounding_box_test')
os.makedirs(path_train)
os.makedirs(path_qurey)
os.makedirs(path_test)

files_a = os.listdir(os.path.join(path_src, 'cam_a'))
files_b = os.listdir(os.path.join(path_src, 'cam_b'))

index_file = open(os.path.join(path_src, 'split.txt'), 'r')
index_info = index_file.readlines()
index_info.append('end')
order = 1

dir_cnt = 0
k = 0
for i in np.arange(len(index_info)):
    if 'Experiment Number' in index_info[i] and order == int(index_info[i].split()[-1]):
        i += 2
        k = i
        while i < len(index_info):
            if 'cam' in index_info[i]:
                name_a = index_info[i].split()[-2].split('/')[-1]
                new_name_a = '0' + index_info[i].split()[-2].split('/')[-1].split('_')[0] + '_c1.bmp'
                name_b = index_info[i].split()[-1].split('/')[-1]
                new_name_b = '0' + index_info[i].split()[-1].split('/')[-1].split('_')[0] + '_c2.bmp'
                shutil.copy(os.path.join(path_src, 'cam_a', name_a), os.path.join(path_train, new_name_a))
                shutil.copy(os.path.join(path_src, 'cam_b', name_b), os.path.join(path_train, new_name_b))
                files_a.remove(name_a)
                files_b.remove(name_b)
                i += 1
            else:
                break
        break

for file in files_a:
    new_name_a = '0' + file.split('_')[0] + '_c1.bmp'
    shutil.copy(os.path.join(path_src, 'cam_a', file), os.path.join(path_qurey, new_name_a))

for file in files_b:
    new_name_b = '0' + file.split('_')[0] + '_c2.bmp'
    shutil.copy(os.path.join(path_src, 'cam_b', file), os.path.join(path_test, new_name_b))

print('dir_cnt = %d' % dir_cnt)
