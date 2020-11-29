# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import os
import scipy.io
import yaml
import torch
from torchvision import datasets, transforms
from model import ft_net
from model import load_network, load_whole_network
import numpy as np
from datasets import Channel_Dataset

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='duke', type=str, help='./test_data')
parser.add_argument('--name', default='', type=str, help='save model path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--net_loss_model', default=0, type=int, help='net_loss_model')
parser.add_argument('--domain_num', default=5, type=int, help='domain_num')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')

opt = parser.parse_args()
print('opt = %s' % opt)
print('opt.which_epoch = %s' % opt.which_epoch)
print('opt.test_dir = %s' % opt.test_dir)
print('opt.name = %s' % opt.name)
print('opt.batchsize = %s' % opt.batchsize)
###load config###
# load the training config
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)

name = opt.name
data_dir = os.path.join('data', opt.test_dir, 'pytorch')
print('data_dir = %s' % data_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#

data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset_list = ['gallery', 'query']
image_datasets = {x: Channel_Dataset(os.path.join(data_dir, x), data_transforms, domain_num=opt.domain_num, train=False) for
                  x in dataset_list}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in dataset_list}
class_names = image_datasets[dataset_list[1]].classes
use_gpu = torch.cuda.is_available()


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    cnt = 0
    for data in dataloaders:
        img_all = data
        img_original = img_all[0]
        n, c, h, w = img_original.size()
        cnt += n
        ff = torch.FloatTensor().cuda()
        # concatenate all feature embeddings of different domains into a whole representation in test stage
        for d in range(opt.domain_num):
            img = img_all[d]
            f = torch.FloatTensor(n, 512).zero_().cuda()
            for i in range(2):
                if i == 1:
                    img = fliplr(img)
                input_img = img.cuda()
                outputs = model(input_img)[1]
                outputs = outputs[d * n: (d + 1) * n]
                f = f + outputs
                # norm feature
                fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
                f = f.div(fnorm.expand_as(f))
            ff = torch.cat((ff, f), 1)
        ff = ff.detach().cpu().float()
        features = torch.cat((features, ff), 0)
    return features

def extract_feature_avg(model, dataloaders):
    features = torch.FloatTensor()
    cnt = 0
    for data in dataloaders:
        img_all = data
        img_original = img_all[0]
        n, c, h, w = img_original.size()
        cnt += n
        ff = torch.FloatTensor().cuda()
        # concatenate all feature embeddings of different domains into a whole representation in test stage
        for d in range(opt.domain_num):
            img = img_all[d]
            f = torch.FloatTensor(n, 512).zero_().cuda()
            for i in range(2):
                if i == 1:
                    img = fliplr(img)
                input_img = img.cuda()
                outputs = model(input_img)[1]
                outputs = outputs[d * n: (d + 1) * n]
                f = f + outputs
                # norm feature
                fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
                f = f.div(fnorm.expand_as(f))
            ff = torch.cat((ff, f), 1)
        ff = ff.view(n, -1, 512).mean(1)
        ff = ff.detach().cpu().float()
        features = torch.cat((features, ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        if 'msmt' in opt.test_dir:
            camera = filename[9:11]
        else:
            camera = filename.split('c')[1][0]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels


dataset_path = []
for i in range(len(dataset_list)):
    dataset_path.append(image_datasets[dataset_list[i]].imgs)

dataset_cam = []
dataset_label = []
for i in range(len(dataset_list)):
    cam, label = get_id(dataset_path[i])
    dataset_cam.append(cam)
    dataset_label.append(label)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
class_num = len(os.listdir(os.path.join(data_dir, 'train_all_new')))
model = ft_net(class_num, domain=opt.domain_num)
if use_gpu:
    model.cuda()
if 'best' in opt.which_epoch or 'last' in opt.which_epoch:
    model = load_whole_network(model, name, opt.which_epoch + '_' + str(opt.net_loss_model))
else:
    model = load_whole_network(model, name, opt.which_epoch)
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
dataset_feature = []
with torch.no_grad():
    for i in range(len(dataset_list)):
        dataset_feature.append(extract_feature(model, dataloaders[dataset_list[i]]))

result = {'gallery_f': dataset_feature[0].numpy(), 'gallery_label': dataset_label[0], 'gallery_cam': dataset_cam[0],
          'query_f': dataset_feature[1].numpy(), 'query_label': dataset_label[1], 'query_cam': dataset_cam[1]}
scipy.io.savemat('pytorch_result.mat', result)
