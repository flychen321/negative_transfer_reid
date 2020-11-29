# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import os
import numpy as np
import yaml
from model import ft_net
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset
from model import save_network, save_whole_network, load_network, load_whole_network
from losses import SoftLabelLoss, ContrastiveLoss_diff, ContrastiveLoss_same, ContrastiveLoss_orth
from datasets import Channel_Dataset, RandomErasing

version = torch.__version__

######################################################################
# Options
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_dir', default='duke', type=str, help='training dir path')
parser.add_argument('--batchsize', default=4, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--net_loss_model', default=0, type=int, help='net_loss_model')
parser.add_argument('--domain_num', default=5, type=int, help='domain_num, in [2,6]')
parser.add_argument('--class_base', default=702, type=int, help='class_base, in [751, 702, 767]')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')

opt = parser.parse_args()
print('opt = %s' % opt)
print('net_loss_model = %d' % opt.net_loss_model)
print('save_model_name = %s' % opt.save_model_name)
print('domain_num = %s' % opt.domain_num)
if opt.domain_num > 6 or opt.domain_num < 2:
    print('domain_num = %s' % opt.domain_num)
    exit()
data_dir = os.path.join('data', opt.data_dir, 'pytorch')
name = opt.name
opt.class_base = len(os.listdir(os.path.join(data_dir, 'train_all_new')))
print('opt.class_base = %d' % opt.class_base)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
######################################################################
# Load Data
# --------------------------------------------------------------------
#
transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
}

image_datasets = {}
image_datasets['train'] = Channel_Dataset(os.path.join(data_dir, 'train_all_new'),
                                      data_transforms['train'], domain_num=opt.domain_num)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8) for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
use_gpu = torch.cuda.is_available()


#######################################################################################################################
# convert 'one-hot' label to soft-label
# Parameters
#         ----------
#         labels : determine which bit correspondings to '1' in 'one-hot' label
#         w_main : determine the value of main ingredient bit
#
# ---------------------------------------------------------------------------------------------------------------------
def get_soft_label_6domain(labels, w_main=0.7, label_sum=1.0):
    class_base = opt.class_base
    w_reg = (label_sum - w_main) / class_base
    if w_reg < 0:
        print('w_main=%s' % w_main)
        exit()
    soft_label = np.zeros((len(labels), class_base))
    soft_label.fill(w_reg)
    for i in np.arange(len(labels)):
        for j in np.arange(class_base):
            soft_label[i][j] = w_reg
        soft_label[i][labels[i]] = w_main + w_reg
    return torch.Tensor(soft_label)


######################################################################
# Training the model
# --------------------------------------------------------------------
def train(model, criterion_identify, criterion_contrastive_same, criterion_contrastive_diff, criterion_orthogonal,
              optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    cnt = 0


    # for ResNet-50
    # Hyper-parameters about Two-level Classification Label Assignment Strategy
    w_main_c = 1.0
    w_main_mix_c = 0.0
    # Weights of Two-level Classification Loss Functions
    r_id_classify = 0.5
    r_id_mix = 0.3
    # Weights of the Structural Consistencies
    r_d = 0.02
    r_t = 0.01
    r_o = 0.05

    print('r_id_classify = %.3f  r_id_mix = %.3f  r_d = %.3f  r_t = %.3f  r_o = %.3f' % (
        r_id_classify, r_id_mix, r_d, r_t, r_o))
    print('w_main_c = %.3f    w_main_mix_c = %.3f' % (w_main_c, w_main_mix_c))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            scheduler.step()
            model.train(True)  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs1_0, inputs1_1, inputs2_0, inputs2_1, inputs3_0, inputs3_1, \
                label1_0, label1_1, label2_0, label2_1, label3_0, label3_1 = data
                inputs = torch.cat(
                    (inputs1_0, inputs1_1, inputs2_0, inputs2_1, inputs3_0, inputs3_1),
                    0)
                id_labels = torch.cat(
                    (label1_0, label1_1, label2_0, label2_1, label3_0, label3_1), 0)
                # Two-Level Labels
                id_labels_soft = get_soft_label_6domain(id_labels % opt.class_base, w_main=w_main_c)
                id_labels_mix_soft = get_soft_label_6domain(id_labels % opt.class_base, w_main=w_main_mix_c)
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size // 6 < opt.batchsize:  # next epoch
                    print('continue')
                    continue

                if use_gpu:
                    inputs = inputs.cuda()
                    id_labels_soft = id_labels_soft.cuda()
                    id_labels_mix_soft = id_labels_mix_soft.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs, features = model(inputs)
                mask = torch.FloatTensor(outputs.shape[0]).zero_().cuda()
                for d in range(opt.domain_num):
                    mask[now_batch_size * d: now_batch_size * (d + 1)] \
                        = (id_labels / opt.class_base >= d) * (id_labels / opt.class_base < (d + 1))
                _, id_preds = torch.max(outputs.detach(), 1)
                id_labels_soft_all = id_labels_soft
                id_labels_mix_soft_all = id_labels_mix_soft
                for d in range(opt.domain_num - 1):
                    id_labels_soft_all = torch.cat((id_labels_soft_all, id_labels_soft), 0)
                    id_labels_mix_soft_all = torch.cat((id_labels_mix_soft_all, id_labels_mix_soft), 0)
                # Intra-domain Category Classification Loss
                loss_id_classify = criterion_identify(outputs, id_labels_soft_all, mask)
                # Domain Classification Loss
                loss_id_mix_classify = criterion_identify(outputs, id_labels_mix_soft_all, (1 - mask))
                loss_id = r_id_classify * loss_id_classify + r_id_mix * loss_id_mix_classify
                features = features[: now_batch_size]
                part_len = features.shape[0] // 6

                feature1_0 = features[part_len * 0:part_len * 1]
                feature1_1 = features[part_len * 1:part_len * 2]
                feature2_0 = features[part_len * 2:part_len * 3]
                feature2_1 = features[part_len * 3:part_len * 4]
                feature3_0 = features[part_len * 4:part_len * 5]
                feature3_1 = features[part_len * 5:part_len * 6]

                loss_con_dist = 0
                loss_con_topol = 0
                loss_con_orth = 0

                # Inter domain
                # Inter-Domain Distance Consistency Loss
                loss_con_dist += criterion_contrastive_same((feature1_0 - feature1_1),
                                                             (feature2_0 - feature2_1))
                loss_con_dist += criterion_contrastive_same((feature1_0 - feature1_1),
                                                             (feature3_0 - feature3_1))
                loss_con_dist += criterion_contrastive_same((feature2_0 - feature2_1),
                                                             (feature3_0 - feature3_1))

                # Intra domain
                # Cross-domain Topology Consistency Loss
                loss_con_topol += criterion_contrastive_same((feature1_0 - feature2_0),
                                                             (feature1_1 - feature2_1))
                loss_con_topol += criterion_contrastive_same((feature1_0 - feature3_0),
                                                             (feature1_1 - feature3_1))
                loss_con_topol += criterion_contrastive_same((feature2_0 - feature3_0),
                                                             (feature2_1 - feature3_1))

                # Cross-domain Orthogonality Loss
                loss_con_orth += criterion_orthogonal(feature1_0, feature1_1)
                loss_con_orth += criterion_orthogonal(feature2_0, feature2_1)
                loss_con_orth += criterion_orthogonal(feature3_0, feature3_1)

                loss_con = r_d * loss_con_dist + r_t * loss_con_topol + r_o * loss_con_orth

                # calculate the total loss
                loss = loss_id + loss_con
                if cnt % 200 == 0:
                    print('cnt = %5d   loss = %.4f  loss_id = %.4f  loss_con = %.4f' % (
                        cnt, loss.cpu().detach().numpy(), loss_id.cpu().detach().numpy(),
                        loss_con.cpu().detach().numpy()))
                    print('loss_con_dist  = %.4f  loss_con_topol  = %.4f  loss_con_orth  = %.4f' % (
                            loss_con_dist.cpu().detach().numpy(),
                            loss_con_topol.cpu().detach().numpy(),
                            loss_con_orth.cpu().detach().numpy()))
                    print('loss_id_classify = %.4f  loss_id_mix_classify = %.4f' % (
                        loss_id_classify.cpu().detach().numpy(), loss_id_mix_classify.cpu().detach().numpy()))

                cnt += 1
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item()  # * opt.batchsize
                running_corrects += float(torch.sum((id_preds == id_labels_soft_all.argmax(1).detach() * mask.long())))
            datasize = dataset_sizes[phase] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects / (datasize * 6)

            print('{} Loss: {:.4f}  Acc_id: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_whole_network(model, name, 'best' + '_' + str(opt.net_loss_model))

            save_whole_network(model, name, epoch)
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    save_whole_network(model, name, 'last' + '_' + str(opt.net_loss_model))


######################################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
# -------------------------------------------------------------------------------------
dir_name = os.path.join('./model', name)
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

class_num = image_datasets['train'].class_num
print('class_num = %d' % class_num)
model = ft_net(class_num, domain=opt.domain_num)
if use_gpu:
    model.cuda()

# Initialize loss functions
criterion_identify = SoftLabelLoss()
criterion_orthogonal = ContrastiveLoss_orth()
criterion_reconstruct = nn.MSELoss()
margin = 2.0
print('margin = %s' % margin)
criterion_contrastive_diff = ContrastiveLoss_diff(margin)
criterion_contrastive_same = ContrastiveLoss_same()

# Set different learning rates for newly added parameters and based layers of ResNet-50
new_id = list(map(id, model.classifier0.parameters())) \
         + list(map(id, model.classifier1.parameters())) \
         + list(map(id, model.classifier2.parameters())) \
         + list(map(id, model.classifier3.parameters())) \
         + list(map(id, model.classifier4.parameters())) \
         + list(map(id, model.classifier5.parameters())) \
         + list(map(id, model.fc.parameters())) \
         + list(map(id, model.model.fc.parameters()))
classifier_params = filter(lambda p: id(p) in new_id, model.parameters())
base_params = filter(lambda p: id(p) not in new_id, model.parameters())


optimizer_ft = torch.optim.SGD([
    {'params': classifier_params, 'lr': 0.1 * opt.lr},
    {'params': base_params, 'lr': 0.1 * opt.lr},
], weight_decay=5e-4, momentum=0.9, nesterov=True)

epoch = 30
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[11, 22, 26], gamma=0.1)
# train the model
train(model, criterion_identify, criterion_contrastive_same, criterion_contrastive_diff, criterion_orthogonal,
          optimizer_ft, exp_lr_scheduler, num_epochs=epoch)
