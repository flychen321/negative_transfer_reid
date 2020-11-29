# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import datasets
import random
import math
#############################################################################################################
# Channel_Dataset: It is used to get samples which include the original images and the augmented counterparts
# Parameters
#         ----------
#         domain_num: The number of augmented samples for each original one
# -----------------------------------------------------------------------------------------------------------
class Channel_Dataset(datasets.ImageFolder):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    def __init__(self, root, transform, domain_num=6, train=True):
        super(Channel_Dataset, self).__init__(root, transform)
        self.domain_num = domain_num
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root
    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        img1, label1 = self.data[index], self.labels[index].item()
        index2 = index
        while self.train and index2 == index:
            index2 = np.random.choice(self.label_to_indices[label1])
        img2, label2 = self.data[index2], self.labels[index2].item()

        label3 = np.random.choice(list(self.labels_set - set([label1])))
        index3 = np.random.choice(self.label_to_indices[label3])
        img3, label3 = self.data[index3], self.labels[index3].item()

        img1 = default_loader(img1)
        img2 = default_loader(img2)
        img3 = default_loader(img3)

        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        img_original = []
        img_original.append(img1)
        img_original.append(img2)
        img_original.append(img3)
        label_original = []
        label_original.append(label1)
        label_original.append(label2)
        label_original.append(label3)
        img_all = []
        label_all = []
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            label = label_original[i]
            img_sub = []
            label_sub = []
            for j in range(self.domain_num):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                      img_3channel[index_channel[j][2]]))
                img_sub.append(img)
                label_sub.append(self.class_num * j + int(label))
            img_all.append(img_sub)
            label_all.append(label_sub)

        if self.transform is not None:
            for i in range(len(img_all)):
                for j in range(self.domain_num):
                    img_all[i][j] = self.transform(img_all[i][j])

        # The below operation can produce data with more diversity
        indices = np.random.permutation(self.domain_num)
        if self.train:
            return img_all[0][indices[0]], img_all[0][indices[1]], \
                   img_all[1][indices[0]], img_all[1][indices[1]], \
                   img_all[2][indices[0]], img_all[2][indices[1]], \
                   label_all[0][indices[0]], label_all[0][indices[1]], \
                   label_all[1][indices[0]], label_all[1][indices[1]], \
                   label_all[2][indices[0]], label_all[2][indices[1]]
        else:
            return img_all[0]

    def __len__(self):
        return len(self.imgs)
#############################################################################################################
# RandomErasing: Executing random erasing on input data
# Parameters
#         ----------
#         probability: The probability that the Random Erasing operation will be performed
#         sl: Minimum proportion of erased area against input image
#         sh: Maximum proportion of erased area against input image
#         r1: Minimum aspect ratio of erased area
#         mean: Erasing value
# -----------------------------------------------------------------------------------------------------------
class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
