import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os
from collections import OrderedDict


######################################################################
# Load parameters of model
# ---------------------------
def load_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'last')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load easy pretrained model: %s' % save_path)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Load model
# ---------------------------
def load_whole_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'whole_last')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load whole pretrained model: %s' % save_path)
    net_original = torch.load(save_path)
    pretrained_dict = net_original.state_dict()
    model_dict = network.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


######################################################################
# Save parameters of model
# ---------------------------
def save_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)


######################################################################
# Save model
# ---------------------------
def save_whole_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network, save_path)


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_out')
        init.constant_(m.bias.detach(), 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.detach(), 1.0, 0.02)
        init.constant_(m.bias.detach(), 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.detach(), std=0.001)
        init.constant_(m.bias.detach(), 0.0)


######################################################################
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
# --------------------------------------------------------------------
class Fc_ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(Fc_ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f

class Fc(nn.Module):
    def __init__(self, input_dim, relu=False, output_dim=512):
        super(Fc, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, output_dim)]
        add_block += [nn.BatchNorm1d(output_dim)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.add_block = add_block

    def forward(self, x):
        x = self.add_block(x)
        return x

class ClassBlock(nn.Module):
    def __init__(self, input_dim=512, class_num=751, dropout=0.5):
        super(ClassBlock, self).__init__()
        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        f = x
        # L2 normalize
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f


# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num, domain=3):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.fc = Fc(2048, relu=False)
        self.classifier0 = ClassBlock(class_num=class_num)
        self.classifier1 = ClassBlock(class_num=class_num)
        self.classifier2 = ClassBlock(class_num=class_num)
        self.classifier3 = ClassBlock(class_num=class_num)
        self.classifier4 = ClassBlock(class_num=class_num)
        self.classifier5 = ClassBlock(class_num=class_num)
        self.classifier = [self.classifier0, self.classifier1, self.classifier2, self.classifier3, self.classifier4,
                           self.classifier5]
        self.domain = domain

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        outputs = torch.FloatTensor().cuda()
        features = torch.FloatTensor().cuda()
        for d in range(self.domain):
            output, feature = self.classifier[d](x)
            outputs = torch.cat((outputs, output), 0)
            features = torch.cat((features, feature), 0)
        return outputs, features


