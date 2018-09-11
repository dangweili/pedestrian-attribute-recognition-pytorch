import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .resnet import resnet50


class DeepMAR_ResNet50(nn.Module):
    def __init__(
        self, 
        **kwargs
    ):
        super(DeepMAR_ResNet50, self).__init__()
        
        # init the necessary parameter for netwokr structure
        if kwargs.has_key('num_att'):
            self.num_att = kwargs['num_att'] 
        else:
            self.num_att = 35
        if kwargs.has_key('last_conv_stride'):
            self.last_conv_stride = kwargs['last_conv_stride']
        else:
            self.last_conv_stride = 2
        if kwargs.has_key('drop_pool5'):
            self.drop_pool5 = kwargs['drop_pool5']
        else:
            self.drop_pool5 = True 
        if kwargs.has_key('drop_pool5_rate'):
            self.drop_pool5_rate = kwargs['drop_pool5_rate']
        else:
            self.drop_pool5_rate = 0.5
        if kwargs.has_key('pretrained'):
            self.pretrained = kwargs['pretrained'] 
        else:
            self.pretrained = True

        self.base = resnet50(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)
        
        self.classifier = nn.Linear(2048, self.num_att)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.drop_pool5:
            x = F.dropout(x, p=self.drop_pool5_rate, training=self.training)
        x = self.classifier(x)
        return x

class DeepMAR_ResNet50_ExtractFeature(object):
    """
    A feature extraction function
    """
    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self, imgs):
        old_train_eval_model = self.model.training

        # set the model to be eval
        self.model.eval()

        # imgs should be Variable
        if not isinstance(imgs, Variable):
            print 'imgs should be type: Variable'
            raise ValueError
        score = self.model(imgs)
        score = score.data.cpu().numpy()

        self.model.train(old_train_eval_model)

        return score
