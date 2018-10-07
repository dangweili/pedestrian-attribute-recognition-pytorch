import sys
import os
import numpy as np
import random
import math

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
import cPickle as pickle
import time
import argparse
from PIL import Image, ImageFont, ImageDraw

from baseline.model.DeepMAR import DeepMAR_ResNet50
from baseline.utils.utils import str2bool
from baseline.utils.utils import save_ckpt, load_ckpt
from baseline.utils.utils import load_state_dict 
from baseline.utils.utils import set_devices
from baseline.utils.utils import set_seed


class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('--set_seed', type=str2bool, default=False)
        # model
        parser.add_argument('--resize', type=eval, default=(224, 224))
        parser.add_argument('--last_conv_stride', type=int, default=2, choices=[1,2])
        # demo image
        parser.add_argument('--demo_image', type=str, default='./dataset/demo/demo_image.png')
        ## dataset parameter
        parser.add_argument('--dataset', type=str, default='peta',
                choices=['peta','rap', 'pa100k'])
        # utils
        parser.add_argument('--load_model_weight', type=str2bool, default=True)
        parser.add_argument('--model_weight_file', type=str, default='./exp/deepmar_resnet50/peta/partition0/run1/model/ckpt_epoch150.pth')
        args = parser.parse_args()
        
        # gpu ids
        self.sys_device_ids = args.sys_device_ids

        # random
        self.set_seed = args.set_seed
        if self.set_seed:
            self.rand_seed = 0
        else: 
            self.rand_seed = None
        self.resize = args.resize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # utils
        self.load_model_weight = args.load_model_weight
        self.model_weight_file = args.model_weight_file
        if self.load_model_weight:
            if self.model_weight_file == '':
                print 'Please input the model_weight_file if you want to load model weight'
                raise ValueError
        # dataset 
        datasets = dict()
        datasets['peta'] = './dataset/peta/peta_dataset.pkl'
        datasets['rap'] = './dataset/rap/rap_dataset.pkl'
        datasets['pa100k'] = './dataset/pa100k/pa100k_dataset.pkl'

        if args.dataset in datasets:
            dataset = pickle.load(open(datasets[args.dataset]))
        else:
            print '%s does not exist.'%(args.dataset)
            raise ValueError
        self.att_list = [dataset['att_name'][i] for i in dataset['selected_attribute']]
        
        # demo image
        self.demo_image = args.demo_image

        # model
        model_kwargs = dict()
        model_kwargs['num_att'] = len(self.att_list)
        model_kwargs['last_conv_stride'] = args.last_conv_stride
        self.model_kwargs = model_kwargs

### main function ###
cfg = Config()

# dump the configuration to log.
import pprint
print('-' * 60)
print('cfg.__dict__')
pprint.pprint(cfg.__dict__)
print('-' * 60)

# set the random seed
if cfg.set_seed:
    set_seed( cfg.rand_seed )
# init the gpu ids
set_devices(cfg.sys_device_ids)

# dataset 
normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
test_transform = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.ToTensor(),
        normalize,])

### Att model ###
model = DeepMAR_ResNet50(**cfg.model_kwargs)

# load model weight if necessary
if cfg.load_model_weight:
    map_location = (lambda storage, loc:storage)
    ckpt = torch.load(cfg.model_weight_file, map_location=map_location)
    model.load_state_dict(ckpt['state_dicts'][0])

model.cuda()
model.eval()

# load one image 
img = Image.open(cfg.demo_image)
img_trans = test_transform( img ) 
img_trans = torch.unsqueeze(img_trans, dim=0)
img_var = Variable(img_trans).cuda()
score = model(img_var).data.cpu().numpy()

# show the score in command line
for idx in range(len(cfg.att_list)):
    if score[0, idx] >= 0:
        print '%s: %.2f'%(cfg.att_list[idx], score[0, idx])

# show the score in the image
img = img.resize(size=(256, 512), resample=Image.BILINEAR)
draw = ImageDraw.Draw(img)
positive_cnt = 0
for idx in range(len(cfg.att_list)):
    if score[0, idx] >= 0:
        txt = '%s: %.2f'%(cfg.att_list[idx], score[0, idx])
        draw.text((10, 10 + 10*positive_cnt), txt, (255, 0, 0))
        positive_cnt += 1
img.save('./dataset/demo/demo_image_result.png')
