import os
import torch
from torch.autograd import Variable
import numpy as np
import copy
import time
import sys

def extract_feat(feat_func, dataset, **kwargs):
    """
    extract feature for images
    """
    test_loader = torch.utils.data.DataLoader(
        dataset = dataset, batch_size = 32,
        num_workers = 2, pin_memory = True)
    # extract feature for all the images of test/val identities
    start_time = time.time()
    total_eps = len(test_loader)
    N = len(dataset.image)
    start = 0
    for ep, (imgs, labels) in enumerate(test_loader):
        imgs_var = Variable(imgs, volatile=True).cuda()
        feat_tmp = feat_func( imgs_var )
        batch_size = feat_tmp.shape[0]
        if ep == 0:
            feat = np.zeros((N, feat_tmp.size/batch_size))
        feat[start:start+batch_size, :] = feat_tmp.reshape((batch_size, -1))
        start += batch_size
    end_time = time.time() 
    print('{} batches done, total {:.2f}s'.format(total_eps, end_time-start_time))
    return feat 

# attribute recognition evaluation 
def attribute_evaluate(feat_func, dataset, **kwargs):
    print "extracting features for attribute recognition"
    pt_result = extract_feat(feat_func, dataset)
    # obain the attributes from the attribute dictionary
    print "computing attribute recognition result"
    N = pt_result.shape[0] 
    L = pt_result.shape[1]
    gt_result = np.zeros(pt_result.shape)
    # get the groundtruth attributes
    for idx, label in enumerate(dataset.label):
        gt_result[idx, :] = label
    pt_result[pt_result>=0] = 1
    pt_result[pt_result<0] = 0 
    return attribute_evaluate_lidw(gt_result, pt_result)

def attribute_evaluate_lidw(gt_result, pt_result):
    """
    Input: 
    gt_result, pt_result, N*L, with 0/1
    Output:
    result 
    a dictionary, including label-based and instance-based evaluation
    label-based: label_pos_acc, label_neg_acc, label_acc
    instance-based: instance_acc, instance_precision, instance_recall, instance_F1
    """
    # obtain the label-based and instance-based accuracy
    # compute the label-based accuracy
    if gt_result.shape != pt_result.shape:
        print 'Shape beteen groundtruth and predicted results are different'
    # compute the label-based accuracy
    result = {}
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)
    label_pos_acc = 1.0*pt_pos/gt_pos
    label_neg_acc = 1.0*pt_neg/gt_neg
    label_acc = (label_pos_acc + label_neg_acc)/2
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    floatersect_pos = np.sum((gt_result == 1).astype(float)*(pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1)+(pt_result == 1)).astype(float),axis=1)
    # avoid empty label in predicted results
    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    instance_acc = np.sum(floatersect_pos/union_pos)/cnt_eff
    instance_precision = np.sum(floatersect_pos/pt_pos)/cnt_eff
    instance_recall = np.sum(floatersect_pos/gt_pos)/cnt_eff
    floatance_F1 = 2*instance_precision*instance_recall/(instance_precision+instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1
    return result
