import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def calc_segmentation_consistency(output, reference,divergence_types=['kl','contour'],
                                    divergence_weights=[1.0,0.5],scales=[0],
                                    mask=None):
    """
    measuring the difference between two predictions (network logits before softmax)
    Args:
        output (torch tensor 4d): network predicts: NCHW (after perturbation)
        reference (torch tensor 4d): network references: NCHW (before perturbation)
        divergence_types (list, string): specify loss types. Defaults to ['kl','contour'].
        divergence_weights (list, float): specify coefficients for each loss above. Defaults to [1.0,0.5].
        scales (list of int): specify a list of downsampling rates so that losses will be calculated on different scales. Defaults to [0].
        mask ([tensor], 0-1 onehotmap): [N*1*H*W]. No losses on the elements with mask=0. Defaults to None.
    Raises:
        NotImplementedError: when loss name is not in ['kl','mse','contour']
    Returns:
        loss (tensor float): 
    """
    dist = 0.
    num_classes = reference.size(1)
    reference = reference.detach()
    if mask is None:
        ## apply masks so that only gradients on certain regions will be backpropagated. 
        mask = torch.ones_like(output).float().to(reference.device)

    for scale in scales:
        if scale>0:
            output_reference = torch.nn.AvgPool2d(2 ** scale)(reference)
            output_new = torch.nn.AvgPool2d(2 ** scale)(output)
        else:
            output_reference = reference
            output_new = output
        for divergence_type, d_weight in zip(divergence_types, divergence_weights):
            loss = 0.
            if divergence_type=='kl':
                '''
                standard kl loss 
                '''
                loss = kl_divergence(pred=output_new,reference=output_reference.detach(),mask=mask)
            elif divergence_type =='mse':
                target_pred = torch.softmax(output_reference, dim=1)
                input_pred = torch.softmax(output_new, dim=1)
                loss = torch.nn.MSELoss(reduction='sum')(target = target_pred*mask, input = input_pred*mask)
                loss = loss/torch.sum(mask[:,0])
            elif divergence_type == 'contour':  ## contour-based loss
                target_pred = torch.softmax(output_reference, dim=1)
                input_pred = torch.softmax(output_new, dim=1)
                cnt = 0
                for i in range(1,num_classes):
                    cnt +=1
                    loss += contour_loss(input=input_pred[:,[i],], target=(target_pred[:,[i]]).detach(), ignore_background=False,mask=mask,
                                                                    one_hot_target=False)
                                        
            else:
                raise NotImplementedError
          
            dist += 2 ** scale*(d_weight * loss)
    return dist / (1.0  * len(scales))




def contour_loss(input, target, size_average=True, use_gpu=True,ignore_background=True,one_hot_target=True,mask=None):
    '''
    calc the contour loss across object boundaries (WITHOUT background class)
    :param input: NDArray. N*num_classes*H*W : pixelwise probs. for each class e.g. the softmax output from a neural network
    :param target: ground truth labels (NHW) or one-hot ground truth maps N*C*H*W
    :param size_average: batch mean
    :param use_gpu:boolean. default: True, use GPU.
    :param ignore_background:boolean, ignore the background class. default: True
    :param one_hot_target: boolean. if true, will first convert the target from NHW to NCHW. Default: True.
    :return:
    '''
    n,num_classes,h,w = input.size(0),input.size(1),input.size(2),input.size(3)
    if one_hot_target:
        onehot_mapper = One_Hot(depth=num_classes, use_gpu=use_gpu)
        target = target.long()
        onehot_target = onehot_mapper(target).contiguous().view(input.size(0), num_classes, input.size(2), input.size(3))
    else:
        onehot_target=target
    assert onehot_target.size() == input.size(), 'pred size: {} must match target size: {}'.format(str(input.size()),str(onehot_target.size()))

    if mask is None:
        ## apply masks so that only gradients on certain regions will be backpropagated. 
        mask = torch.ones_like(input).long().to(input.device)
        mask.requires_grad = False
    else:
        pass
        # print ('mask applied')
    
    
    if ignore_background:
        object_classes = num_classes - 1
        target_object_maps = onehot_target[:, 1:].float()
        input = input[:, 1:]
    else:
        target_object_maps=onehot_target
        object_classes  = num_classes

    x_filter = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]]).reshape(1, 1, 3, 3)

    x_filter = np.repeat(x_filter, axis=1, repeats=object_classes)
    x_filter = np.repeat(x_filter, axis=0, repeats=object_classes)
    conv_x = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                       dilation=1, bias=False)

    conv_x.weight = nn.Parameter(torch.from_numpy(x_filter).float())

    y_filter = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]).reshape(1, 1, 3, 3)
    y_filter = np.repeat(y_filter, axis=1, repeats=object_classes)
    y_filter = np.repeat(y_filter, axis=0, repeats=object_classes)
    conv_y = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                      bias=False)
    conv_y.weight = nn.Parameter(torch.from_numpy(y_filter).float())

    if use_gpu:
        conv_y = conv_y.cuda()
        conv_x = conv_x.cuda()
    for param in conv_y.parameters():
        param.requires_grad = False
    for param in conv_x.parameters():
        param.requires_grad = False

    g_x_pred = conv_x(input)*mask[:,:object_classes]
    g_y_pred = conv_y(input)*mask[:,:object_classes]
    g_y_truth = conv_y(target_object_maps)*mask[:,:object_classes]
    g_x_truth = conv_x(target_object_maps)*mask[:,:object_classes]

    ## mse loss
    loss =torch.nn.MSELoss(reduction='sum')(input=g_x_pred,target=g_x_truth) +torch.nn.MSELoss(reduction='sum')(input=g_y_pred,target=g_y_truth)
    loss/= torch.sum(mask[:,0,:,:])
    return loss


def kl_divergence(reference, pred,mask=None):
    '''
    calc the kl div distance between two outputs p and q from a network/model: p(y1|x1).p(y2|x2).
    :param reference p: directly output from network using origin input without softmax
    :param output q: approximate output: directly output from network using perturbed input without softmax
    :return: kl divergence: DKL(P||Q) = mean(\sum_1 \to C (p^c log (p^c|q^c)))
    '''
    p=reference
    q=pred
    p_logit = F.softmax(p, dim=1)
    if mask is None:
        mask = torch.ones_like(p_logit, device =p_logit.device)
        mask.requires_grad=False
    cls_plogp = mask*(p_logit * F.log_softmax(p, dim=1))
    cls_plogq = mask*(p_logit * F.log_softmax(q, dim=1))
    plogp = torch.sum(cls_plogp,dim=1,keepdim=True)
    plogq = torch.sum(cls_plogq,dim=1,keepdim=True)

    kl_loss = torch.sum(plogp - plogq)
    kl_loss/=torch.sum(mask[:,0,:,:])
    return kl_loss



