import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from pdb import set_trace as pause

from torchvision.ops.boxes import box_iou
from torchvision.ops import nms


def SUP_BOX(boxes, cls_prob, im_labels):
    im_labels    = im_labels.long()

    cls_prob     = cls_prob.clone().detach()
    
    # if cls_prob have the background dimenssion, we cut it out
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    
    #  avoiding NaNs.
    eps = 1e-9
    cls_prob = cls_prob.clamp(eps, 1 - eps)

    num_images, num_classes = im_labels.shape
    #
    max_values, max_indexes = cls_prob.max(dim=0)

    gt_boxes   = boxes[max_indexes, :][im_labels[0]==1,:] 
    gt_scores  = max_values[im_labels[0]==1].view(-1,1)

    overlaps = box_iou(boxes, gt_boxes)

    max_overlaps, gt_assignment = overlaps.max(dim=1)

    cls_loss_weights = gt_scores[gt_assignment, 0]


    in_boxes = torch.where(max_overlaps > 0.0)[0]
    gt_boxes = boxes[in_boxes]
    gt_scores = cls_loss_weights[in_boxes]


    keep = nms(gt_boxes, gt_scores, 0.3)

    gt_boxes = gt_boxes[keep]
    gt_scores = gt_scores[keep]

    return {'boxes': gt_boxes, 'gt_scores':gt_scores}
    