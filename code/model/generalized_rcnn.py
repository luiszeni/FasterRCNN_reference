# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

'''
    This code was copied/adapted from https://github.com/pytorch/vision/tree/master/references/detection
    
'''
import torch
import warnings
import cv2
from torch import nn
from torch import Tensor
from typing import Union
from collections import OrderedDict
import numpy as np
from torch.jit.annotations import Tuple, List, Dict, Optional
from pdb import set_trace as pause
#

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        # proposals, proposal_losses = self.rpn(images, features, targets)
        proposals = [t['proposals'] for t in targets]
        

        DEDUP_BOXES = 1/8
        for prop in proposals:
            v = np.array([1e3, 1e6, 1e9, 1e12])
            hashes = np.round(prop.cpu().numpy() * DEDUP_BOXES).dot(v)
            
            _, index, inv_index = np.unique(hashes, return_index=True, return_inverse=True)

            sp = prop.shape
            prop  = prop[index, :]
            # print('removed ', sp[0] - prop.shape[0])


        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)


        display_input_data = False
        if display_input_data: 
            for i, img in enumerate(images.tensors):

                img = self.transform.unnormalize(img)

                img_cv = img.permute(1,2,0).cpu().numpy()
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
                box_scores = detections[i]
                proposal    = proposals[i]

                score, index = box_scores[:,1:].max(dim=0)
                # score, index = box_scores[:].max(dim=0)
                
                class_ids = torch.where(score>0.1)[0]
                index = index[score>0.1]

                detected = proposal[index]

                for box in detected:
                    box =  box.to(torch.int)
                    p1 = tuple(box[:2].cpu().tolist())
                    p2 = tuple(box[2:].cpu().tolist())

                    cv2.rectangle(img_cv, p1, p2,  (0,0,255), 3)



                img_cv = cv2.resize(img_cv, None, fx=0.5, fy=0.5)
                cv2.imshow("image", img_cv)
                
                
                if cv2.waitKey(0) == ord('q'):
                    exit()

        #         # #display annotations
        #         # target = targets[i]
        #         # print("image_id", target['image_id'])
        #         # for box, label, keypoints, in zip(target['boxes'], target['labels'], target['keypoints']):

        #         #     box =  box.to(torch.int)
        #         #     p1 = tuple(box[:2].cpu().tolist())
        #         #     p2 = tuple(box[2:].cpu().tolist())

        #         #     cv2.rectangle(img_cv, p1, p2,  (0,0,255), 3)

        #         #     keypoints = keypoints[:,:2].to(torch.int)
        #         #     for i, pt in enumerate(keypoints):

        #         #         color = (0,0,255)
        #         #         if i in [0, 5, 6, 13]:
        #         #             color = (0,255,0)
        #         #         elif i in [1,2,3,4]:
        #         #             color = (255,0,0)


        #         #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         #         font_scale = 1
        #         #         thickness = 1
        #         #         det_color=(255,0,0)

        #         #         img_cv = cv2.circle(img_cv,  tuple(pt.cpu().numpy().astype(np.int32).tolist()), radius=6, color=color, thickness=-1)
        #         #         cv2.putText(img_cv, 'p_'+str(i), tuple(pt.cpu().numpy().astype(np.int32).tolist()), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
                        
                    
                    
        #         #     # text = '{:s} {:1.2f}'.format(classes[label],score)
                    
        #         #     # draw_bb_text(frame, text, p1)
        





        losses = {}
        losses.update(detector_losses)
        # losses.update(proposal_losses)
        return losses

        # if torch.jit.is_scripting():
        #     if not self._has_warned:
        #         warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
        #         self._has_warned = True
        #     return (losses, detections)
        # else:
        #     return self.eager_outputs(losses, detections)