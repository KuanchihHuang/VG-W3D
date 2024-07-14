from utils.calibration import Calibration
import numpy as np
import torch
import torch.nn as nn

def generalized_iou_loss(gt_bboxes, pr_bboxes, reduction='mean'):
    """
    gt_bboxes: tensor (-1, 4) xyxy
    pr_bboxes: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
    gt_area = (gt_bboxes[:, 2]-gt_bboxes[:, 0])*(gt_bboxes[:, 3]-gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2]-pr_bboxes[:, 0])*(pr_bboxes[:, 3]-pr_bboxes[:, 1])

    # iou
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    # enclosure
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    giou = iou - (enclosure-union)/enclosure
    loss = 1. - giou
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    return loss


def bbox_overlaps(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """

    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious


def boxes3d_to_corners3d_torch(boxes3d, flip = False):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return: corners_rotated: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    h, w, l, ry = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6], boxes3d[:, 6:7]
    if flip:
        ry = ry + np.pi
    centers = boxes3d[:, 0:3]
    zeros = torch.cuda.FloatTensor(boxes_num, 1).fill_(0)
    ones = torch.cuda.FloatTensor(boxes_num, 1).fill_(1)

    x_corners = torch.cat([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dim = 1)  # (N, 8)
    y_corners = torch.cat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim = 1)  # (N, 8)
    z_corners = torch.cat([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dim = 1)  # (N, 8)
    corners = torch.cat((x_corners.unsqueeze(dim = 1), y_corners.unsqueeze(dim = 1), z_corners.unsqueeze(dim = 1)),
                        dim = 1)  # (N, 3, 8)

    cosa, sina = torch.cos(ry), torch.sin(ry)
    raw_1 = torch.cat([cosa, zeros, sina], dim = 1)
    raw_2 = torch.cat([zeros, ones, zeros], dim = 1)
    raw_3 = torch.cat([-sina, zeros, cosa], dim = 1)
    R = torch.cat((raw_1.unsqueeze(dim = 1), raw_2.unsqueeze(dim = 1), raw_3.unsqueeze(dim = 1)), dim = 1)  # (N, 3, 3)

    corners_rotated = torch.matmul(R, corners)  # (N, 3, 8)
    corners_rotated = corners_rotated + centers.unsqueeze(dim = 2).expand(-1, -1, 8)
    corners_rotated = corners_rotated.permute(0, 2, 1)
    return corners_rotated

class OutputLevelLoss(nn.Module):
    def __init__(self):
        super(OutputLevelLoss, self).__init__()
        self.iou_thres = 0.3
    
    def forward(self, roi_box3d, boxes2d_label, calib, image_size=(384,1280)):
        
        img_h, img_w = image_size
        corners3d_torch = boxes3d_to_corners3d_torch(roi_box3d)
        boxes2d_pred, _ = calib.corners3d_to_img_boxes_torch(corners3d_torch)
        boxes2d_pred[:,0].clamp_(min=0, max=img_w - 1)
        boxes2d_pred[:,1].clamp_(min=0, max=img_h - 1)
        boxes2d_pred[:,2].clamp_(min=0, max=img_w - 1)
        boxes2d_pred[:,3].clamp_(min=0, max=img_h - 1)
        
        overlaps = bbox_overlaps(boxes2d_pred, boxes2d_label)                
        max_overlap, argmax_overlap = torch.max(overlaps, 1)

        fg_inds = torch.nonzero(max_overlap >= self.iou_thres).view(-1)
                
        sampled_rois = boxes2d_pred[fg_inds]
        sampled_gts = boxes2d_label[argmax_overlap[fg_inds]]
        

        if sampled_gts.shape[0] != 0:       
            weak_iou_loss = generalized_iou_loss(sampled_rois, sampled_gts)
        else:
            weak_iou_loss = torch.tensor([0])
        return weak_iou_loss

if __name__ == "__main__":

    img_w = 1280
    img_h = 384

    label_file = "data/label/001264.txt"
    calib_file = "data/calib/001264.txt"
    roi_file = "data/roi/rois.pt"

    calib = Calibration(calib_file)
    rois = torch.load(roi_file)

    with open(label_file, 'r') as f:
        lines = f.readlines()

    box2d_list = []

    for line in lines:
        label = line.strip().split(' ')
        box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype = np.float32)
        if label[0] == 'Car':
            box2d_list.append(box2d)
    boxes2d_label = torch.tensor(box2d_list).cuda()

    out_loss = OutputLevelLoss()
    loss = out_loss(rois,boxes2d_label,calib)
    print(loss)
