import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample

def _sigmoid_cross_entropy_with_logits(logits, labels):
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss

class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focusses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self,
                prediction_tensor,
                target_tensor,
                weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return focal_cross_entropy_loss * weights



class FeatureLevelLoss(nn.Module):
    def __init__(self):
        super(FeatureLevelLoss, self).__init__()
        self.seg_loss = SigmoidFocalClassificationLoss()
    
    def forward(self, point_logit, img_logit, l_xy_norm, point_seg_label):
        """
        Args:
            point_logit: (B, N, 1)
            img_logit: ([B, 1, H, W])
            l_xy_coord: torch.Size([B, N, 2])
            point_seg_label: (B, N, 1)
        """
        B = point_logit.shape[0]
        l_xy_norm = l_xy_norm.unsqueeze(1)
        proj_img_logit = grid_sample(img_logit, l_xy_norm).squeeze(2)  # (B,C,1,N)
        proj_img_logit = F.softmax(proj_img_logit.view(B, -1), dim=-1)
        point_logit = F.softmax(point_logit.view(B, -1), dim=-1)
        kl_loss = F.kl_div(point_logit, proj_img_logit.detach(), reduction='none')

        pos = (point_seg_label > 0).float()
        seg_loss = self.seg_loss(point_logit.view(-1), point_seg_label.view(-1), pos.view(-1))

        return kl_loss.mean() + seg_loss.mean()

if __name__ == "__main__":
    #Sample code for batch_size = 4, number of point = 16384, image height=384, image_width=1280
    lidar_feat = torch.rand(4, 16384, 1).cuda()
    img_feat = torch.rand(4,1,384,1280).cuda()
    l_xy_norm = (torch.rand(4,16384,2) * 2 - 1).cuda() #the range should be -1~1, (x_pixel/(width-1)*2 -1, y_pixel/(height-1)*2 -1)
    seg_label = torch.randint(0, 1, (4,16384,1)).cuda()
    f_loss = FeatureLevelLoss()

    loss = f_loss(lidar_feat, img_feat, l_xy_norm, seg_label)
    print(loss)
    
