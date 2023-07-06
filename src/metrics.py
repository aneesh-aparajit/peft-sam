import torch
import torch.nn.functional as F
import numpy as np


class Metrics:
    def __init__(self) -> None:
        self.hard_iou = []
        self.soft_iou = []
        self.hard_dice = []
        self.soft_dice = []
        self.bce = []
    
    @staticmethod
    @torch.no_grad()
    def dice_coef(pred: torch.Tensor, gt: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
        intersection = torch.sum(pred * gt) + eps# this will return the # of pixels which are equal for both.
        union = torch.sum(pred) + torch.sum(gt) + eps # this will be the number of true pixels from the mask and the gt
        dice = (2*intersection)/union
        return dice.mean(dim=(0, 1))
    
    @staticmethod
    @torch.no_grad()
    def iou_coef(pred: torch.Tensor, gt: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
        intersection = torch.sum(pred*gt) + eps
        union =torch.sum(pred + gt - pred*gt) + eps
        iou = intersection / union
        return iou.mean(dim=(0, 1))
    
    @torch.no_grad()
    def get_metrics(self, pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 0.001):
        self.hard_dice.append(self.dice_coef(pred=(pred > thr).float(), gt=gt, eps=eps).item())
        self.soft_dice.append(self.dice_coef(pred=pred, gt=gt, eps=eps).item())
        self.hard_iou.append(self.iou_coef(pred=(pred > thr).float(), gt=gt, eps=eps).item())
        self.soft_iou.append(self.iou_coef(pred=pred, gt=gt, eps=eps).item())
        self.bce.append(F.binary_cross_entropy_with_logits(input=pred, target=gt))

        return {
            'hard_dice': np.mean(self.hard_dice), 
            'soft_dice': np.mean(self.soft_dice),
            'soft_iou': np.mean(self.soft_iou), 
            'hard_iou': np.mean(self.hard_iou),
            'bce_loss': np.mean(self.bce)
        }
