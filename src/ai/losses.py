import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    'C:/Users/aphimaneso/Work/Projects/mmsegmentation/src/')))

def construct_loss(loss_name):
    loss = None
    if loss_name == 'BCEWithLogits_disk':
        loss = BCEWithLogitsLoss_disk()
    if loss_name == "BCE_disk":
        loss = BCELoss_disk()
    if loss_name == 'BCEWithLogits':
        loss = nn.BCEWithLogitsLoss()
    return loss


class BCEWithLogitsLoss_disk(nn.Module):

    def __init__(self):
        super(BCEWithLogitsLoss_disk, self).__init__()

    def forward(self, output, target, disk_mask):      
        output_disk = torch.mul(output, disk_mask)
        target_disk = torch.mul(target, disk_mask)
        bcewl = nn.BCEWithLogitsLoss()
        loss = bcewl(output_disk, target_disk)
        return loss


class BCELoss_disk(nn.Module):

    def __init__(self):
        super(BCELoss_disk, self).__init__()

    def forward(self, output, target, disk_mask):
        output_disk = torch.mul(output, disk_mask)
        target_disk = torch.mul(target, disk_mask)
        bce = nn.BCELoss()
        loss = bce(output_disk, target_disk)
        return loss
