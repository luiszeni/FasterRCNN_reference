import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from pdb import set_trace as pause
import torchvision.ops as ops


class RoiPoolLayer(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super().__init__()
        self.dim_in = dim_in

        res = 7
        self.roi_pool = ops.RoIPool(output_size=(res, res), spatial_scale=spatial_scale)

        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = 4096

        roi_size = res
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, rois):

        x = self.roi_pool(x, rois)
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x
