import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MultiviewBase(nn.Module):
    def __init__(self, dataset, aggregation='max'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.aggregation = aggregation
        self.control_module = None

    def forward(self, imgs, M=None, proj_mats=None, down=1, visualize=False):
        feat, aux_res = self.get_feat(imgs, M, proj_mats, down, visualize)
        overall_feat = feat.mean(dim=1) if self.aggregation == 'mean' else feat.max(dim=1)[0]
        overall_res = self.get_output(overall_feat, visualize)
        return overall_res, aux_res

    def get_feat(self, imgs, M, proj_mats, down=1, visualize=False):
        raise NotImplementedError

    def get_output(self, overall_feat, visualize=False):
        raise NotImplementedError
