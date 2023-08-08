import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


class CamControl(nn.Module):
    def __init__(self, action_dim, hidden_dim, kernel_size=1, aggregation='max'):
        super().__init__()
        self.aggregation = aggregation

        if kernel_size == 1:
            stride, padding = 1, 0
        elif kernel_size == 3:
            stride, padding = 2, 1
        else:
            raise Exception

        self.feat = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding), nn.ReLU(), )
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.action_head.weight.data.fill_(0)
        self.action_head.bias.data.fill_(0)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.value_head.weight.data.fill_(0)
        self.value_head.bias.data.fill_(0)
        # TODO: need way to converge log_std
        self.log_std = nn.Parameter(-2.0 * torch.ones(action_dim, dtype=torch.float32))

    def forward(self, feat):
        overall_feat = feat.mean(dim=1) if self.aggregation == 'mean' else feat.max(dim=1)[0]
        overall_feat = self.feat(overall_feat).amax(dim=[2, 3])
        overall_feat = self.fc(overall_feat)

        action_mean = torch.sigmoid(self.action_head(overall_feat))
        state_value = self.value_head(overall_feat)
        action_std = torch.exp(self.log_std)

        action_dist = Normal(action_mean, action_std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        # return log_prob, state_value, action, entropy
        return log_prob, state_value, action.detach().cpu().numpy()
