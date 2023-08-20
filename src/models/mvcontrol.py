import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CamControl(nn.Module):
    def __init__(self, action_dim, hidden_dim, variant="conv_base", kernel_size=1, aggregation='max', seed=0):
        super().__init__()
        self.aggregation = aggregation
        self.variant = variant
        self.action_dim = action_dim

        if kernel_size == 1:
            stride, padding = 1, 0
        elif kernel_size == 3:
            stride, padding = 2, 1
        else:
            raise Exception

        if variant == "conv_base":
            self.feat = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding), nn.ReLU(), )
            self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
            self.action_head = nn.Linear(hidden_dim, action_dim)
            self.action_head.weight.data.fill_(0)
            self.action_head.bias.data.fill_(0)
            self.value_head = nn.Linear(hidden_dim, 1)
            self.value_head.weight.data.fill_(0)
            self.value_head.bias.data.fill_(0)
        elif variant == "conv_deep_leaky":
            self.feat = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding),
                                    nn.LeakyReLU(),
                                    nn.AdaptiveAvgPool2d((1, 1)))
            self.value_head = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                        layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                        layer_init(nn.Linear(hidden_dim, 1), std=1.0))
            self.action_head = nn.Sequential(layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(),
                                            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01))
        elif variant == "random_action":
            self.np_random_generator = np.random.default_rng(seed)
        else:
            raise NotImplementedError(f"{variant} not implemented")
        self.log_std = nn.Parameter(-1 * torch.ones(action_dim, dtype=torch.float32))

    def forward(self, feat, randomise, action=None):
        # NOTE: random_action variant is used for random agent
        if self.variant == "random_action":
            # log_prob and state_value are not used in random agent
            log_prob = torch.log(torch.ones(feat.shape[0], self.action_dim, dtype=torch.float32)).to(feat.device)
            state_value = torch.zeros(feat.shape[0], 1, dtype=torch.float32).to(feat.device)
            action = self.np_random_generator.uniform(0, 1, size=(feat.shape[0], self.action_dim))
            return log_prob, state_value, action

        # NOTE: randomise represents whether to sample actions from the distribution or not
        overall_feat = feat.mean(dim=1) if self.aggregation == 'mean' else feat.max(dim=1)[0]
        if self.variant == "conv_base":
            overall_feat = self.feat(overall_feat).amax(dim=[2, 3])
            overall_feat = self.fc(overall_feat)
        elif self.variant == "conv_deep_leaky":
            overall_feat = self.feat(overall_feat)[:, :, 0, 0]

        action_mean = torch.sigmoid(self.action_head(overall_feat))
        state_value = self.value_head(overall_feat)

        action_std = torch.exp(self.log_std).expand_as(action_mean)
        action_dist = Normal(action_mean, action_std)

        # if training network, use randomised action, otherwise use the mean as action
        if action is None:
            if randomise:
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            else:
                action = action_mean
                log_prob = torch.log(torch.ones_like(action))
        else:
            log_prob = action_dist.log_prob(action)

        # return log_prob, state_value, action
        return log_prob, state_value, action.detach().cpu().numpy()
