import itertools
import random
import time
import copy
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from src.loss import *
from src.evaluation.evaluate import evaluate, evaluateDetection_py
from src.utils.decode import ctdet_decode, mvdet_decode
from src.utils.nms import nms
from src.utils.meters import AverageMeter
from src.utils.image_utils import add_heatmap_to_image, img_color_denormalize


class BaseTrainer(object):
    def __init__(self, model, logdir, args, ):
        self.model = model
        self.args = args
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def rollout(self, dataset, step, frame, feat, tgt):
        # feat: [B, N, C, H, W]
        # determine the current action based on the feat
        log_prob, state_value, action = \
            self.model.control_module(feat)
        
        # squeeze from batch
        log_prob, action = log_prob.squeeze(), action.squeeze()
        
        # step the current action into the environment, get rid of the batch, get the new feat
        (imgs, aug_mats, proj_mats, _, _, _), done = dataset.step(action, frame)
        # unsqueeze to make batch
        imgs = imgs.unsqueeze(0)
        aug_mats = aug_mats.unsqueeze(0)
        proj_mats = proj_mats.unsqueeze(0)
        # extract corresponding features
        new_feat, _ = \
                self.model.get_feat(imgs.cuda(), aug_mats, proj_mats, self.args.down)
        feat[:, step, :, :, :] = new_feat

        # aggregate new feature, calculate loss and reward
        overall_feat = feat.mean(dim=1) if self.model.aggregation == 'mean' else feat.max(dim=1)[0]
        task_loss, reward = self.task_loss_reward(overall_feat, tgt, done)
        return feat, task_loss, reward, done, (log_prob, state_value, action)

    def expand_episode(self, dataset, feat, frame, tgt):
        # use a [B, N, C, H, W] to record all the feats, input feat is at index 0
        first_feat = feat
        B, _, C, H, W = feat.shape
        N = dataset.num_cam
        feat = torch.zeros([B, N, C, H, W], dtype=feat.dtype).to(feat.device)
        feat[:, 0, :, :, :] = first_feat

        loss = []
        # consider all cameras as initial one
        log_probs, values, actions, rewards = [], [], [], []
        task_loss_s = []

        # indicator whether episode ends, and a counter
        done = False
        steps = 0

        # rollout episode
        while not done:
            feat, task_loss, reward, done, (log_prob, state_value, action) = \
                self.rollout(dataset, steps + 1, frame, feat, tgt)
            # record state & transitions
            log_probs.append(log_prob)
            values.append(state_value)
            actions.append(action)
            rewards.append(reward)
            # loss
            task_loss_s.append(task_loss)
            # increament the counter
            steps += 1

        log_probs, values, actions, rewards = torch.stack(log_probs), torch.cat(values), \
            np.stack(actions), torch.stack(rewards)
        task_loss_s = torch.stack(task_loss_s)

        # calculate returns for each step in episode
        # the batch size is always 1
        R = torch.zeros([B]).cuda()
        returns = torch.empty([steps, B]).float().cuda()
        for i in reversed(range(steps)):
            R = rewards[i] + self.args.gamma * R
            returns[i] = R

        # average return value
        return_avg = returns.mean(1)

        # policy & value loss
        # value_loss = value_loss_s.mean()
        value_loss = F.smooth_l1_loss(values, returns)
        policy_loss = (-log_probs * (returns - values.detach())).mean()

        # task loss
        # task_loss = task_loss_s[-1]
        # loss.append(value_loss + policy_loss + task_loss -
        #             entropies.mean() * self.args.beta_entropy * eps_thres)

        loss = value_loss + policy_loss

        return loss, (return_avg, value_loss, policy_loss), (feat, actions)

    def task_loss_reward(self, overall_feat, tgt, step):
        raise NotImplementedError


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, logdir, args, ):
        super(PerspectiveTrainer, self).__init__(model, logdir, args, )

    def task_loss_reward(self, overall_feat, tgt, done):
        world_heatmap, world_offset = self.model.get_output(overall_feat)
        task_loss = focal_loss(world_heatmap, tgt, reduction='none')
        reward = torch.zeros_like(task_loss).cuda() if not done else -task_loss.detach()
        return task_loss, reward

    def train(self, epoch, dataloader, optimizer, scheduler=None, log_interval=100):
        self.model.train()
        if self.args.base_lr_ratio == 0:
            # fix batch_norm in the backbone when lr is 0
            self.model.base.eval()
        losses = 0
        t0 = time.time()
        for batch_idx, (imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].flatten(0, 1)
            feat, (imgs_heatmap, imgs_offset, imgs_wh) = \
                self.model.get_feat(imgs.cuda(), aug_mats, proj_mats, self.args.down)
            if self.args.interactive:
                # feat is only from the first cam
                loss, (return_avg, value_loss, policy_loss), _ = \
                    self.expand_episode(dataloader.dataset, feat, frame, world_gt['heatmap'])
            else:
                overall_feat = feat.mean(dim=1) if self.model.aggregation == 'mean' else feat.max(dim=1)[0]
                world_heatmap, world_offset = self.model.get_output(overall_feat)
                loss_w_hm = focal_loss(world_heatmap, world_gt['heatmap'])
                loss_w_off = regL1loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
                # loss_w_id = self.ce_loss(world_id, world_gt['reg_mask'], world_gt['idx'], world_gt['pid'])
                loss_img_hm = focal_loss(imgs_heatmap, imgs_gt['heatmap'])
                loss_img_off = regL1loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
                loss_img_wh = regL1loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
                # loss_img_id = self.ce_loss(imgs_id, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['pid'])

                w_loss = loss_w_hm + loss_w_off  # + self.args.id_ratio * loss_w_id
                img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1  # + self.args.id_ratio * loss_img_id
                if self.args.use_mse:
                    loss = F.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device)) + \
                           self.args.alpha * F.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device)) / N
                else:
                    loss = w_loss + img_loss / N * self.args.alpha

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))
            # logging
            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train epoch: {epoch}, batch:{(batch_idx + 1)}, '
                      f'loss: {losses / (batch_idx + 1):.3f}, time: {t_epoch:.1f}')
                if self.args.interactive:
                    print(f'value loss: {value_loss:.3f}, policy loss: {policy_loss:.3f}, '
                          f'return: {return_avg[-1]:.3f}')
                # log the learning rate in each epoch
                print(f"lr: {optimizer.param_groups[0]['lr']}; "
                    f"other_lr: {optimizer.param_groups[1]['lr']}; "
                    f"control_lr: {optimizer.param_groups[2]['lr']};"
                )
        return losses / len(dataloader), None

    def test(self, dataloader):
        t0 = time.time()
        self.model.eval()
        losses = 0.0
        res_list = []
        for batch_idx, (imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) in enumerate(dataloader):
            B, N = imgs_gt['heatmap'].shape[:2]
            with torch.no_grad():
                if not self.args.interactive:
                    # non interactive mode
                    (world_heatmap, world_offset), _ = \
                        self.model(imgs.cuda(), aug_mats, proj_mats, self.args.down)
                    if self.args.use_mse:
                        loss = F.mse_loss(world_heatmap, world_gt['heatmap'].cuda())
                    else:
                        loss = focal_loss(world_heatmap, world_gt['heatmap'])
                else:
                    # interactive mode, initial feat
                    feat, _ = self.model.get_feat(imgs.cuda(), aug_mats, proj_mats, self.args.down)

                    # provide the initial feat, expand episode to get new feat and actions
                    loss, (return_avg, _, _), (feat, actions) = \
                        self.expand_episode(dataloader.dataset, feat, frame, world_gt['heatmap'])

                    overall_feat = feat.mean(dim=1) if self.model.aggregation == 'mean' else feat.max(dim=1)[0]
                    world_heatmap, world_offset = self.model.get_output(overall_feat)
            # append current loss
            losses += loss.item()

            xys = mvdet_decode(torch.sigmoid(world_heatmap), world_offset,
                                reduce=dataloader.dataset.world_reduce).cpu()
            grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
            if dataloader.dataset.base.indexing == 'xy':
                positions = grid_xy
            else:
                positions = grid_xy[:, :, [1, 0]]
            for b in range(B):
                ids = scores[b].squeeze() > self.args.cls_thres
                pos, s = positions[b, ids], scores[b, ids, 0]
                ids, count = nms(pos, s, 20, np.inf)
                res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                res_list.append(res)

        # stack all results from all frames together
        res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        moda, modp, precision, recall, stats = evaluateDetection_py(res,
                                                                    dataloader.dataset.gt_array,
                                                                    dataloader.dataset.frames)
        print(f'Test, loss: {losses / len(dataloader):.6f}, moda: {moda:.1f}%, modp: {modp:.1f}%, '
                f'prec: {precision:.1f}%, recall: {recall:.1f}%'
                f', time: {time.time() - t0:.1f}s')

        return losses / len(dataloader), [moda, modp, precision, recall, ]
