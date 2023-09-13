import itertools
import random
from collections import deque
import time
import copy
import json
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


class PerspectiveTrainer(object):
    def __init__(self, model, logdir, args, ):
        self.model = model
        self.args = args
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def rollout(self, dataset, step, frame, feat, tgt, randomise):
        # feat: [B, N, C, H, W]
        # determine the current action based on the feat
        log_prob, state_value, action = \
            self.model.control_module(feat, randomise)
        
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
        task_loss, reward = self.task_loss_reward(dataset, frame, overall_feat, tgt, done, step)
        return feat, task_loss, reward, done, (log_prob, state_value, action)

    def expand_episode(self, dataset, feat, frame, tgt, return_avg, training=False):
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

        # obtain predicion for the first feature map
        overall_feat = feat.mean(dim=1) if self.model.aggregation == 'mean' else feat.max(dim=1)[0]
        world_heatmap, world_offset = self.model.get_output(overall_feat)
        task_loss = focal_loss(world_heatmap, tgt, reduction='none')

        # initialize last_reward
        if self.args.reward in ["epi_loss", "epi_cover_mean", "epi_cover_max", "step_cover", "epi_moda"]:
            # NOTE: we don't need last_reward for these reward
            self.last_reward = 0
        elif self.args.reward == "delta_loss":
            # option 2: use (delta -task_loss) as the reward
            self.last_reward = -task_loss.detach()
        elif self.args.reward in ["delta_moda", "cover+delta_moda"]:
            # option 7: use (MODA) as the reward
            # option 8: use (coverage + MODA) as the reward
            res_list = self.bev_prediction(world_heatmap, world_offset, dataset, frame)
            res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            # only evaluate stats for the current frame
            moda, modp, precision, recall, stats = evaluateDetection_py(res, dataset.gt_array)
            moda = torch.tensor([moda / 100]).cuda()
            self.last_reward = moda
        else:
            raise NotImplementedError("Reward type not implemented")

        # rollout episode
        while not done:
            feat, task_loss, reward, done, (log_prob, state_value, action) = \
                self.rollout(dataset, steps + 1, frame, feat, tgt, training)
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
        return_avg = returns.mean(1) if return_avg is None else returns.mean(1) * 0.05 + return_avg * 0.95

        # policy & value loss
        # value_loss = value_loss_s.mean()
        value_loss = F.smooth_l1_loss(values, returns)
        policy_loss = (-log_probs * (returns - values.detach())).mean()

        # task loss
        task_loss = task_loss_s[-1]

        # task loss is not used in the training, but used for fine-tuning
        if self.args.fine_tune:
            loss = task_loss
        else:
            loss = value_loss * self.args.vf_ratio + policy_loss

        return loss, (return_avg, value_loss, policy_loss), (feat, actions)
    
    def bev_prediction(self, world_heatmap, world_offset, dataset, frame):
        res_list = []
        xys = mvdet_decode(torch.sigmoid(world_heatmap), world_offset,
                            reduce=dataset.world_reduce).cpu()
        grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
        if dataset.base.indexing == 'xy':
            positions = grid_xy
        else:
            positions = grid_xy[:, :, [1, 0]]
        for b in range(world_heatmap.shape[0]):
            ids = scores[b].squeeze() > self.args.cls_thres
            pos, s = positions[b, ids], scores[b, ids, 0]
            ids, count = nms(pos, s, 20, np.inf)
            res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
            res_list.append(res)
        return res_list

    def task_loss_reward(self, dataset, frame, overall_feat, tgt, done, step):
        world_heatmap, world_offset = self.model.get_output(overall_feat)
        task_loss = focal_loss(world_heatmap, tgt, reduction='none')
        # reward is defined as the delta value from previous reward
        if self.args.reward == "epi_loss":
            # option 1: use (episode -task_loss) as the reward
            reward = torch.zeros_like(task_loss).cuda() if not done else -task_loss.detach()
            # set current_reward as last_reward for the next step
            self.last_reward = reward
        elif self.args.reward == "delta_loss":
            # option 2: use (delta -task_loss) as the reward
            reward = -task_loss.detach() - self.last_reward
            # set current_reward as last_reward for the next step
            self.last_reward = -task_loss.detach()
        elif self.args.reward == "epi_cover_mean":
            # option 3: use (episode cover mean) as the reward
            world_coverage = dataset.Rworld_coverage.mean(0).mean(-1).mean(-1).cuda()
            reward = torch.zeros_like(task_loss).cuda() if not done else world_coverage
            self.last_reward = reward
        elif self.args.reward == "epi_cover_max":
            # option 4： 
            world_coverage = dataset.Rworld_coverage.max(0)[0].mean(-1).mean(-1).cuda()
            reward = torch.zeros_like(task_loss).cuda() if not done else world_coverage
            self.last_reward = reward
        elif self.args.reward == "step_cover":
            # option 5: use (coverage) as the reward
            world_coverage = dataset.Rworld_coverage[step].mean(-1).mean(-1).cuda()
            reward = world_coverage
            # set current_reward as last_reward for the next step
            self.last_reward = reward
        elif self.args.reward == "epi_moda":
            # option 6: use (episode MODA) as the reward
            if done:
                res_list = self.bev_prediction(world_heatmap, world_offset, dataset, frame)
                res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                # only evaluate stats for the current frame
                moda, modp, precision, recall, stats = evaluateDetection_py(res, dataset.gt_array)
                moda = torch.tensor([moda / 100]).cuda()
                reward = moda
            else:
                reward = torch.zeros_like(task_loss).cuda()
            # set current `moda` as last_reward for the next step
            self.last_reward = reward
        elif self.args.reward == "delta_moda":
            # option 7: use (delta MODA) as the reward
            res_list = self.bev_prediction(world_heatmap, world_offset, dataset, frame)
            res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            # only evaluate stats for the current frame
            moda, modp, precision, recall, stats = evaluateDetection_py(res, dataset.gt_array)
            moda = torch.tensor([moda / 100]).cuda()
            reward = moda - self.last_reward
            # set current `moda` as last_reward for the next step
            self.last_reward = moda
        elif self.args.reward == "cover+delta_moda":
            # option 8: use (coverage + MODA) as the reward
            world_coverage = dataset.Rworld_coverage[step].mean(-1).mean(-1)
            res_list = self.bev_prediction(world_heatmap, world_offset, dataset, frame)
            res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            # only evaluate stats for the current frame
            moda, modp, precision, recall, stats = evaluateDetection_py(res, dataset.gt_array)
            moda = torch.tensor([moda / 100]).cuda()
            # NOTE: coefficient for balancing two factors - 0.167
            reward = 0.167 * world_coverage.cuda() + moda - self.last_reward
            # set current `moda` as last_reward for the next step
            self.last_reward = moda
        else:
            raise NotImplementedError("Reward type not implemented")
        return task_loss, reward

    def train(self, epoch, dataloader, optimizer, scheduler=None, log_interval=100):
        # make the directory for saving coverage_maps for the current epoch
        if self.args.interactive:
            coverage_path = os.path.join(self.logdir, "coverage_maps", f"epoch_{epoch}")
            if not os.path.exists(coverage_path):
                os.makedirs(coverage_path)
            cam_configs_path = os.path.join(self.logdir, "cam_configs", f"epoch_{epoch}")
            if not os.path.exists(cam_configs_path):
                os.makedirs(cam_configs_path)
        else:
            coverage_path = os.path.join(self.logdir, "coverage.jpg")
        # make the directory for saving feature_maps for the current epoch
        feature_map_path = os.path.join(self.logdir, "feature_maps", f"epoch_{epoch}")
        if not os.path.exists(feature_map_path):
            os.makedirs(feature_map_path)
        
        # set model mode
        self.model.train()
        if self.args.base_lr_ratio == 0:
            # fix batch_norm in the backbone when lr is 0
            self.model.base.eval()
        losses = 0
        return_avg = None
        t0 = time.time()
        for batch_idx, (imgs, aug_mats, proj_mats, world_gt, imgs_gt, frame) in enumerate(dataloader):
            B, N = imgs.shape[:2]
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].flatten(0, 1)
            feat, (imgs_heatmap, imgs_offset, imgs_wh) = \
                self.model.get_feat(imgs.cuda(), aug_mats, proj_mats, self.args.down)
            if self.args.interactive:
                # feat is only from the first cam
                loss, (return_avg, value_loss, policy_loss), (feat, action) = \
                    self.expand_episode(dataloader.dataset, feat, frame, world_gt['heatmap'], return_avg, True)
                overall_feat = feat.mean(dim=1) if self.model.aggregation == 'mean' else feat.max(dim=1)[0]
            else:
                # in non-interactive mode, save the initial camera coverage if it doesn't exist
                if not os.path.exists(coverage_path):
                    plt.imsave(coverage_path,
                        dataloader.dataset.Rworld_coverage.max(dim=0)[0].squeeze())

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
                # log the learning rate in each epoch
                print(f"lr: {optimizer.param_groups[0]['lr']}; "
                    f"other_lr: {optimizer.param_groups[1]['lr']}; "
                    f"control_lr: {optimizer.param_groups[2]['lr']};")
                
                # save feature map as images for logs
                plt.imsave(os.path.join(feature_map_path, f"{batch_idx + 1}.jpg"),
                           torch.norm(overall_feat[0].detach(), dim=0).cpu().numpy())

                if self.args.interactive:
                    print(f'value loss: {value_loss:.3f}, policy loss: {policy_loss:.3f}, '
                          f'return: {return_avg[-1]:.3f}')
                    # log camera positions & directions
                    with open(os.path.join(cam_configs_path, f"{batch_idx + 1}.json"), "w") as fp:
                        json.dump(dataloader.dataset.base.env.camera_configs, fp, indent=4)
                    # log the variance parameter in the control module
                    print(f'std of control module: {torch.exp(self.model.control_module.log_std).detach().cpu().numpy()}')
                    # print world coverage for debug
                    world_coverage = dataloader.dataset.Rworld_coverage
                    print(f'World coverage (max): {world_coverage.max(dim=0)[0].mean(-1).mean(-1).item()}')
                    # save the world_coverage map as images for logs
                    plt.imsave(os.path.join(coverage_path, f"{batch_idx + 1}_max.jpg"), world_coverage.max(dim=0)[0].squeeze())
                    print(f'World coverage (mean): {world_coverage.mean(dim=0).mean(-1).mean(-1).item()}')
                    # save the world_coverage map as images for logs
                    plt.imsave(os.path.join(coverage_path, f"{batch_idx + 1}_mean.jpg"), world_coverage.mean(dim=0).squeeze())
                print()
        return losses / len(dataloader), None

    def test(self, dataloader):
        t0 = time.time()
        self.model.eval()
        losses = 0.0
        return_avg = None
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
                        self.expand_episode(dataloader.dataset, feat, frame, world_gt['heatmap'], return_avg, False)

                    overall_feat = feat.mean(dim=1) if self.model.aggregation == 'mean' else feat.max(dim=1)[0]
                    world_heatmap, world_offset = self.model.get_output(overall_feat)
            # append current loss
            losses += loss.item()

            # put BEV prediction result list into res_list
            res_list.extend(self.bev_prediction(world_heatmap, world_offset, dataloader.dataset, frame))

        # stack all results from all frames together
        res = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        moda, modp, precision, recall, stats = evaluateDetection_py(res,
                                                                    dataloader.dataset.gt_array,
                                                                    dataloader.dataset.frames)
        print(f'Test, loss: {losses / len(dataloader):.6f}, moda: {moda:.1f}%, modp: {modp:.1f}%, '
                f'prec: {precision:.1f}%, recall: {recall:.1f}%'
                f', time: {time.time() - t0:.1f}s')
        if self.args.interactive:
            # losses & print world coverage for debug
            world_coverage = dataloader.dataset.Rworld_coverage
            print(f'return: {return_avg[-1]:.3f} '
                f'World coverage (max): {world_coverage.max(dim=0)[0].mean(-1).mean(-1).item()} '
                f'World coverage (mean): {world_coverage.mean(dim=0).mean(-1).mean(-1).item()}')
        print()

        return losses / len(dataloader), [moda, modp, precision, recall, ]
