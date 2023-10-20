####################################
# EXTERNAL LIBRARY
####################################
import torch
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
import os
import shutil
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from tensorboard_logger import Logger as TbLogger
####################################
# INTERNAL LIBRARY
####################################
from source.utilities import Get_Logger, Average_Meter, augment_xy_data_by_n_fold_3obj, augment_xy_data_by_128_fold_3obj
from matplotlib import pyplot as plt
# from non_dominated_sort import get_non_dominated
import hvwfg
import torch.nn.functional as F
from pygmo import hypervolume
from update_PE6 import *
from cal_ps_hv import cal_ps_hv

####################################
# PROJECT VARIABLES
####################################
from HYPER_PARAMS import *
from TORCH_OBJECTS import *



####################################
# PROJECT MODULES (to swap as needed)
####################################
import source.MODEL__Actor.grouped_actors as A_Module
import source.TRAIN_N_EVAL.Train_Grouped_Actors as T_Module
# import source.TRAIN_N_EVAL.Evaluate_Grouped_Actors as E_Module
from source.mo_travelling_saleman_problem import TSP_DATA_LOADER__RANDOM, GROUP_ENVIRONMENT

if USE_CUDA:
    torch.cuda.set_device(CUDA_DEVICE_NUM)
    device = torch.device('cuda', CUDA_DEVICE_NUM)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

torch.manual_seed(SEED)
np.random.seed(SEED)

############################################################################################################
############################################################################################################

# Objects to Use
actor = A_Module.ACTOR().to(device)
# optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)

class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, model):

        super(Meta, self).__init__()
        self.meta_lr = META_LR
        self.task_num = TASK_NUM
        self.pomo_size = TSP_SIZE
        self.model = deepcopy(model)
        self.submodel = deepcopy(model)
        self.suboptimizer = optim.Adam(self.submodel.parameters(), lr=ACTOR_LEARNING_RATE)

    def forward(self, epoch=0):
        support_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * UPDATE_STEP, num_nodes=TSP_SIZE, batch_size=TRAIN_BATCH_SIZE)
        batch_s = TRAIN_BATCH_SIZE
        alpha = 1
        train_sols = [torch.empty(batch_s, 0, 3) for _ in range(UPDATE_STEP)]
        train_sols_mask = [torch.empty(batch_s, 0) for _ in range(UPDATE_STEP)]
        ref = REF[None, None, :].repeat(batch_s, 1, 1).float().cuda()  # ref point
        ref_mask = torch.zeros(batch_s, ref.shape[1])
        hv_reward = torch.zeros(batch_s, self.pomo_size)
        for st in tqdm(range(SOLVING_TIMES)):
            meta_lr = META_LR * (1. - (epoch * SOLVING_TIMES + st) / (TOTAL_EPOCH * SOLVING_TIMES))
            actor_new_weights = []
            actor_weights_original = deepcopy(self.model.state_dict())
            for task_id in range(self.task_num):
                hv_w = torch.rand(1)
                pref = np.random.dirichlet((alpha, alpha, alpha), None)
                pref = torch.tensor(pref)
                self.submodel.load_state_dict(actor_weights_original)
                self.submodel.train()
                batch_i = 0
                for data in support_loader:
                    # data.shape = (batch_s, TSP_SIZE, 2)
                    # batch_s = data.size(0)
                    sols = train_sols[batch_i]
                    sols_mask = train_sols_mask[batch_i]
                    if sols.shape[1] > NEIGHBOR:
                        ws = (pref * sols).sum(dim=2)
                        ind = torch.topk(ws, NEIGHBOR, dim=-1, largest=False).indices
                        sel = sols.gather(1, ind[:, :, None].repeat(1, 1, 3))
                        sel_mask = sols_mask.gather(1, ind)
                    else:
                        sel = sols
                        sel_mask = sols_mask
                    sel = torch.cat((ref, sel), dim=1)
                    sel_mask = torch.cat((ref_mask, sel_mask), dim=-1)
                    sel_mask_pomo = sel_mask[:, None, :].repeat(1, self.pomo_size, 1)

                    # Actor Group Move
                    ###############################################
                    env = GROUP_ENVIRONMENT(data)
                    group_s = TSP_SIZE
                    group_state, reward, done = env.reset(group_size=group_s)
                    self.submodel.reset(group_state, sel, sel_mask, sel_mask_pomo)

                    # First Move is given
                    first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
                    group_state, reward, done = env.step(first_action)

                    group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
                    while not done:
                        self.submodel.update(group_state)
                        action_probs = self.submodel.get_action_probabilities()
                        # shape = (batch, group, TSP_SIZE)
                        action = action_probs.reshape(batch_s * group_s, -1).multinomial(1).squeeze(dim=1).reshape(
                            batch_s,
                            group_s)
                        # shape = (batch, group)
                        group_state, reward, done = env.step(action)

                        batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
                        group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                        chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s,
                                                                                                        group_s)
                        # shape = (batch, group)
                        group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

                    # LEARNING - Actor
                    ###############################################
                    assert AGG == 1 or AGG == 2, "Only support Weighted-Sum and Weighted-Tchebycheff"
                    # reward was negative, here we set it to positive to calculate TCH
                    reward = - reward
                    if AGG == 1:
                        ws_reward = (pref * reward).sum(dim=-1)
                        agg_reward = ws_reward
                    elif AGG == 2:
                        z = torch.ones(reward.shape).cuda() * 0.0
                        tch_reward = pref * (reward - z)
                        tch_reward, _ = tch_reward.max(dim=-1)

                        agg_reward = tch_reward
                    else:
                        return NotImplementedError

                    # HV reward
                    s_ = sel[:, 1:, :].clone()
                    s_[s_[:, :, 0] >= REF[0]] = REF[0].cuda() - 1e-4
                    s_[s_[:, :, 1] >= REF[1]] = REF[1].cuda() - 1e-4
                    s_[s_[:, :, 2] >= REF[2]] = REF[2].cuda() - 1e-4
                    r_ = reward.clone()
                    r_[r_[:, :, 0] >= REF[0]] = REF[0].cuda() - 1e-4
                    r_[r_[:, :, 1] >= REF[1]] = REF[1].cuda() - 1e-4
                    r_[r_[:, :, 2] >= REF[2]] = REF[2].cuda() - 1e-4

                    for b_i in range(batch_s):
                        for i_i in range(self.pomo_size):
                            hv_reward[b_i, i_i] = hypervolume(
                                torch.cat((s_[b_i, :, :], r_[b_i, i_i, None]), dim=0).cpu().numpy().astype(
                                    float)).compute(REF.numpy()) / (REF[0] * REF[1] * REF[2])
                    # update non-dominated solution set
                    sols, flag, NDS = update_EP(reward, sols)
                    sols[sols == 1e4] = REF[0].float().cuda()
                    sols[sols == 1e3] = REF[1].float().cuda()
                    sols[sols == 1e5] = REF[2].float().cuda()
                    sols_mask = flag.float()
                    sols_mask[sols_mask == 1] = float('-inf')
                    train_sols[batch_i] = sols
                    train_sols_mask[batch_i] = sols_mask

                    # set back reward to negative
                    reward = -reward
                    # agg_reward = -agg_reward
                    # group_reward = agg_reward
                    group_reward = -agg_reward * (1 - hv_w) + hv_reward * hv_w
                    group_log_prob = group_prob_list.log().sum(dim=2)
                    # shape = (batch, group)

                    group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

                    group_loss = -group_advantage * group_log_prob
                    loss = group_loss.mean()
                    # shape = (batch, group)

                    self.suboptimizer.zero_grad()
                    loss.backward()
                    self.suboptimizer.step()
                    batch_i += 1

                actor_new_weights.append(deepcopy(self.submodel.state_dict()))

            actor_ws = len(actor_new_weights)
            actor_fweights = {name: actor_new_weights[0][name] / float(actor_ws) for name in actor_new_weights[0]}
            for i in range(1, actor_ws):
                for name in actor_new_weights[i]:
                    actor_fweights[name] += actor_new_weights[i][name] / float(actor_ws)

            self.model.load_state_dict({name:
                                            actor_weights_original[name] + (
                                                    actor_fweights[name] - actor_weights_original[
                                                name]) * meta_lr for
                                        name in actor_weights_original})

        hv = torch.zeros(batch_s)
        for b_i in range(batch_s):
            hv[b_i] = hvwfg.wfg(sols[b_i][:NDS[b_i]].cpu().numpy().astype(float),
                                REF.numpy().astype(float)) / (REF[0] * REF[1] * REF[2])
        print('hv:', hv.mean().item(), 'NDS:', NDS.float().mean().item(), 'loss:', loss.item())
        return

    def finetune(self, pref=None, finetune_loader=None, model=None, hv_w=None, train_sols=None, train_sols_mask=None):
        if pref is None:
            pref = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        if finetune_loader is None:
            finetune_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * FINETUNE_STEP, num_nodes=TSP_SIZE, batch_size=TRAIN_BATCH_SIZE)
        if model is None:
            fine_model = deepcopy(self.model)
        else:
            fine_model = deepcopy(model)
        print("--------------------------------------------")
        print("pref:{}, {}, {}, hv_w:{}".format(pref[0].item(), pref[1].item(), pref[2].item(), hv_w.item()))
        fine_model.train()
        fine_optimizer = optim.Adam(fine_model.parameters(), lr=ACTOR_LEARNING_RATE)

        batch_s = TRAIN_BATCH_SIZE
        ref = REF[None, None, :].repeat(batch_s, 1, 1).float().cuda()  # ref point
        ref_mask = torch.zeros(batch_s, ref.shape[1])
        hv_reward = torch.zeros(batch_s, self.pomo_size)

        step = 0
        for data in finetune_loader:
            sols = train_sols[step]
            sols_mask = train_sols_mask[step]
            if sols.shape[1] > NEIGHBOR:
                ws = (pref * sols).sum(dim=2)
                ind = torch.topk(ws, NEIGHBOR, dim=-1, largest=False).indices
                sel = sols.gather(1, ind[:, :, None].repeat(1, 1, 3))
                sel_mask = sols_mask.gather(1, ind)
            else:
                sel = sols
                sel_mask = sols_mask
            sel = torch.cat((ref, sel), dim=1)
            sel_mask = torch.cat((ref_mask, sel_mask), dim=-1)
            sel_mask_pomo = sel_mask[:, None, :].repeat(1, self.pomo_size, 1)

            # Actor Group Move
            ###############################################
            env = GROUP_ENVIRONMENT(data)
            group_s = TSP_SIZE
            group_state, reward, done = env.reset(group_size=group_s)
            fine_model.reset(group_state, sel, sel_mask, sel_mask_pomo)

            # First Move is given
            first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(first_action)

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                fine_model.update(group_state)
                action_probs = fine_model.get_action_probabilities()
                # shape = (batch, group, TSP_SIZE)
                action = action_probs.reshape(batch_s * group_s, -1).multinomial(1).squeeze(dim=1).reshape(
                    batch_s,
                    group_s)
                # shape = (batch, group)
                group_state, reward, done = env.step(action)

                batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s,
                                                                                                group_s)
                # shape = (batch, group)
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

            # LEARNING - Actor
            ###############################################
            assert AGG == 1 or AGG == 2, "Only support Weighted-Sum and Weighted-Tchebycheff"
            # reward was negative, here we set it to positive to calculate TCH
            reward = - reward
            if AGG == 1:
                ws_reward = (pref * reward).sum(dim=-1)
                agg_reward = ws_reward
            elif AGG == 2:
                z = torch.ones(reward.shape).cuda() * 0.0
                tch_reward = pref * (reward - z)
                tch_reward, _ = tch_reward.max(dim=-1)

                agg_reward = tch_reward
            else:
                return NotImplementedError

            # HV reward
            s_ = sel[:, 1:, :].clone()
            s_[s_[:, :, 0] >= REF[0]] = REF[0].cuda() - 1e-4
            s_[s_[:, :, 1] >= REF[1]] = REF[1].cuda() - 1e-4
            s_[s_[:, :, 2] >= REF[2]] = REF[2].cuda() - 1e-4
            r_ = reward.clone()
            r_[r_[:, :, 0] >= REF[0]] = REF[0].cuda() - 1e-4
            r_[r_[:, :, 1] >= REF[1]] = REF[1].cuda() - 1e-4
            r_[r_[:, :, 2] >= REF[2]] = REF[2].cuda() - 1e-4

            for b_i in range(batch_s):
                for i_i in range(self.pomo_size):
                    hv_reward[b_i, i_i] = hypervolume(
                        torch.cat((s_[b_i, :, :], r_[b_i, i_i, None]), dim=0).cpu().numpy().astype(
                            float)).compute(REF.numpy()) / (REF[0] * REF[1] * REF[2])
            # update non-dominated solution set

            if reward.shape[1] > CANDIDATE:
                ws_r = (pref * reward).sum(dim=2)
                ind_r = torch.topk(ws_r, CANDIDATE, dim=-1, largest=False).indices
                cdd = reward.gather(1, ind_r[:, :, None].repeat(1, 1, 3))
            else:
                cdd = reward
            # cdd = reward
            if sols.shape[1] > NEIGHBOR:
                split_flag = torch.zeros(batch_s, sols.shape[1])
                split_flag = split_flag.scatter(1, ind, 1)
                rest = sols[split_flag == 0].reshape(batch_s, -1, 3)
                rest_mask = sols_mask[split_flag == 0].reshape(batch_s, -1)
                sols, flag, NDS = update_EP(cdd, sel)
                sols[sols == 1e4] = REF[0].float().cuda()
                sols[sols == 1e3] = REF[1].float().cuda()
                sols[sols == 1e5] = REF[2].float().cuda()
                sols_mask = flag.float()
                sols_mask[sols_mask == 1] = float('-inf')
                sols = torch.cat((sols, rest), dim=1)
                sols_mask = torch.cat((sols_mask, rest_mask), dim=1)
            else:
                sols, flag, NDS = update_EP(cdd, sols)
                sols[sols == 1e4] = REF[0].float().cuda()
                sols[sols == 1e3] = REF[1].float().cuda()
                sols[sols == 1e5] = REF[2].float().cuda()
                sols_mask = flag.float()
                sols_mask[sols_mask == 1] = float('-inf')
            train_sols[step] = sols
            train_sols_mask[step] = sols_mask

            # set back reward to negative
            reward = -reward
            # agg_reward = -agg_reward
            # group_reward = agg_reward
            group_reward = -agg_reward * (1 - hv_w) + hv_reward * hv_w
            group_log_prob = group_prob_list.log().sum(dim=2)
            # shape = (batch, group)

            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

            group_loss = -group_advantage * group_log_prob
            # shape = (batch, group)
            loss = group_loss.mean()

            fine_optimizer.zero_grad()
            loss.backward()
            fine_optimizer.step()

            step += 1
            print('finetune_step:{}, reward:{}, loss:{}'.format(step, group_reward.mean().item(), loss.item()))


        return fine_model, train_sols, train_sols_mask

    def test(self, model=None, testdata=None, pref=None, test_sols=None, test_sols_mask=None):
        if model is None:
            test_model = deepcopy(self.model)
        else:
            test_model = model
        if testdata is None:
            testdata = Tensor(np.random.rand(TEST_BATCH_SIZE, TSP_SIZE, 6))
        if pref is None:
            pref = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        print("testing meta_model...")

        test_model.eval()
        test_NDS = torch.zeros(TEST_DATASET_SIZE)
        episode = 0
        batch_i = 0
        while True:
            remaining = testdata.size(0) - episode
            batch_s = min(TEST_BATCH_SIZE, remaining)
            testdata_batch = testdata[episode: episode + batch_s]

            ref = REF[None, None, :].repeat(batch_s, 1, 1).float().cuda()  # ref point
            ref_mask = torch.zeros(batch_s, ref.shape[1])
            sols = test_sols[batch_i]
            sols_mask = test_sols_mask[batch_i]
            if sols.shape[1] > NEIGHBOR:
                ws = (pref * sols).sum(dim=2)
                ind = torch.topk(ws, NEIGHBOR, dim=-1, largest=False).indices
                sel = sols.gather(1, ind[:, :, None].repeat(1, 1, 3))
                sel_mask = sols_mask.gather(1, ind)
            else:
                sel = sols
                sel_mask = sols_mask
            sel = torch.cat((ref, sel), dim=1)
            sel_mask = torch.cat((ref_mask, sel_mask), dim=-1)
            sel_mask_pomo = sel_mask[:, None, :].repeat(1, self.pomo_size, 1)

            with torch.no_grad():

                env = GROUP_ENVIRONMENT(testdata_batch)
                group_s = TSP_SIZE
                group_state, reward, done = env.reset(group_size=group_s)
                test_model.reset(group_state, sel, sel_mask, sel_mask_pomo)

                # First Move is given
                first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
                group_state, reward, done = env.step(first_action)

                while not done:
                    test_model.update(group_state)
                    action_probs = test_model.get_action_probabilities()
                    # shape = (batch, group, TSP_SIZE)
                    action = action_probs.argmax(dim=2)
                    # shape = (batch, group)
                    group_state, reward, done = env.step(action)

                # reward was negative, here we set it to positive
                reward = -reward
                # update non-dominated solution set

                if reward.shape[1] > CANDIDATE:
                    ws_r = (pref * reward).sum(dim=2)
                    ind_r = torch.topk(ws_r, CANDIDATE, dim=-1, largest=False).indices
                    cdd = reward.gather(1, ind_r[:, :, None].repeat(1, 1, 3))
                else:
                    cdd = reward
                # cdd = reward
                if sols.shape[1] > NEIGHBOR:
                    split_flag = torch.zeros(batch_s, sols.shape[1])
                    split_flag = split_flag.scatter(1, ind, 1)
                    rest = sols[split_flag == 0].reshape(batch_s, -1, 3)
                    rest_mask = sols_mask[split_flag == 0].reshape(batch_s, -1)
                    sols, flag, NDS = update_EP(cdd, sel)
                    sols[sols == 1e4] = REF[0].float().cuda()
                    sols[sols == 1e3] = REF[1].float().cuda()
                    sols[sols == 1e5] = REF[2].float().cuda()
                    sols_mask = flag.float()
                    sols_mask[sols_mask == 1] = float('-inf')
                    sols = torch.cat((sols, rest), dim=1)
                    sols_mask = torch.cat((sols_mask, rest_mask), dim=1)
                else:
                    sols, flag, NDS = update_EP(cdd, sols)
                    sols[sols == 1e4] = REF[0].float().cuda()
                    sols[sols == 1e3] = REF[1].float().cuda()
                    sols[sols == 1e5] = REF[2].float().cuda()
                    sols_mask = flag.float()
                    sols_mask[sols_mask == 1] = float('-inf')
                test_sols[batch_i] = sols
                test_sols_mask[batch_i] = sols_mask
                test_NDS[episode: episode + batch_s] = NDS

            batch_i += 1
            episode = episode + batch_s
            if episode == TEST_DATASET_SIZE:
                break

        return test_sols, test_sols_mask, test_NDS

    def test_aug(self, model=None, testdata=None, pref=None, test_sols_aug=None, test_sols_mask_aug=None):
        if model is None:
            test_model_aug = deepcopy(self.model)
        else:
            test_model_aug = model
        if testdata is None:
            testdata = Tensor(np.random.rand(TEST_BATCH_SIZE, TSP_SIZE, 6))
        if pref is None:
            pref = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        print("testing meta_model_aug...")

        test_model_aug.eval()
        aug_factor = AUG_NUM
        if aug_factor == 512:
            testdata_aug = augment_xy_data_by_n_fold_3obj(testdata, aug_factor)
        elif aug_factor == 128:
            testdata_aug = augment_xy_data_by_128_fold_3obj(testdata)

        test_NDS_aug = torch.zeros(TEST_DATASET_SIZE)
        episode = 0
        batch_i = 0
        while True:
            remaining = testdata.size(0) - episode
            batch_s = min(TESTAUG_BATCH_SIZE, remaining) * aug_factor
            testdata_batch = testdata_aug.reshape(aug_factor, -1, TSP_SIZE, 6)[:,
                             episode: episode + batch_s // aug_factor].reshape(batch_s, TSP_SIZE, 6)

            ref = REF[None, None, :].repeat(batch_s // aug_factor, 1, 1).float().cuda()  # ref point
            ref_mask = torch.zeros(batch_s // aug_factor, ref.shape[1])
            sols = test_sols_aug[batch_i]
            sols_mask = test_sols_mask_aug[batch_i]
            if sols.shape[1] > NEIGHBOR:
                ws = (pref * sols).sum(dim=2)
                ind = torch.topk(ws, NEIGHBOR, dim=-1, largest=False).indices
                sel = sols.gather(1, ind[:, :, None].repeat(1, 1, 3))
                sel_mask = sols_mask.gather(1, ind)
            else:
                sel = sols
                sel_mask = sols_mask
            sel = torch.cat((ref, sel), dim=1)
            sel_mask = torch.cat((ref_mask, sel_mask), dim=-1)
            sel_mask_pomo = sel_mask[:, None, :].repeat(1, self.pomo_size, 1)
            sel_aug = sel[None, :, :, :].repeat(aug_factor, 1, 1, 1).reshape(batch_s, -1, 3)  # aug
            sel_mask_aug = sel_mask[None, :, :].repeat(aug_factor, 1, 1).reshape(batch_s, -1)  # aug
            sel_mask_pomo_aug = sel_mask_pomo[None, :, :, :].repeat(aug_factor, 1, 1, 1).reshape(
                batch_s, self.pomo_size, -1)  # aug

            with torch.no_grad():

                env = GROUP_ENVIRONMENT(testdata_batch)
                group_s = TSP_SIZE
                group_state, reward, done = env.reset(group_size=group_s)
                test_model_aug.reset(group_state, sel_aug, sel_mask_aug, sel_mask_pomo_aug)

                # First Move is given
                first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
                group_state, reward, done = env.step(first_action)

                while not done:
                    test_model_aug.update(group_state)
                    action_probs = test_model_aug.get_action_probabilities()
                    # shape = (batch, group, TSP_SIZE)
                    action = action_probs.argmax(dim=2)
                    # shape = (batch, group)
                    group_state, reward, done = env.step(action)

                # reward was negative, here we set it to positive
                reward = -reward
                reward = reward.reshape(aug_factor, -1, self.pomo_size, 3).permute(1, 0, 2, 3).reshape(
                    -1, aug_factor * self.pomo_size, 3)
                # update non-dominated solution set

                if reward.shape[1] > CANDIDATE:
                    ws_r = (pref * reward).sum(dim=2)
                    ind_r = torch.topk(ws_r, CANDIDATE, dim=-1, largest=False).indices
                    cdd = reward.gather(1, ind_r[:, :, None].repeat(1, 1, 3))
                else:
                    cdd = reward
                # cdd = reward
                if sols.shape[1] > NEIGHBOR:
                    split_flag = torch.zeros(batch_s // aug_factor, sols.shape[1])
                    split_flag = split_flag.scatter(1, ind, 1)
                    rest = sols[split_flag == 0].reshape(batch_s // aug_factor, -1, 3)
                    rest_mask = sols_mask[split_flag == 0].reshape(batch_s // aug_factor, -1)
                    sols, flag, NDS = update_EP(cdd, sel)
                    sols[sols == 1e4] = REF[0].float().cuda()
                    sols[sols == 1e3] = REF[1].float().cuda()
                    sols[sols == 1e5] = REF[2].float().cuda()
                    sols_mask = flag.float()
                    sols_mask[sols_mask == 1] = float('-inf')
                    sols = torch.cat((sols, rest), dim=1)
                    sols_mask = torch.cat((sols_mask, rest_mask), dim=1)
                else:
                    sols, flag, NDS = update_EP(cdd, sols)
                    sols[sols == 1e4] = REF[0].float().cuda()
                    sols[sols == 1e3] = REF[1].float().cuda()
                    sols[sols == 1e5] = REF[2].float().cuda()
                    sols_mask = flag.float()
                    sols_mask[sols_mask == 1] = float('-inf')
                test_sols_aug[batch_i] = sols
                test_sols_mask_aug[batch_i] = sols_mask
                test_NDS_aug[episode: episode + batch_s // aug_factor] = NDS

            batch_i += 1
            episode = episode + batch_s // aug_factor
            if episode == TEST_DATASET_SIZE:
                break

        return test_sols_aug, test_sols_mask_aug, test_NDS_aug

# Meta Model
meta_learner = Meta(actor)

def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

if MODE == 1:  # Train
    SAVE_FOLDER_NAME = 'TRAIN_' + METHOD + '_size{}'.format(TSP_SIZE)
    print(SAVE_FOLDER_NAME)

    # Make Log File
    # logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
    _, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

    # Save used HYPER_PARAMS
    hyper_param_filepath = './HYPER_PARAMS.py'
    hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(result_folder_path)
    shutil.copy(hyper_param_filepath, hyper_param_save_path)


    tb_logger = TbLogger('logs/TSP_' + METHOD + '_n{}_{}'.format(TSP_SIZE, time.strftime("%Y%m%dT%H%M%S")))

    start_epoch = 0
    if LOAD_PATH is not None:
        # checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint_fullname = LOAD_PATH
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        actor.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("Loaded trained_model")

    # save initial model
    checkpoint_dict = {
        'epoch': start_epoch,
        'model_state_dict': actor.state_dict()
    }
    torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(result_folder_path, start_epoch))
    print("Saved meta_model")

    # GO
    for epoch in range(start_epoch, TOTAL_EPOCH):
        meta_learner(epoch=epoch)
        if ((epoch + 1) % SAVE_INTVL) == 0:
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': meta_learner.model.state_dict()
            }
            torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(result_folder_path, epoch + 1))
            print("Saved meta_model")
        print('Ep:{}({}%)  T:{}'.format(epoch, epoch / TOTAL_EPOCH * 100,
                                        time.strftime("%H%M%S")))

elif MODE == 2:  # Test
    print('TEST_' + METHOD + '_size{}'.format(TSP_SIZE))
    model_dir = MODEL_DIR
    n_weight = N_WEIGHT
    # testdata = Tensor(np.random.rand(TEST_DATASET_SIZE, TSP_SIZE, 4))
    testdata = torch.load('../test_tsp_3o/testdata_tsp_3o_size{}.pt'.format(TSP_SIZE))
    testdata = testdata.to(device)
    test_save_ = 'test/' + METHOD + '_size{}-{}'.format(TSP_SIZE, time.strftime("%Y%m%d_%H%M"))
    if TSP_SIZE == 20:
        ref = np.array([20, 20, 20])
    elif TSP_SIZE == 50:
        ref = np.array([35, 35, 35])
    elif TSP_SIZE == 100:
        ref = np.array([65, 65, 65])
    else:
        raise NotImplementedError
    test_ep = TOTAL_EPOCH
    checkpoint_fullname = MODEL_DIR + '/checkpoint-{}.pt'.format(test_ep)
    test_save_dir = test_save_ + '/checkpoint-{}'.format(test_ep)
    os.makedirs(test_save_dir)

    # Save used HYPER_PARAMS
    hyper_param_filepath = './HYPER_PARAMS.py'
    hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(test_save_dir)
    shutil.copy(hyper_param_filepath, hyper_param_save_path)

    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    actor.load_state_dict(checkpoint['model_state_dict'])
    # start_epoch = checkpoint['epoch']
    print('Loaded meta-model-{}'.format(test_ep))
    finetune_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * FINETUNE_STEP, num_nodes=TSP_SIZE,
                                              batch_size=TRAIN_BATCH_SIZE)
    test_sols = []
    test_sols_mask = []
    test_sols_aug = []
    test_sols_mask_aug = []
    episode = 0
    while True:
        remaining = TEST_DATASET_SIZE - episode
        batch_s = min(TEST_BATCH_SIZE, remaining)

        test_sols.append(torch.empty(batch_s, 0, 3))
        test_sols_mask.append(torch.empty(batch_s, 0))
        episode = episode + batch_s
        if episode == TEST_DATASET_SIZE:
            break
    episode = 0
    while True:
        remaining = TEST_DATASET_SIZE - episode
        batch_s = min(TESTAUG_BATCH_SIZE, remaining)

        test_sols_aug.append(torch.empty(batch_s, 0, 3))
        test_sols_mask_aug.append(torch.empty(batch_s, 0))
        episode = episode + batch_s
        if episode == TEST_DATASET_SIZE:
            break
    train_sols = [torch.empty(TRAIN_BATCH_SIZE, 0, 3) for _ in range(FINETUNE_STEP)]
    train_sols_mask = [torch.empty(TRAIN_BATCH_SIZE, 0) for _ in range(FINETUNE_STEP)]

    if n_weight == 105:
        uniform_weights = torch.Tensor(das_dennis(13, 3))  # 105
    elif n_weight == 210:
        uniform_weights = torch.Tensor(das_dennis(19, 3))  # 210
    else:
        raise NotImplementedError

    pref = uniform_weights
    pref = pref[torch.randperm(pref.size(0))]
    total_test_time = 0
    total_test_time_aug = 0
    for i in range(n_weight):
        print('finetune and test, pref {}'.format(i))
        hv_w = torch.tensor([i / (n_weight - 1)])
        fine_model, train_sols, train_sols_mask = meta_learner.finetune(pref=pref[i], finetune_loader=finetune_loader, model=actor, hv_w=hv_w, train_sols=train_sols, train_sols_mask=train_sols_mask)
        test_timer_start = time.time()
        test_sols, test_sols_mask, test_NDS = meta_learner.test(model=fine_model, testdata=testdata, pref=pref[i], test_sols=test_sols, test_sols_mask=test_sols_mask)
        test_timer_end = time.time()
        total_test_time += test_timer_end - test_timer_start
        test_timer_start_aug = time.time()
        test_sols_aug, test_sols_mask_aug, test_NDS_aug = meta_learner.test_aug(model=fine_model, testdata=testdata, pref=pref[i], test_sols_aug=test_sols_aug, test_sols_mask_aug=test_sols_mask_aug)
        test_timer_end_aug = time.time()
        total_test_time_aug += test_timer_end_aug - test_timer_start_aug

    sols = torch.ones(TEST_DATASET_SIZE, 40000, 3) * 1000
    sols_aug = torch.ones(TEST_DATASET_SIZE, 40000, 3) * 1000
    episode = 0
    for t in test_sols:
        sols[episode: episode + t.shape[0], :t.shape[1]] = t
        episode = episode + t.shape[0]
    episode = 0
    for t in test_sols_aug:
        sols_aug[episode: episode + t.shape[0], :t.shape[1]] = t
        episode = episode + t.shape[0]
    p_sols, _, p_sols_num = update_EP(sols, None)
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=REF.numpy())
    p_sols_aug, _, p_sols_num_aug = update_EP(sols_aug, None)
    hvs_aug = cal_ps_hv(pf=p_sols_aug, pf_num=p_sols_num_aug, ref=REF.numpy())
    print('Test Time(s): {:.4f}'.format(total_test_time))
    print('HV Ratio: {:.4f}'.format(hvs.mean()))
    print('NDS: {:.4f}'.format(p_sols_num.float().mean().item()))
    print('Aug Test Time(s): {:.4f}'.format(total_test_time_aug))
    print('Aug HV Ratio: {:.4f}'.format(hvs_aug.mean()))
    print('Aug NDS: {:.4f}'.format(p_sols_num_aug.float().mean().item()))

    os.makedirs(os.path.join(test_save_dir, "sols"))
    os.makedirs(os.path.join(test_save_dir, "sols_aug"))
    for i in range(TEST_DATASET_SIZE):
        np.savetxt(os.path.join(test_save_dir, "sols", "ins{}.txt".format(i)), p_sols[i, :p_sols_num[i]].cpu().numpy(),
                   fmt='%1.4f\t%1.4f\t%1.4f', delimiter='\t')
        np.savetxt(os.path.join(test_save_dir, "sols_aug", "ins{}.txt".format(i)), p_sols_aug[i, :p_sols_num_aug[i]].cpu().numpy(),
                   fmt='%1.4f\t%1.4f\t%1.4f', delimiter='\t')
    print(MODEL_DIR)
    print('meta-model-{}'.format(test_ep))
    np.savetxt(os.path.join(test_save_dir, "all_hv.txt"), hvs, fmt='%1.4f', delimiter='\t')
    np.savetxt(os.path.join(test_save_dir, "all_hv_aug.txt"), hvs_aug, fmt='%1.4f', delimiter='\t')
    file = open(test_save_ + '/results.txt', 'w')
    file.write('HV Ratio: ' + str(hvs.mean()) + '\n')
    file.write('NDS: ' + str(p_sols_num.float().mean().item()) + '\n')
    file.write('Test Time(s): ' + str(total_test_time) + '\n')
    file.write('Aug HV Ratio: ' + str(hvs_aug.mean()) + '\n')
    file.write('Aug Test Time(s): ' + str(p_sols_num_aug.float().mean().item()) + '\n')
    file.write('Aug Test Time(s): ' + str(total_test_time_aug) + '\n')
