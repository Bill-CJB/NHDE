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
from source.utilities import Get_Logger, Average_Meter
from matplotlib import pyplot as plt
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
from source.mo_knapsack_problem import KNAPSACK_DATA_LOADER__RANDOM, GROUP_ENVIRONMENT

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
        self.task_num = TASK_NUM
        self.pomo_size = PROBLEM_SIZE
        self.model = deepcopy(model)
        self.submodel = deepcopy(model)
        self.suboptimizer = optim.Adam(self.submodel.parameters(), lr=ACTOR_LEARNING_RATE)

    def forward(self, epoch=0):
        support_loader = KNAPSACK_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * UPDATE_STEP, num_items=PROBLEM_SIZE, batch_size=TRAIN_BATCH_SIZE)
        batch_s = TRAIN_BATCH_SIZE
        alpha = 1
        train_sols = [torch.empty(batch_s, 0, 2) for _ in range(UPDATE_STEP)]
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
                pref = np.random.dirichlet((alpha, alpha), None)
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
                        ind = torch.topk(ws, NEIGHBOR, dim=-1, largest=True).indices
                        sel = sols.gather(1, ind[:, :, None].repeat(1, 1, 2))
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
                    group_s = PROBLEM_SIZE
                    group_state, reward, done = env.reset(group_size=group_s)
                    self.submodel.reset(group_state, sel, sel_mask)

                    # First Move is given
                    first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
                    group_state, reward, done = env.step(first_action)

                    group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
                    while not done:
                        action_probs = self.submodel.get_action_probabilities(group_state, sel_mask_pomo)
                        # shape = (batch, group, TSP_SIZE)
                        action = action_probs.reshape(batch_s * group_s, PROBLEM_SIZE).multinomial(1).squeeze(
                            dim=1).reshape(
                            batch_s, group_s)
                        # shape = (batch, group)
                        action_w_finisehd = action.clone()
                        action_w_finisehd[group_state.finished] = PROBLEM_SIZE  # dummy item
                        group_state, reward, done = env.step(action_w_finisehd)

                        batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
                        group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                        chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s,
                                                                                                        group_s)
                        # shape = (batch, group)
                        chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
                        group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

                    # LEARNING - Actor
                    ###############################################
                    assert AGG == 1 or AGG == 2, "Only support Weighted-Sum and Weighted-Tchebycheff"
                    # KP is to maximize the reward
                    if AGG == 1:
                        ws_reward = (pref * reward).sum(dim=-1)
                        agg_reward = ws_reward
                    elif AGG == 2:
                        z = torch.ones(reward.shape).cuda() * 80.0
                        tch_reward = pref * (reward - z)
                        tch_reward, _ = tch_reward.max(dim=-1)

                        agg_reward = -tch_reward
                    else:
                        return NotImplementedError

                    # HV reward
                    s_ = sel[:, 1:, :].clone()
                    s_[s_[:, :, 0] <= REF[0]] = REF[0].cuda() + 1e-4
                    s_[s_[:, :, 1] <= REF[1]] = REF[1].cuda() + 1e-4
                    r_ = reward.clone()
                    r_[r_[:, :, 0] <= REF[0]] = REF[0].cuda() + 1e-4
                    r_[r_[:, :, 1] <= REF[1]] = REF[1].cuda() + 1e-4

                    for b_i in range(batch_s):
                        for i_i in range(self.pomo_size):
                            hv_reward[b_i, i_i] = hypervolume(
                                -torch.cat((s_[b_i, :, :], r_[b_i, i_i, None]), dim=0).cpu().numpy().astype(
                                    float)).compute(-REF.numpy()) / ((IDEAL[0] - REF[0]) * (IDEAL[1] - REF[1]))
                    # update non-dominated solution set
                    sols, flag, NDS = update_EP(-reward, -sols)
                    sols[sols == 1e4] = -REF[0].float().cuda()
                    sols[sols == 1e3] = -REF[1].float().cuda()
                    sols = -sols
                    sols_mask = flag.float()
                    sols_mask[sols_mask == 1] = float('-inf')
                    train_sols[batch_i] = sols
                    train_sols_mask[batch_i] = sols_mask


                    group_reward = agg_reward * (1 - hv_w) + hv_reward * hv_w
                    group_log_prob = group_prob_list.log().sum(dim=2)
                    # shape = (batch, group)

                    group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

                    group_loss = -group_advantage * group_log_prob
                    loss = group_loss.mean()

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
            hv[b_i] = hvwfg.wfg(-sols[b_i][:NDS[b_i]].cpu().numpy().astype(float),
                                -REF.numpy().astype(float)) / ((IDEAL[0] - REF[0]) * (IDEAL[1] - REF[1]))
        print('hv:', hv.mean().item(), 'NDS:', NDS.float().mean().item(), 'loss:', loss.item())
        return

    def finetune(self, pref=None, finetune_loader=None, model=None, hv_w=None, train_sols=None, train_sols_mask=None):
        if pref is None:
            pref = torch.tensor([0.5, 0.5])
        if finetune_loader is None:
            finetune_loader = KNAPSACK_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * FINETUNE_STEP, num_items=PROBLEM_SIZE, batch_size=TRAIN_BATCH_SIZE)
        if model is None:
            fine_model = deepcopy(self.model)
        else:
            fine_model = deepcopy(model)
        print("--------------------------------------------")
        print("pref:{}, {}, hv_w:{}".format(pref[0].item(), pref[1].item(), hv_w.item()))
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
                ind = torch.topk(ws, NEIGHBOR, dim=-1, largest=True).indices
                sel = sols.gather(1, ind[:, :, None].repeat(1, 1, 2))
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
            group_s = PROBLEM_SIZE
            group_state, reward, done = env.reset(group_size=group_s)
            fine_model.reset(group_state, sel, sel_mask)

            # First Move is given
            first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
            group_state, reward, done = env.step(first_action)

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                action_probs = fine_model.get_action_probabilities(group_state, sel_mask_pomo)
                # shape = (batch, group, TSP_SIZE)
                action = action_probs.reshape(batch_s * group_s, PROBLEM_SIZE).multinomial(1).squeeze(
                    dim=1).reshape(
                    batch_s, group_s)
                # shape = (batch, group)
                action_w_finisehd = action.clone()
                action_w_finisehd[group_state.finished] = PROBLEM_SIZE  # dummy item
                group_state, reward, done = env.step(action_w_finisehd)

                batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s,
                                                                                                group_s)
                # shape = (batch, group)
                chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

            # LEARNING - Actor
            ###############################################
            assert AGG == 1 or AGG == 2, "Only support Weighted-Sum and Weighted-Tchebycheff"
            # KP is to maximize the reward
            if AGG == 1:
                ws_reward = (pref * reward).sum(dim=-1)
                agg_reward = ws_reward
            elif AGG == 2:
                z = torch.ones(reward.shape).cuda() * 80.0
                tch_reward = pref * (reward - z)
                tch_reward, _ = tch_reward.max(dim=-1)

                agg_reward = -tch_reward
            else:
                return NotImplementedError

            # HV reward
            s_ = sel[:, 1:, :].clone()
            s_[s_[:, :, 0] <= REF[0]] = REF[0].cuda() + 1e-4
            s_[s_[:, :, 1] <= REF[1]] = REF[1].cuda() + 1e-4
            r_ = reward.clone()
            r_[r_[:, :, 0] <= REF[0]] = REF[0].cuda() + 1e-4
            r_[r_[:, :, 1] <= REF[1]] = REF[1].cuda() + 1e-4

            for b_i in range(batch_s):
                for i_i in range(self.pomo_size):
                    hv_reward[b_i, i_i] = hypervolume(
                        -torch.cat((s_[b_i, :, :], r_[b_i, i_i, None]), dim=0).cpu().numpy().astype(
                            float)).compute(-REF.numpy()) / ((IDEAL[0] - REF[0]) * (IDEAL[1] - REF[1]))
            # update non-dominated solution set
            if reward.shape[1] > CANDIDATE:
                ws_r = (pref * reward).sum(dim=2)
                ind_r = torch.topk(ws_r, CANDIDATE, dim=-1, largest=True).indices
                cdd = reward.gather(1, ind_r[:, :, None].repeat(1, 1, 2))
            else:
                cdd = reward
            # cdd = reward
            if sols.shape[1] > NEIGHBOR:
                split_flag = torch.zeros(batch_s, sols.shape[1])
                split_flag = split_flag.scatter(1, ind, 1)
                rest = sols[split_flag == 0].reshape(batch_s, -1, 2)
                rest_mask = sols_mask[split_flag == 0].reshape(batch_s, -1)
                sols, flag, NDS = update_EP(-cdd, -sel)
                sols[sols == 1e4] = -REF[0].float().cuda()
                sols[sols == 1e3] = -REF[1].float().cuda()
                sols = -sols
                sols_mask = flag.float()
                sols_mask[sols_mask == 1] = float('-inf')
                sols = torch.cat((sols, rest), dim=1)
                sols_mask = torch.cat((sols_mask, rest_mask), dim=1)
            else:
                sols, flag, NDS = update_EP(-cdd, -sols)
                sols[sols == 1e4] = -REF[0].float().cuda()
                sols[sols == 1e3] = -REF[1].float().cuda()
                sols = -sols
                sols_mask = flag.float()
                sols_mask[sols_mask == 1] = float('-inf')
            train_sols[step] = sols
            train_sols_mask[step] = sols_mask

            group_reward = agg_reward * (1 - hv_w) + hv_reward * hv_w
            group_log_prob = group_prob_list.log().sum(dim=2)
            # shape = (batch, group)

            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

            group_loss = -group_advantage * group_log_prob
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
            testdata = Tensor(np.random.rand(TEST_BATCH_SIZE, PROBLEM_SIZE, 3))
        if pref is None:
            pref = torch.tensor([0.5, 0.5])
        print("testing meta_model...")


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
                ind = torch.topk(ws, NEIGHBOR, dim=-1, largest=True).indices
                sel = sols.gather(1, ind[:, :, None].repeat(1, 1, 2))
                sel_mask = sols_mask.gather(1, ind)
            else:
                sel = sols
                sel_mask = sols_mask
            sel = torch.cat((ref, sel), dim=1)
            sel_mask = torch.cat((ref_mask, sel_mask), dim=-1)
            sel_mask_pomo = sel_mask[:, None, :].repeat(1, self.pomo_size, 1)

            with torch.no_grad():

                env = GROUP_ENVIRONMENT(testdata_batch)
                group_s = PROBLEM_SIZE
                group_state, reward, done = env.reset(group_size=group_s)
                test_model.reset(group_state, sel, sel_mask)

                # First Move is given
                first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
                group_state, reward, done = env.step(first_action)

                while not done:
                    action_probs = test_model.get_action_probabilities(group_state, sel_mask_pomo)
                    # shape = (batch, group, problem)
                    action = action_probs.argmax(dim=2)
                    # shape = (batch, group)

                    action_w_finished = action.clone()
                    action_w_finished[group_state.finished] = PROBLEM_SIZE  # this is dummy item with 0 size 0 value
                    group_state, reward, done = env.step(action_w_finished)


                if reward.shape[1] > CANDIDATE:
                    ws_r = (pref * reward).sum(dim=2)
                    ind_r = torch.topk(ws_r, CANDIDATE, dim=-1, largest=True).indices
                    cdd = reward.gather(1, ind_r[:, :, None].repeat(1, 1, 2))
                else:
                    cdd = reward
                # cdd = reward
                if sols.shape[1] > NEIGHBOR:
                    split_flag = torch.zeros(batch_s, sols.shape[1])
                    split_flag = split_flag.scatter(1, ind, 1)
                    rest = sols[split_flag == 0].reshape(batch_s, -1, 2)
                    rest_mask = sols_mask[split_flag == 0].reshape(batch_s, -1)
                    sols, flag, NDS = update_EP(-cdd, -sel)
                    sols[sols == 1e4] = -REF[0].float().cuda()
                    sols[sols == 1e3] = -REF[1].float().cuda()
                    sols = -sols
                    sols_mask = flag.float()
                    sols_mask[sols_mask == 1] = float('-inf')
                    sols = torch.cat((sols, rest), dim=1)
                    sols_mask = torch.cat((sols_mask, rest_mask), dim=1)
                else:
                    sols, flag, NDS = update_EP(-cdd, -sols)
                    sols[sols == 1e4] = -REF[0].float().cuda()
                    sols[sols == 1e3] = -REF[1].float().cuda()
                    sols = -sols
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

# Meta Model
meta_learner = Meta(actor)

if MODE == 1:  # Train
    SAVE_FOLDER_NAME = 'TRAIN_' + METHOD + '_size{}'.format(PROBLEM_SIZE)
    print(SAVE_FOLDER_NAME)

    # Make Log File
    # logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
    _, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

    # Save used HYPER_PARAMS
    hyper_param_filepath = './HYPER_PARAMS.py'
    hyper_param_save_path = '{}/used_HYPER_PARAMS.txt'.format(result_folder_path)
    shutil.copy(hyper_param_filepath, hyper_param_save_path)

    tb_logger = TbLogger('logs/TSP_' + METHOD + '_n{}_{}'.format(PROBLEM_SIZE, time.strftime("%Y%m%dT%H%M%S")))

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
        if ((epoch + 1) % (TOTAL_EPOCH // SAVE_NUM)) == 0:
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': meta_learner.model.state_dict()
            }
            torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(result_folder_path, epoch + 1))
            print("Saved meta_model")

        print('Ep:{}({}%)  T:{}'.format(epoch, epoch / TOTAL_EPOCH * 100,
                                        time.strftime("%H%M%S")))

elif MODE == 2:  # Test
    print('TEST_' + METHOD + '_size{}'.format(PROBLEM_SIZE))
    model_dir = MODEL_DIR
    n_weight = N_WEIGHT
    # testdata = Tensor(np.random.rand(TEST_DATASET_SIZE, TSP_SIZE, 4))
    testdata = torch.load('../test_kp/testdata_kp_size{}.pt'.format(PROBLEM_SIZE))
    testdata = testdata.to(device)
    test_save_ = 'test/' + METHOD + '_size{}-{}'.format(PROBLEM_SIZE, time.strftime("%Y%m%d_%H%M"))
    weight = torch.zeros(2).cuda()
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
    finetune_loader = KNAPSACK_DATA_LOADER__RANDOM(num_sample=TRAIN_BATCH_SIZE * FINETUNE_STEP, num_items=PROBLEM_SIZE,
                                              batch_size=TRAIN_BATCH_SIZE)

    test_sols = []
    test_sols_mask = []
    episode = 0
    while True:
        remaining = TEST_DATASET_SIZE - episode
        batch_s = min(TEST_BATCH_SIZE, remaining)

        test_sols.append(torch.empty(batch_s, 0, 2))
        test_sols_mask.append(torch.empty(batch_s, 0))
        episode = episode + batch_s
        if episode == TEST_DATASET_SIZE:
            break
    train_sols = [torch.empty(TRAIN_BATCH_SIZE, 0, 2) for _ in range(FINETUNE_STEP)]
    train_sols_mask = [torch.empty(TRAIN_BATCH_SIZE, 0) for _ in range(FINETUNE_STEP)]
    pref = torch.zeros(n_weight, 2).cuda()
    # shuffle
    for i in range(n_weight):
        pref[i, 0] = 1 - i / (n_weight - 1)
        pref[i, 1] = i / (n_weight - 1)
    pref = pref[torch.randperm(pref.size(0))]
    total_test_time = 0
    for i in range(n_weight):
        print('finetune and test, pref {}'.format(i))
        hv_w = torch.tensor([i / (n_weight - 1)])
        fine_model, train_sols, train_sols_mask = meta_learner.finetune(pref=pref[i], finetune_loader=finetune_loader,
                                                                        model=actor, hv_w=hv_w, train_sols=train_sols,
                                                                        train_sols_mask=train_sols_mask)
        test_timer_start = time.time()
        test_sols, test_sols_mask, test_NDS = meta_learner.test(model=fine_model, testdata=testdata, pref=pref[i],
                                                                test_sols=test_sols, test_sols_mask=test_sols_mask)
        test_timer_end = time.time()
        total_test_time += test_timer_end - test_timer_start

    sols = torch.zeros(TEST_DATASET_SIZE, 1000, 2)
    episode = 0
    for t in test_sols:
        sols[episode: episode + t.shape[0], :t.shape[1]] = t
        episode = episode + t.shape[0]
    episode = 0
    p_sols, _, p_sols_num = update_EP(-sols, None)
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=-REF.numpy(), ideal=-IDEAL.numpy())
    print('Test Time(s): {:.4f}'.format(total_test_time))
    print('HV Ratio: {:.4f}'.format(hvs.mean()))
    print('NDS: {:.4f}'.format(p_sols_num.float().mean().item()))

    os.makedirs(os.path.join(test_save_dir, "sols"))
    p_sols = -p_sols
    for i in range(TEST_DATASET_SIZE):
        np.savetxt(os.path.join(test_save_dir, "sols", "ins{}.txt".format(i)), p_sols[i, :p_sols_num[i]].cpu().numpy(),
                   fmt='%1.4f\t%1.4f', delimiter='\t')
    print(MODEL_DIR)
    print('meta-model-{}'.format(test_ep))
    np.savetxt(os.path.join(test_save_dir, "all_hv.txt"), hvs, fmt='%1.4f', delimiter='\t')
    file = open(test_save_ + '/results.txt', 'w')
    file.write('HV Ratio: ' + str(hvs.mean()) + '\n')
    file.write('NDS: ' + str(p_sols_num.float().mean().item()) + '\n')
    file.write('Test Time(s): ' + str(total_test_time) + '\n')


