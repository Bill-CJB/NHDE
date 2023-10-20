import torch

import os
from logging import getLogger

from MOTSPEnv_3obj import TSPEnv as Env
from MOTSPModel_3obj import TSPModel as Model

from MOTSProblemDef_3obj import get_random_problems, augment_xy_data_by_n_fold_3obj, augment_xy_data_by_128_fold_3obj

# from einops import rearrange
import math
import torch.nn.functional as F
from update_PE6 import *
from utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, shared_problem, pref, episode=0):
        self.time_estimator.reset()

        remaining = self.tester_params['test_episodes'] - episode
        batch_size = min(self.tester_params['test_batch_size'], remaining)
        sols = self._test_one_batch(shared_problem, pref, batch_size, episode)
        return sols
              
    def _test_one_batch(self, shared_probelm, pref, batch_size, episode):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
            
        self.env.batch_size = batch_size 
        self.env.problems = shared_probelm[episode: episode + batch_size]
        
        if aug_factor == 512:
            self.env.batch_size = self.env.batch_size * aug_factor
            self.env.problems = augment_xy_data_by_n_fold_3obj(self.env.problems, aug_factor)
        elif aug_factor == 128:
            self.env.batch_size = self.env.batch_size * aug_factor
            self.env.problems = augment_xy_data_by_128_fold_3obj(self.env.problems)
            
        self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
        self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)

        ref = self.env_params['ref'][None, None, :].repeat(batch_size, 1, 1).float().cuda()  # ref point
        ref_mask = torch.zeros(batch_size, ref.shape[1])
        sols = torch.empty(batch_size, 0, 3)
        sols_mask = torch.empty(batch_size, 0)

        self.model.eval()
        for st in range(pref.shape[0]):
            hv_w = torch.tensor([st / (pref.shape[0] - 1)])
            print('ins:', episode + batch_size, 'pref num:', st, 'hv_w: {:.2f}'.format(hv_w.item()))
            if sols.shape[1] > self.tester_params['neighbor']:
                ws = (pref[st] * sols).sum(dim=2)
                ind = torch.topk(ws, self.tester_params['neighbor'], dim=-1, largest=False).indices
                sel = sols.gather(1, ind[:, :, None].repeat(1, 1, 3))
                sel_mask = sols_mask.gather(1, ind)
            else:
                sel = sols
                sel_mask = sols_mask
            sel = torch.cat((ref, sel), dim=1)
            sel_mask = torch.cat((ref_mask, sel_mask), dim=-1)
            sel_mask_pomo = sel_mask[:, None, :].repeat(1, self.env.pomo_size, 1)
            sel_aug = sel[None, :, :, :].repeat(aug_factor, 1, 1, 1).reshape(aug_factor * batch_size, -1, 3)  # aug
            sel_mask_aug = sel_mask[None, :, :].repeat(aug_factor, 1, 1).reshape(aug_factor * batch_size, -1)  # aug
            sel_mask_pomo_aug = sel_mask_pomo[None, :, :, :].repeat(aug_factor, 1, 1, 1).reshape(
                aug_factor * batch_size, self.env.pomo_size, -1)  # aug

            with torch.no_grad():
                reset_state, _, _ = self.env.reset()

                self.model.decoder.assign(torch.cat((pref[st], 1 - hv_w, hv_w), dim=-1).float())
                self.model.pre_forward(reset_state, sel_aug, sel_mask_aug)

            state, reward, done = self.env.pre_step()

            while not done:
                selected, _ = self.model(state, sel_mask_pomo_aug)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)

            # reward was negative, here we set it to positive objective value
            reward = - reward
            # aug
            reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size, 3).permute(1, 0, 2, 3).reshape(
                batch_size, aug_factor * self.env.pomo_size, 3)

            # update non-dominated solution set and hv

            if reward.shape[1] > self.tester_params['candidate']:
                ws_r = (pref[st] * reward).sum(dim=2)
                ind_r = torch.topk(ws_r, self.tester_params['candidate'], dim=-1, largest=False).indices
                cdd = reward.gather(1, ind_r[:, :, None].repeat(1, 1, 3))
            else:
                cdd = reward
            # cdd = reward
            if sols.shape[1] > self.tester_params['neighbor']:
                split_flag = torch.zeros(batch_size, sols.shape[1])
                split_flag = split_flag.scatter(1, ind, 1)
                rest = sols[split_flag == 0].reshape(batch_size, -1, 3)
                rest_mask = sols_mask[split_flag == 0].reshape(batch_size, -1)
                sols, flag, NDS = update_EP(cdd, sel)
                sols[sols == 1e4] = self.env_params['ref'][0].float().cuda()
                sols[sols == 1e3] = self.env_params['ref'][1].float().cuda()
                sols[sols == 1e5] = self.env_params['ref'][2].float().cuda()
                sols_mask = flag.float()
                sols_mask[sols_mask == 1] = float('-inf')
                sols = torch.cat((sols, rest), dim=1)
                sols_mask = torch.cat((sols_mask, rest_mask), dim=1)
            else:
                sols, flag, NDS = update_EP(cdd, sols)
                sols[sols == 1e4] = self.env_params['ref'][0].float().cuda()
                sols[sols == 1e3] = self.env_params['ref'][1].float().cuda()
                sols[sols == 1e5] = self.env_params['ref'][2].float().cuda()
                sols_mask = flag.float()
                sols_mask[sols_mask == 1] = float('-inf')

        return sols  # except ref
