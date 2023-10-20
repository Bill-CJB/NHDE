import torch
from logging import getLogger
import numpy
from MOTSPEnv import TSPEnv as Env
from MOTSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import torch.nn.functional as F
import hvwfg
from pygmo import hypervolume
from utils import *
from update_PE6 import *

class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        # self.aug_factor = env_params['aug_factor']

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
       
            if epoch == self.start_epoch or all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint_motsp-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
    
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_hv, avg_loss, avg_nds = self._train_one_batch(batch_size)
            score_AM.update(avg_hv, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  HV Score: {:.4f},  NDS: {:.4f},  Loss: {:.4f}'
                             .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                     avg_hv, avg_nds, avg_loss))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        ref = self.env_params['ref'][None, None, :].repeat(batch_size, 1, 1).float().cuda()  # ref point
        ref_mask = torch.zeros(batch_size, ref.shape[1])
        sols = torch.empty(batch_size, 0, 2)
        sols_mask = torch.empty(batch_size, 0)
        hv_reward = torch.zeros(batch_size, self.env.pomo_size)

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)

        for st in range(self.trainer_params['solving_times']):
            # hv_w = (torch.rand(1) + st) / self.trainer_params['solving_times']
            hv_w = torch.rand(1)
            alpha = 1
            pref = numpy.random.dirichlet((alpha, alpha), 1)
            pref = torch.tensor(pref[0])
            if sols.shape[1] > self.trainer_params['neighbor']:
                ws = (pref * sols).sum(dim=2)
                ind = torch.topk(ws, self.trainer_params['neighbor'], dim=-1, largest=False).indices
                sel = sols.gather(1, ind[:, :, None].repeat(1, 1, 2))
                sel_mask = sols_mask.gather(1, ind)
            else:
                sel = sols
                sel_mask = sols_mask
            sel = torch.cat((ref, sel), dim=1)
            sel_mask = torch.cat((ref_mask, sel_mask), dim=-1)
            sel_mask_pomo = sel_mask[:, None, :].repeat(1, self.env.pomo_size, 1)

            reset_state, _, _ = self.env.reset()

            self.model.decoder.assign(torch.cat((pref, 1 - hv_w, hv_w), dim=-1).float())
            self.model.pre_forward(reset_state, sel, sel_mask)

            prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()

            while not done:
                selected, prob = self.model(state, sel_mask_pomo)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            # Loss
            ###############################################
            # reward was negative, here we set it to positive to calculate TCH
            reward = - reward
            if self.trainer_params['dec_method'] == "WS":
                tch_reward = (pref * reward).sum(dim=2)
            elif self.trainer_params['dec_method'] == "TCH":
                z = torch.ones(reward.shape).cuda() * 0.0
                tch_reward = pref * (reward - z)
                tch_reward, _ = tch_reward.max(dim=2)
            else:
                return NotImplementedError

            # HV reward
            s_ = sel[:, 1:, :].clone()
            s_[s_[:, :, 0] >= self.env_params['ref'][0]] = self.env_params['ref'][0].cuda() - 1e-4
            s_[s_[:, :, 1] >= self.env_params['ref'][1]] = self.env_params['ref'][1].cuda() - 1e-4
            r_ = reward.clone()
            r_[r_[:, :, 0] >= self.env_params['ref'][0]] = self.env_params['ref'][0].cuda() - 1e-4
            r_[r_[:, :, 1] >= self.env_params['ref'][1]] = self.env_params['ref'][1].cuda() - 1e-4

            for b_i in range(batch_size):
                for i_i in range(self.env.pomo_size):
                    hv_reward[b_i, i_i] = hypervolume(
                        torch.cat((s_[b_i, :, :], r_[b_i, i_i, None]), dim=0).cpu().numpy().astype(
                            float)).compute(self.env_params['ref'].numpy()) / (
                                                      self.env_params['ref'][0] * self.env_params['ref'][1])
            # update non-dominated solution set
            sols, flag, NDS = update_EP(reward, sols)
            sols[sols == 1e4] = self.env_params['ref'][0].float().cuda()
            sols[sols == 1e3] = self.env_params['ref'][1].float().cuda()
            sols_mask = flag.float()
            sols_mask[sols_mask == 1] = float('-inf')

            # set back reward to negative
            reward = -reward
            # tch_reward = -tch_reward
            tch_reward = -tch_reward * (1 - hv_w) + hv_reward * hv_w

            # log_prob = prob_list.log().sum(dim=2).reshape(batch_size, -1)
            log_prob = prob_list.log().sum(dim=2)

            # shape = (batch, group)

            tch_advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)

            tch_loss = -tch_advantage * log_prob # Minus Sign
            # shape = (batch, group)
            loss_mean = tch_loss.mean()

            #Step & Return
            ################################################
            self.model.zero_grad()
            loss_mean.backward()
            self.optimizer.step()

        hv = torch.zeros(batch_size)
        for b_i in range(batch_size):
            hv[b_i] = hvwfg.wfg(sols[b_i][:NDS[b_i]].cpu().numpy().astype(float),
                                self.env_params['ref'].numpy().astype(float)) / (
                                               self.env_params['ref'][0] * self.env_params['ref'][1])

        return hv.mean().item(), loss_mean.item(), NDS.float().mean().item()