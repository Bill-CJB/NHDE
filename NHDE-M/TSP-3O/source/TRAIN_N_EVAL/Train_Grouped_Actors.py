
"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import time

# For debugging
from IPython.core.debugger import set_trace

# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

from source.utilities import Average_Meter
from source.mo_travelling_saleman_problem import TSP_DATA_LOADER__RANDOM, GROUP_ENVIRONMENT

from tensorboard_logger import Logger as TbLogger
from tqdm import tqdm


########################################
# TRAIN
########################################

# def TRAIN(actor_group, epoch, timer_start, logger):
def TRAIN(actor_group, epoch):

    actor_group.train()

    distance_AM = Average_Meter()
    actor_loss_AM = Average_Meter()

    train_loader = TSP_DATA_LOADER__RANDOM(num_sample=TRAIN_DATASET_SIZE, num_nodes=TSP_SIZE, batch_size=TRAIN_BATCH_SIZE)

    tb_logger = TbLogger('logs/TSP_n{}_{}'.format(TSP_SIZE, time.strftime("%Y%m%dT%H%M%S")))

    # logger_start = time.time()
    step = (epoch - 1) * (TRAIN_DATASET_SIZE // TRAIN_BATCH_SIZE + 1)
    episode = 0
    for data in tqdm(train_loader):
        # data.shape = (batch_s, TSP_SIZE, 2)

        batch_s = data.size(0)
        episode = episode + batch_s

        # Actor Group Move
        ###############################################
        env = GROUP_ENVIRONMENT(data)
        group_s = TSP_SIZE
        group_state, reward, done = env.reset(group_size=group_s)
        actor_group.reset(group_state)

        # First Move is given
        first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
        group_state, reward, done = env.step(first_action)

        group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
        while not done:
            actor_group.update(group_state)
            action_probs = actor_group.get_action_probabilities()
            # shape = (batch, group, TSP_SIZE)
            action = action_probs.reshape(batch_s*group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s)
            # shape = (batch, group)
            group_state, reward, done = env.step(action)

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

        # LEARNING - Actor
        ###############################################
        group_reward = reward
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()

        actor_group.optimizer.zero_grad()
        loss.backward()
        actor_group.optimizer.step()

        # RECORDING
        ###############################################
        max_reward, _ = group_reward.max(dim=1)
        distance_AM.push(-max_reward)  # reward was given as negative dist
        actor_loss_AM.push(group_loss.detach().reshape(-1))

        # LOGGING
        ###############################################
        obj_avg = distance_AM.result()
        loss_avg = actor_loss_AM.result()
        step = step + 1
        tb_logger.log_value('obj', obj_avg, step)
        tb_logger.log_value('loss', loss_avg, step)

        # if (time.time()-logger_start > LOG_PERIOD_SEC) or (episode == TRAIN_DATASET_SIZE):
        #     timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
        #     log_str = 'Ep:{:03d}-{:07d}({:5.1f}%)  T:{:s}  ALoss:{:+5f}  CLoss:{:5f}  Avg.dist:{:5f}' \
        #         .format(epoch, episode, episode/TRAIN_DATASET_SIZE*100,
        #                 timestr, actor_loss_AM.result(), 0,
        #                 distance_AM.result())
        #     logger.info(log_str)
        #     logger_start = time.time()
    print('Ep:{}({}%)  T:{}  ALoss:{}  Avg.dist:{}'.format(epoch, epoch / TOTAL_EPOCH * 100,
                                                              time.strftime("%H%M%S"), loss_avg, obj_avg))

    # LR STEP, after each epoch
    # actor_group.lr_stepper.step()

