
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

# For Logging
import time

# For debugging
from IPython.core.debugger import set_trace

# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

from source.utilities import Average_Meter
from source.mo_knapsack_problem import KNAPSACK_DATA_LOADER__RANDOM, GROUP_ENVIRONMENT


########################################
# TRAIN
########################################

def TRAIN(grouped_actor, epoch, timer_start, logger):

    grouped_actor.train()

    score_AM = Average_Meter()
    actor_loss_AM = Average_Meter()
    train_loader = KNAPSACK_DATA_LOADER__RANDOM(num_sample=TRAIN_DATASET_SIZE,
                                                num_items=PROBLEM_SIZE,
                                                batch_size=BATCH_SIZE)

    logger_start = time.time()
    episode = 0
    for item_data in train_loader:
        # item_data.shape = (batch, problem, 2)

        batch_s = item_data.size(0)
        episode = episode + batch_s

        # Actor Group Move
        ###############################################
        env = GROUP_ENVIRONMENT(item_data)
        group_s = PROBLEM_SIZE
        group_state, reward, done = env.reset(group_size=group_s)
        grouped_actor.reset(group_state)

        # First Move is given
        first_action = LongTensor(np.arange(group_s))[None, :].expand(batch_s, group_s)
        group_state, reward, done = env.step(first_action)

        group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
        while not done:
            action_probs = grouped_actor.get_action_probabilities(group_state)
            # shape = (batch, group, problem)
            action = action_probs.reshape(batch_s*group_s, PROBLEM_SIZE).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s)
            # shape = (batch, group)

            action_w_finisehd = action.clone()
            action_w_finisehd[group_state.finished] = PROBLEM_SIZE  # dummy item
            group_state, reward, done = env.step(action_w_finisehd)

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)
            chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)
            # shape = (batch, group, x)

        # LEARNING - Actor
        ###############################################
        group_reward = reward
        # shape = (batch, group)
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()

        grouped_actor.optimizer.zero_grad()
        loss.backward()
        grouped_actor.optimizer.step()

        # RECORDING
        ###############################################
        max_reward, _ = group_reward.max(dim=1)
        score_AM.push(max_reward)
        actor_loss_AM.push(group_loss.detach())

        # LOGGING
        ###############################################
        if (time.time()-logger_start > LOG_PERIOD_SEC) or (episode == TRAIN_DATASET_SIZE):
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
            log_str = 'Ep:{:03d}-{:07d}({:5.1f}%)  T:{:s}  ALoss:{:+5f}  Avg.dist:{:5f}' \
                .format(epoch, episode, episode/TRAIN_DATASET_SIZE*100,
                        timestr, actor_loss_AM.result(),
                        score_AM.result())
            logger.info(log_str)
            logger_start = time.time()

    # LR STEP, after each epoch
    grouped_actor.lr_stepper.step()

