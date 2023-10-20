##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 7


##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils import create_logger, copy_all_src
from cal_ps_hv import cal_ps_hv
from update_PE6 import *

from MOTSPTester_3obj import TSPTester as Tester
from MOTSProblemDef_3obj import get_random_problems

##########################################################################################
import time
import hvwfg

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'ref': torch.tensor([20, 20, 20])
}

model_params = {
    'node_size': 20,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'hyper_hidden_dim': 256,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'neighbor': 20,
    'candidate': 200,
    "dec_method": "WS",
    'model_load': {
        'path': './result/train__tsp_n20',
        'info': "MOTSP_3obj_20",
        'epoch': 200,
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    # 'aug_factor': 1,
    'aug_factor': 128,
    # 'aug_factor': 512,
    'aug_batch_size': 200,

    'n_sols': 210
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


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

##########################################################################################
time_start = time.time()
logger_start = time.time()

def main(n_sols = tester_params['n_sols']):
    if DEBUG_MODE:
        _set_debug_mode()

    if tester_params['aug_factor'] == 1:
        sols_floder = f"PMOCO_mean_sols_n{env_params['problem_size']}.txt"
        sols2_floder = f"PMOCO_mean_sols2_n{env_params['problem_size']}.txt"
        pareto_fig = f"PMOCO_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"PMOCO_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"PMOCO_hv_n{env_params['problem_size']}.txt"
    else:
        sols_floder = f"PMOCO(aug)_mean_sols_n{env_params['problem_size']}.txt"
        sols2_floder = f"PMOCO(aug)_mean_sols2_n{env_params['problem_size']}.txt"
        pareto_fig = f"PMOCO(aug)_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"PMOCO(aug)_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"PMOCO(aug)_hv_n{env_params['problem_size']}.txt"


    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)


    test_path = f"./data/testdata_tsp_3o_size{env_params['problem_size']}.pt"
    shared_problem = torch.load(test_path).to(device=CUDA_DEVICE_NUM)
    # shared_problem = get_random_problems(tester_params['test_episodes'], env_params['problem_size'])

    if n_sols == 105:
        uniform_weights = torch.Tensor(das_dennis(13, 3))  # 105
    elif n_sols == 210:
        uniform_weights = torch.Tensor(das_dennis(19, 3))  # 210
    else:
        raise NotImplementedError


    batch_size = shared_problem.shape[0]
    sols = torch.ones([batch_size, 20000, 3]) * 1000
    pref = uniform_weights.cuda()
    # shuffle
    pref = pref[torch.randperm(pref.size(0))]

    mini_batch_size = tester_params['test_batch_size']
    b_cnt = tester_params['test_episodes'] / mini_batch_size
    b_cnt = int(b_cnt)
    total_test_time = 0
    sols_len = 0
    for bi in range(0, b_cnt):
        b_start = bi * mini_batch_size
        b_end = b_start + mini_batch_size
        test_timer_start = time.time()
        sols_ = tester.run(shared_problem, pref, episode=b_start)
        test_timer_end = time.time()
        total_test_time += test_timer_end - test_timer_start

        sols[b_start:b_end, :sols_.shape[1], :] = sols_
        sols_len = max(sols_len, sols_.shape[1])

    sols = sols[:, :sols_len, :]
    print('Avg Test Time(s): {:.4f}\n'.format(total_test_time))


    max_obj1 = sols.reshape(-1, 3)[:, 0].max()
    max_obj2 = sols.reshape(-1, 3)[:, 1].max()
    max_obj3 = sols.reshape(-1, 3)[:, 2].max()
    txt2 = F"{tester.result_folder}/max_cost_n{env_params['problem_size']}.txt"
    f = open(
        txt2,
        'a')
    f.write(f"MAX OBJ1:{max_obj1}\n")
    f.write(f"MAX OBJ2:{max_obj2}\n")
    f.write(f"MAX OBJ3:{max_obj3}\n")
    f.close()

    sols_mean = sols.mean(0).cpu()

    np.savetxt(F"{tester.result_folder}/{sols_floder}", sols_mean,
               delimiter='\t', fmt="%.4f\t%.4f\t%.4f")
    

    ref = np.array([20,20,20])    #20
    #ref = np.array([35,35,35])   #50
    #ref = np.array([65,65,65])   #100


    p_sols, _, p_sols_num = update_EP(sols, None)
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref)

    print('HV Ratio: {:.4f}'.format(hvs.mean()))
    print('NDS: {:.4f}'.format(p_sols_num.float().mean()))

    os.makedirs(tester.result_folder + '/sols')
    for i in range(p_sols.shape[0]):
        np.savetxt(F"{tester.result_folder}/sols/ins{i}.txt", p_sols[i, :p_sols_num[i]].cpu().numpy(),
                   delimiter='\t', fmt="%.4f\t%.4f\t%.4f")
    # np.savetxt(F"{result_folder}/{all_sols_floder}", sols.reshape(-1, 2),
    #            delimiter='\t', fmt="%.4f\t%.4f")
    np.savetxt(F"{tester.result_folder}/{hv_floder}", hvs,
               delimiter='\t', fmt="%.4f")

    if tester_params['aug_factor'] == 1:
        f = open(
            F"{tester.result_folder}/PMOCO-TSP_3obj{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO-TSP_3obj{env_params['problem_size']}\n")
    else:
        f = open(
            F"{tester.result_folder}/PMOCO(aug)-TSP_3obj{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO(aug)-TSP_3obj{env_params['problem_size']}\n")


    f.write(f"MOTSP_3obj Type1\n")
    f.write(f"Model Path: {tester_params['model_load']['path']}\n")
    f.write(f"Model Epoch: {tester_params['model_load']['epoch']}\n")
    f.write(f"Neighbor Num: {tester_params['neighbor']}\n")
    f.write(f"Pref Num: {n_sols}\n")
    f.write(f"Hyper Hidden Dim: {model_params['hyper_hidden_dim']}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Aug Factor: {tester_params['aug_factor']}\n")
    f.write('Test Time(s): {:.4f}\n'.format(total_test_time))
    f.write('HV Ratio: {:.4f}\n'.format(hvs.mean()))
    f.write('NDS: {:.4f}\n'.format(p_sols_num.float().mean()))
    f.write(f"Ref Point:[{ref[0]},{ref[1]},{ref[2]}] \n")
    f.write(f"Info: {tester_params['model_load']['info']}\n")
    # f.write(f"{compare_type}_{optim} avg_hv:{avg_hvs} s\n")
    f.close()



##########################################################################################

if __name__ == "__main__":
    main()
