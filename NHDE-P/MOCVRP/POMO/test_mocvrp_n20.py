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
from MOCVRPTester import CVRPTester as Tester
from MOCVRProblemDef import get_random_problems
##########################################################################################
import time


from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'ref': torch.tensor([30, 4])
}

model_params = {
    'node_size': 21,
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
    'dec_method': 'WS',
    'model_load': {
        'path': './result/train_cvrp_n20',
        'info': "MOCVRP20",
        'epoch': 200,
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    # 'aug_factor': 1,
    'aug_factor': 8,
    'aug_batch_size': 200
}
if tester_params['aug_factor'] > 1:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']
# if tester_params['augmentation_enable']:
#     tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__cvrp_n20',
        # 'desc': 'test__eil_20',
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
##########################################################################################
def main(n_sols = 40):

    if tester_params['aug_factor'] == 1:
        sols_floder = "PMOCO_mean_sols.txt"
        pareto_fig = "PMOCO_Pareto.png"
        all_sols_floder = "PMOCO_all_mean_sols.txt"
        hv_floder = "PMOCO_hv.txt"
    else:
        sols_floder = "PMOCO(aug)_mean_sols.txt"
        pareto_fig = "PMOCO(aug)_Pareto.png"
        all_sols_floder = "PMOCO(aug)_all_mean_sols.txt"
        hv_floder = "PMOCO(aug)_hv.txt"

    logger_start = time.time()

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    test_path = f"./data/testdata_cvrp_size{env_params['problem_size']}.pt"
    data = torch.load(test_path)
    shared_node_demand = data['demand_data'].squeeze(-1).to(device=CUDA_DEVICE_NUM)
    shared_depot_xy = data['node_data'][:, 0, :].unsqueeze(1).to(device=CUDA_DEVICE_NUM)
    shared_node_xy = data['node_data'][:, 1:, :].to(device=CUDA_DEVICE_NUM)

    # right makespan
    ref = np.array([30,4])   #20
    # ref = np.array([45, 4])   #50
    # ref = np.array([80,4])   #100

    batch_size = shared_node_xy.shape[0]
    sols = torch.ones([batch_size, 10000, 2]) * 1000
    pref = torch.zeros(n_sols, 2).cuda()
    # shuffle
    for i in range(n_sols):
        pref[i, 0] = 1 - i / (n_sols - 1)
        pref[i, 1] = i / (n_sols - 1)
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
        sols_ = tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref, episode=b_start)
        test_timer_end = time.time()
        total_test_time += test_timer_end - test_timer_start

        sols[b_start:b_end, :sols_.shape[1], :] = sols_
        sols_len = max(sols_len, sols_.shape[1])

    sols = sols[:, :sols_len, :]

    print('Avg Test Time(s): {:.4f}\n'.format(total_test_time))

    max_obj1 = sols.reshape(-1, 2)[:, 0].max()
    max_obj2 = sols.reshape(-1, 2)[:, 1].max()
    txt2 = F"{tester.result_folder}/max_cost_n{env_params['problem_size']}.txt"
    f = open(
        txt2,
        'a')
    f.write(f"MOCVRP{env_params['problem_size']}\n")
    f.write(f"MAX OBJ1:{max_obj1}\n")
    f.write(f"MAX OBJ2:{max_obj2}\n")
    f.close()

    fig = plt.figure()


    sols_mean = sols.mean(0).cpu()
    plt.plot(sols_mean[:,0],sols_mean[:,1], marker = 'o', c = 'C1',ms = 3,  label='PSL-MOCO (Ours)')
    plt.savefig(F"{tester.result_folder}/{pareto_fig}")
    # np.savetxt(F"{tester.result_folder}/PMOCO_mean_sols.txt", sols.reshape(-1, 2),
    #            delimiter='\t', fmt="%.4f\t%.4f")

    np.savetxt(F"{tester.result_folder}/{sols_floder}", sols_mean,
               delimiter='\t', fmt="%.4f\t%.4f")

    plt.legend()

    p_sols, _, p_sols_num = update_EP(sols, None)
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref)
    print('HV Ratio: {:.4f}'.format(hvs.mean()))
    print('NDS: {:.4f}'.format(p_sols_num.float().mean()))

    os.makedirs(tester.result_folder + '/sols')
    for i in range(p_sols.shape[0]):
        np.savetxt(F"{tester.result_folder}/sols/ins{i}.txt", p_sols[i, :p_sols_num[i]].cpu().numpy(),
                   delimiter='\t', fmt="%.4f\t%.4f")
    np.savetxt(F"{tester.result_folder}/{hv_floder}", hvs,
               delimiter='\t', fmt="%.4f")

    if tester_params['aug_factor'] == 1:
        f = open(
            F"{tester.result_folder}/PMOCO-CVRP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO-CVRP{env_params['problem_size']}\n")
    else:
        f = open(
            F"{tester.result_folder}/PMOCO(aug)-CVRP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO(aug)-CVRP{env_params['problem_size']}\n")

    f.write(f"Model Path: {tester_params['model_load']['path']}\n")
    f.write(f"Model Epoch: {tester_params['model_load']['epoch']}\n")
    f.write(f"Pref Num: {n_sols}\n")
    f.write(f"Neighbor Num: {tester_params['neighbor']}\n")
    f.write(f"Hyper Hidden Dim: {model_params['hyper_hidden_dim']}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Aug Factor: {tester_params['aug_factor']}\n")
    f.write('Test Time(s): {:.4f}\n'.format(total_test_time))
    f.write('HV Ratio: {:.4f}\n'.format(hvs.mean()))
    f.write('NDS: {:.4f}\n'.format(p_sols_num.float().mean()))
    f.write(f"Ref Point:[{ref[0]},{ref[1]}] \n")
    f.write(f"Info: {tester_params['model_load']['info']}\n")
    # f.write(f"{compare_type}_{optim} avg_hv:{avg_hvs} s\n")
    f.close()


##########################################################################################
if __name__ == "__main__":
    main()