##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 1

##########################################################################################
# Path Config
import os
import sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################

##########################################################################################
import hvwfg


PROBLEM = "TSP_3obj"
if PROBLEM == "TSP":
    TEST_DIR = "test_ws"
else:
    TEST_DIR = "test_ws_all(hv)"
##########################################################################################
def cal_hvs(sols_file, ref, n_sols=101):
    batch_size = 200
    sols = np.zeros([batch_size*n_sols, 3])
    hvs = np.zeros([batch_size, 1])
    idx = 0
    with open(sols_file, "r", encoding="utf") as f:
        strs = f.readlines()

    for i in strs:
        if "\t" in i:
            sols[idx][0] = float(i.split("\t")[0])
            sols[idx][1] = float(i.split("\t")[1])
            sols[idx][2] = float(i.split("\t")[2])
            idx += 1
    sols = sols.reshape(batch_size, n_sols, 3)

    if PROBLEM == "KP":
        if ref[0] == -5:
            max_value = 30
        elif ref[0] == -20:
            max_value = 50
        elif ref[0] == -40:
            max_value = 60
        else:
            return NotImplementedError
        for pi in range(batch_size):
            hv = hvwfg.wfg(-sols[pi].astype(float), ref.astype(float))
            hv_ratio = hv / ((max_value + ref[0]) * (max_value + ref[1]))
            hvs[pi] = hv_ratio
    else:
        for pi in range(batch_size):
            hv = hvwfg.wfg(sols[pi].astype(float), ref.astype(float))
            hv_ratio = hv / (ref[0] * ref[1] * ref[2])
            hvs[pi] = hv_ratio

    return hvs


def get_filename(problem_size, sols_file):
    if "aug" in sols_file:
        hv_file = f"PMOCO(aug)_{PROBLEM}{problem_size}_hv.txt"
        result_file = f"PMOCO(aug)-{PROBLEM}{problem_size}_result.txt"
    else:
        hv_file = f"PMOCO_{PROBLEM}{problem_size}_hv.txt"
        result_file = f"PMOCO-{PROBLEM}{problem_size}_result.txt"
    return hv_file, result_file

def get_ref(problem_size):

    if PROBLEM == "KP":
        if problem_size == 50:
            ref = np.array([-5, -5])  # 20
        elif problem_size == 100:
            ref = np.array([-20, -20])
        elif problem_size == 200:
            ref = np.array([-40, -40])
        else:
            return NotImplementedError
    elif PROBLEM == "TSP":
        if problem_size == 20:
            ref = np.array([20, 20])  # 20
        elif problem_size == 50:
            ref = np.array([35, 35])
        elif problem_size == 100:
            ref = np.array([65, 65])
        else:
            return NotImplementedError
    elif PROBLEM == "TSP_3obj":
        if problem_size == 20:
            ref = np.array([20, 20, 20])  # 20
        elif problem_size == 50:
            ref = np.array([35, 35, 35])
        elif problem_size == 100:
            ref = np.array([65, 65, 65])
        else:
            return NotImplementedError
    elif PROBLEM == "CVRP":
        if problem_size == 20:
            ref = np.array([30, 4])  # 20
        elif problem_size == 50:
            ref = np.array([45, 4])
        elif problem_size == 100:
            ref = np.array([80, 4])
        else:
            return NotImplementedError
    else:
        return NotImplementedError
    return ref

def main(problem_size, data_dir, aug_data_dir):
    result_folder = f"./result/MO{PROBLEM}/PMOCO/{problem_size}"
    if "3" in PROBLEM:
        n_sols = 105
    else:
        n_sols = 101

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    ref = get_ref(problem_size)

    aug_factor = 1
    sols_file = f"./result/{TEST_DIR}/{data_dir}/PMOCO_all_mean_sols_n{problem_size}.txt"
    # sols_file = f"./result/{TEST_DIR}/{data_dir}/PMOCO_all_mean_sols.txt"

    hv_file, result_file = get_filename(problem_size, sols_file)

    f = open(F"{result_folder}/{result_file}", 'w')

    hvs = cal_hvs(sols_file, ref, n_sols)

    print(f"MOTSP{problem_size}_3obj Aug:{aug_factor} ", 'HV Ratio: {:.4f}'.format(hvs.mean()),
          f" Ref Point:[{ref[0]},{ref[1]},{ref[2]}]")

    np.savetxt(F"{result_folder}/{hv_file}", hvs,
               delimiter='\t', fmt="%.4f")

    f.write(f"Sols File: {sols_file}\n")
    f.write(f"Aug Factor: {aug_factor}\n")
    f.write('HV Ratio: {:.4f}\n'.format(hvs.mean()))

    if PROBLEM == "KP":
        return

    aug_factor = 64
    sols_file = f"./result/{TEST_DIR}/{aug_data_dir}/PMOCO(aug)_all_mean_sols_n{problem_size}.txt"
    # sols_file = f"./result/{TEST_DIR}/{aug_data_dir}/PMOCO(aug)_all_mean_sols.txt"

    hv_file, result_file = get_filename(problem_size, sols_file)
    hvs = cal_hvs(sols_file, ref, n_sols)

    print(f"MOTSP{problem_size}_3obj Aug:{aug_factor} ", 'HV Ratio: {:.4f}'.format(hvs.mean()),
          f" Ref Point:[{ref[0]},{ref[1]},{ref[2]}]")

    np.savetxt(F"{result_folder}/{hv_file}", hvs,
               delimiter='\t', fmt="%.4f")
    f.write(f"Aug Factor: {aug_factor}\n")
    f.write('HV Ratio: {:.4f}\n'.format(hvs.mean()))
    f.write(f"Ref Point:[{ref[0]},{ref[1]},{ref[2]}] \n")
    f.close()

##########################################################################################
if __name__ == "__main__":
    data_dir = "20220718_222857_test__tsp_n20"
    aug_data_dir = "20220718_223010_test__tsp_n20"
    main(20, data_dir, aug_data_dir)
    data_dir = "20220718_223103_test__tsp_n50"
    aug_data_dir = "20220718_223134_test__tsp_n50"
    main(50, data_dir, aug_data_dir)
    data_dir = "20220719_093521_test__tsp_n100"
    aug_data_dir = "20220719_093532_test__tsp_n100"
    main(100, data_dir, aug_data_dir)
    #
    # data_dir = ""
    # aug_data_dir = ""
    # main(50, data_dir, aug_data_dir)
    # data_dir = ""
    # aug_data_dir = ""
    # main(100, data_dir, aug_data_dir)
    # data_dir = ""
    # aug_data_dir = ""
    # main(200, data_dir, aug_data_dir)