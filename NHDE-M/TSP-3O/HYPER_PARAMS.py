import torch

USE_CUDA = True
CUDA_DEVICE_NUM = 7
SEED = 1234

TSP_SIZE = 20  # TSP number of nodes
if TSP_SIZE == 20:
    REF = torch.tensor([20, 20, 20])
elif TSP_SIZE == 50:
    REF = torch.tensor([35, 35, 35])
elif TSP_SIZE == 100:
    REF = torch.tensor([65, 65, 65])
else:
    raise NotImplementedError

MODE = 2  # 1 denotes Train, 2 denotes Test, 3 denotes Test convergence
METHOD = 'NHDE-M'
SAVE_INTVL = 1

# LOAD_PATH = 'result/size20/checkpoint-1.pt'  # load model to train
LOAD_PATH = None
MODEL_DIR = 'result/size20'  # load model to test

# Hyper-Parameters
SOLVING_TIMES = 20
NEIGHBOR = 20
CANDIDATE = 200
TOTAL_EPOCH = 150
UPDATE_STEP = 100
FINETUNE_STEP = 50
TASK_NUM = 3
N_WEIGHT = 210  # 105, 210
AGG = 1  # Weight Aggregation, 1 denotes Weighted-Sum, 2 denotes Weighted-Tchebycheff
TEST_DATASET_SIZE = 200
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 200
TESTAUG_BATCH_SIZE = 200

NODE_SIZE = TSP_SIZE
EMBEDDING_DIM = 128
KEY_DIM = 16  # Length of q, k, v of EACH attention head
HEAD_NUM = 8
ENCODER_LAYER_NUM = 6
FF_HIDDEN_DIM = 512
LOGIT_CLIPPING = 10  # (C in the paper)

META_LR = 1
ACTOR_LEARNING_RATE = 1e-4
ACTOR_WEIGHT_DECAY = 1e-6

LR_DECAY_EPOCH = 1
LR_DECAY_GAMMA = 1.00

# Logging
LOG_PERIOD_SEC = 15

OBJ_NUM = 3
AUG_NUM = 128