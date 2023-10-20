import torch

USE_CUDA = True
CUDA_DEVICE_NUM = 7
SEED = 1234

PROBLEM_SIZE = 20  # number of customer nodes
if PROBLEM_SIZE == 20:
    REF = torch.tensor([30, 4])
elif PROBLEM_SIZE == 50:
    REF = torch.tensor([45, 4])
elif PROBLEM_SIZE == 100:
    REF = torch.tensor([80, 4])
else:
    raise NotImplementedError

MODE = 2  # 1 denotes Train, 2 denotes Test
METHOD = 'NHDE-M'
SAVE_NUM = 150

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
TASK_NUM = 2
N_WEIGHT = 40
AGG = 1  # Weight Aggregation, 1 denotes Weighted-Sum, 2 denotes Weighted-Tchebycheff
TEST_DATASET_SIZE = 200
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 200
TESTAUG_BATCH_SIZE = 200
AUG = 8

NODE_SIZE = PROBLEM_SIZE + 1
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
