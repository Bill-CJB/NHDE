import torch
import numpy as np
PROBLEM_SIZE = 20
torch.manual_seed(1234)
if PROBLEM_SIZE == 20:
    demand_scaler = 30
elif PROBLEM_SIZE == 50:
    demand_scaler = 40
elif PROBLEM_SIZE == 100:
    demand_scaler = 50
else:
    raise NotImplementedError
testdata={'node_data': torch.FloatTensor(np.random.rand(200, PROBLEM_SIZE+1, 2)),
    'demand_data': torch.FloatTensor(np.random.randint(1, 10, 200 * PROBLEM_SIZE) / demand_scaler).reshape(200, PROBLEM_SIZE, 1)}
torch.save(testdata, 'testdata_size{}.pt'.format(PROBLEM_SIZE))

