import torch
import numpy as np
size = 200
torch.manual_seed(1234)
testdata = torch.FloatTensor(np.random.rand(200, size, 3))
torch.save(testdata, 'testdata_kp_size{}.pt'.format(size))
