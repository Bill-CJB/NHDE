import torch
import numpy as np
size = 100
torch.manual_seed(1234)
testdata = torch.FloatTensor(np.random.rand(200, size, 6))
torch.save(testdata, 'testdata_tsp_3o_size{}.pt'.format(size))
