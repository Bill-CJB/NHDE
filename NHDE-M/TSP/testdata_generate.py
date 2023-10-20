import torch
import numpy as np
size = 20
torch.manual_seed(1234)
testdata = torch.FloatTensor(np.random.rand(200, size, 4))
torch.save(testdata, 'testdata_size{}.pt'.format(size))
