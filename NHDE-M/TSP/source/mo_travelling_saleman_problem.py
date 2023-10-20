import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from HYPER_PARAMS import *
from TORCH_OBJECTS import *

def TSP_DATA_LOADER__RANDOM(num_sample, num_nodes, batch_size, seed=None):
    if seed:
        np.random.seed(seed)
    dataset = TSP_Dataset__Random(num_sample=num_sample, num_nodes=num_nodes)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=TSP_collate_fn)
    return data_loader


class TSP_Dataset__Random(Dataset):
    def __init__(self, num_sample, num_nodes):
        self.num_sample = num_sample
        self.num_nodes = num_nodes
        self.data = np.random.rand(num_sample, num_nodes, 4)

    def __getitem__(self, index):
        # node_xy_data = np.random.rand(self.num_nodes, 4)
        node_xy_data = self.data[index]

        return node_xy_data

    def __len__(self):
        return self.num_sample

def TSP_collate_fn(batch):
    node_xy = Tensor(batch)
    return node_xy

class STATE:

    def __init__(self, seq):
        self.seq = seq
        self.batch_s = seq.size(0)

        self.current_node = None

        # History
        ####################################
        self.selected_count = 0
        self.available_mask = BoolTensor(np.ones((self.batch_s, TSP_SIZE)))
        self.ninf_mask = Tensor(np.zeros((self.batch_s, TSP_SIZE)))
        # shape = (batch_s, TSP_SIZE)
        self.selected_node_list = LongTensor(np.zeros((self.batch_s, 0)))
        # shape = (batch_s, selected_count)

    def move_to(self, selected_node_idx):
        # selected_node_idx.shape = (batch,)

        self.current_node = selected_node_idx

        # History
        ####################################
        self.selected_count += 1
        self.available_mask[torch.arange(self.batch_s), selected_node_idx] = False
        self.ninf_mask[torch.arange(self.batch_s), selected_node_idx] = -np.inf
        self.selected_node_list = torch.cat((self.selected_node_list, selected_node_idx[:, None]), dim=1)


class GROUP_STATE:

    def __init__(self, group_size, data):
        # data.shape = (batch, group, 2)
        self.batch_s = data.size(0)
        self.group_s = group_size
        self.data = data

        # History
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape = (batch, group)
        self.selected_node_list = LongTensor(np.zeros((self.batch_s, group_size, 0)))
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.ninf_mask = Tensor(np.zeros((self.batch_s, group_size, TSP_SIZE)))
        # shape = (batch, group, TSP_SIZE)


    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # History
        ####################################
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat((self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)

        # Status
        ####################################
        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(self.batch_s, self.group_s)
        self.ninf_mask[batch_idx_mat, group_idx_mat, selected_idx_mat] = -np.inf


class ENVIRONMENT:

    def __init__(self, seq):
    
        self.seq = seq
        self.batch_s = seq.size(0)
        self.state = None

    def reset(self):
        self.state = STATE(self.seq)

        reward = None
        done = False
        return self.state, reward, done

    def step(self, selected_node_idx):
        # selected_node_idx.shape = (batch,)

        # move state
        self.state.move_to(selected_node_idx)

        # returning values
        done = self.state.selected_count == TSP_SIZE
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.state, reward, done

    def _get_travel_distance(self):

        gathering_index = self.state.selected_node_list.unsqueeze(2).expand(-1, TSP_SIZE, 4)
        
        ordered_seq = self.seq.gather(dim=1, index=gathering_index)

        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(2).sqrt()
        # size = (batch, TSP_SIZE)

        travel_distances = segment_lengths.sum(1)
        # size = (batch,)
        return travel_distances

class GROUP_ENVIRONMENT:

    def __init__(self, data):
       

        self.data = data
        self.batch_s = data.size(0)
        self.group_s = None
        self.group_state = None

    def reset(self, group_size):
        self.group_s = group_size
        self.group_state = GROUP_STATE(group_size=group_size, data=self.data)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # selected_idx_mat.shape = (batch, group)

        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = (self.group_state.selected_count == TSP_SIZE)
        if done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_group_travel_distance(self):
      
        gathering_index = self.group_state.selected_node_list.unsqueeze(3).expand(self.batch_s, -1, TSP_SIZE, 4)
       
        
        seq_expanded = self.data[:, None, :, :].expand(self.batch_s, self.group_s, TSP_SIZE, 4)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        
        
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        
        segment_lengths_obj1 = ((ordered_seq[:, :, :, :2]-rolled_seq[:, :, :, :2])**2).sum(3).sqrt()
        segment_lengths_obj2 = ((ordered_seq[:, :, :, 2:]-rolled_seq[:, :, :, 2:])**2).sum(3).sqrt()

        group_travel_distances_obj1 = segment_lengths_obj1.sum(2)
        group_travel_distances_obj2 = segment_lengths_obj2.sum(2)
    
        group_travel_distances_combined = torch.stack([group_travel_distances_obj1,group_travel_distances_obj2],axis = 2)
       
        
        return group_travel_distances_combined
