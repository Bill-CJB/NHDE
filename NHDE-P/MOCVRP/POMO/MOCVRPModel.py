import torch
import torch.nn as nn
import torch.nn.functional as F


class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state, sols, sols_mask):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand, sols, sols_mask)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state, sols_mask_pomo):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)


        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, state.load, sols_mask_pomo, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.sols_embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand, sols, sols_mask):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)
        sols_embedded_input = self.sols_embedding(sols)
        # shape: (batch, sols, embedding)

        # out = torch.cat((embedded_depot, embedded_node), dim=1)
        out = torch.cat((embedded_depot, embedded_node, sols_embedded_input), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out, sols_mask)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.Wq_s = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_s = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_s = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine_s = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1, sols_mask):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q_n = reshape_by_heads(self.Wq(input1[:, :self.model_params['node_size']]), head_num=head_num)
        k_n = reshape_by_heads(self.Wk(input1[:, :self.model_params['node_size']]), head_num=head_num)
        v_n = reshape_by_heads(self.Wv(input1[:, :self.model_params['node_size']]), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        q_s = reshape_by_heads(self.Wq_s(input1[:, self.model_params['node_size']:]), head_num=head_num)
        k_s = reshape_by_heads(self.Wk_s(input1[:, self.model_params['node_size']:]), head_num=head_num)
        v_s = reshape_by_heads(self.Wv_s(input1[:, self.model_params['node_size']:]), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat_nn = multi_head_attention(q_n, k_n, v_n)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        out_concat_ns = multi_head_attention(q_n, k_s, v_s, rank2_ninf_mask=sols_mask)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out_n = self.multi_head_combine(out_concat_nn + out_concat_ns)
        # shape: (batch, problem, EMBEDDING_DIM)

        out_concat_sn = multi_head_attention(q_s, k_n, v_n)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out_s = self.multi_head_combine_s(out_concat_sn)
        # shape: (batch, problem, EMBEDDING_DIM)

        multi_head_out = torch.cat((multi_head_out_n, multi_head_out_s), dim=1)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        hyper_input_dim = 2 + 2
        hyper_hidden_embd_dim = self.model_params['hyper_hidden_dim']
        self.embd_dim = 2 + 2
        self.hyper_output_dim = 6 * self.embd_dim
        
        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)
        
        self.hyper_Wq_last = nn.Linear(self.embd_dim, (embedding_dim + 1) * head_num * qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wv = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_multi_head_combine = nn.Linear(self.embd_dim, head_num * qkv_dim * embedding_dim, bias=False)
        self.hyper_Wk_s = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)
        self.hyper_Wv_s = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)

        self.Wq_last_para = None
        self.multi_head_combine_para = None


        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
     
    def assign(self, pref):
        
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[:1 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim + 1)
        self.Wk_para = self.hyper_Wk(mid_embd[1 * self.embd_dim: 2 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim)
        self.Wv_para = self.hyper_Wv(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_para = self.hyper_multi_head_combine(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim)
        self.Wk_para_s = self.hyper_Wk_s(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(embedding_dim,
                                                                                                 head_num * qkv_dim)
        self.Wv_para_s = self.hyper_Wv_s(mid_embd[5 * self.embd_dim: 6 * self.embd_dim]).reshape(embedding_dim,
                                                                                                 head_num * qkv_dim)
        

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(F.linear(encoded_nodes[:, :self.model_params['node_size']], self.Wk_para),
                                  head_num=head_num)
        self.v = reshape_by_heads(F.linear(encoded_nodes[:, :self.model_params['node_size']], self.Wv_para),
                                  head_num=head_num)

        self.k_s = reshape_by_heads(F.linear(encoded_nodes[:, self.model_params['node_size']:], self.Wk_para_s),
                                    head_num=head_num)
        self.v_s = reshape_by_heads(F.linear(encoded_nodes[:, self.model_params['node_size']:], self.Wv_para_s),
                                    head_num=head_num)
        
        self.single_head_key = encoded_nodes[:, :self.model_params['node_size']].transpose(1, 2)
       
 
    def forward(self, encoded_last_node, load, sols_mask_pomo, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)

        # shape = (batch, group, EMBEDDING_DIM+1)
        q_last = reshape_by_heads(F.linear(input_cat, self.Wq_last_para), head_num = head_num)
        q = q_last

        out_concat_n = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        out_concat_s = multi_head_attention(q, self.k_s, self.v_s, rank3_ninf_mask=sols_mask_pomo)
        out_concat = out_concat_n + out_concat_s
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = F.linear(out_concat, self.multi_head_combine_para)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))