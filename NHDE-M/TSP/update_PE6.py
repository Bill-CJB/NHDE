"""
input non-dominated solution set and solution set, shape are both [batch, ins, obj]
update non-dominated solution set with initial max length, padded with ref point
with flag
not use Class
"""

import torch

# INF = 1e4
# MAX_EP_NUM = int(1e4)


def sort(EP):
    bs = EP.size(0)
    ps = EP.size(1)
    BATCH_IDX = torch.arange(bs)[:, None].expand(-1, ps)
    SORT_IDX = EP[:, :, 0].argsort(-1)
    EP = EP[BATCH_IDX, SORT_IDX]
    return EP


def update_EP(objs, dummy_EP=None, ref=torch.tensor([1e4, 1e3])):
    next_objs = objs.clone()
    bs, ps, obj_dim = next_objs.size()

    # initial
    if dummy_EP is None:
        if ref[0] == ref[1]:
            dummy_EP = torch.ones((bs, ps, obj_dim)).to(objs.device) * ref[0]
        else:
            dummy_EP = torch.ones((bs, ps, obj_dim)).to(objs.device)
            for i in range(obj_dim):
                dummy_EP[:, :, i] = dummy_EP[:, :, i] * ref[i]
    else:
        if ref[0] == ref[1]:
            dummy_EP_expand = torch.ones((bs, ps, obj_dim)).to(objs.device) * ref[0]
        else:
            dummy_EP_expand = torch.ones((bs, ps, obj_dim)).to(objs.device)
            for i in range(obj_dim):
                dummy_EP_expand[:, :, i] = dummy_EP_expand[:, :, i] * ref[i]
        dummy_EP = torch.cat((dummy_EP, dummy_EP_expand), 1)


    EP_len = dummy_EP.shape[1]
    if ref[0] == ref[1]:
        infs = torch.ones((bs, EP_len, obj_dim)).to(objs.device) * ref[0]
    else:
        infs = torch.ones((bs, EP_len, obj_dim)).to(objs.device)
        for i in range(obj_dim):
            infs[:, :, i] = infs[:, :, i] * ref[i]


    ep_len = len(dummy_EP[0])
    for pi in range(ps):
        cur_objs = next_objs[:, pi][:, None, :].expand(-1, ep_len, -1)
        pareto_mask = cur_objs < dummy_EP

        nd_mask = pareto_mask.any(-1).all(-1)
        idx_mask = pareto_mask.all(-1)

        # protect not put idx
        idx_mask[:, -1] = True

        # check if all sols non_dominated except inf
        next_pareto_idx = [idx_mask[i].nonzero()[0] for i in range(bs)]
        next_pareto_idx = torch.stack(next_pareto_idx, 0)

        # update pareto solutions
        tmp_value = dummy_EP.scatter(1, next_pareto_idx[:, :, None].expand(-1, -1, obj_dim),
                                     next_objs[:, pi][:, None, :])
        dummy_EP = torch.where(nd_mask[:, None, None].expand(-1, ep_len, obj_dim),
                               tmp_value, dummy_EP)

        # continue to remove other dominated solutions
        while True:
            pareto_mask = next_objs[:, pi][:, None, :].expand(-1, ep_len, -1) < dummy_EP
            inf_mask = dummy_EP[:, :, 0] == ref[0]
            idx_mask = pareto_mask.all(-1)
            update_mask = (idx_mask & ~inf_mask).any(-1)

            if update_mask.any() == False:
                break


            idx_mask = idx_mask & ~inf_mask

            # protect not put idx
            idx_mask[:, -1] = True

            next_pareto_idx = [idx_mask[i].nonzero()[0] for i in range(bs)]
            next_pareto_idx = torch.stack(next_pareto_idx, 0)

            tmp_value = dummy_EP.scatter(1, next_pareto_idx[:, :, None].expand(-1, -1, obj_dim), infs)
            dummy_EP = torch.where(update_mask[:, None, None].expand(-1, ep_len, obj_dim),
                                   tmp_value, dummy_EP)

    inf_mask = dummy_EP[:, :, 0] == ref[0]
    set_num = (~inf_mask).long().sum(-1)
    max_num = set_num.max().item()
    EP_num = set_num

    dummy_EP = sort(dummy_EP)

    FLAG_IDX = torch.arange(max_num)[None, :].expand(bs, max_num).to(dummy_EP.device)
    flag = FLAG_IDX > (EP_num[:, None] - 1).expand(-1, max_num)
    flag = flag.int()

    dummy_EP = dummy_EP[:, :max_num]

    return dummy_EP, flag, EP_num
