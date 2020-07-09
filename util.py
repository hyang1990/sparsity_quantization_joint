import copy
import math
import time

import torch
from torch import nn

_NZ_PES = 1e-5
_MAX_KMEANS_N = 1 * (2 ** 20)


def quantized_sparsify(weight_list, weight_bits, model_size, in_place=False):
    param_flats = [p.data.view(-1) for p in weight_list]
    param_flats_all = torch.cat(param_flats, dim=0)
    knapsack_weight_all = torch.cat([torch.ones_like(p) * weight_bits[i] for i, p in enumerate(param_flats)], dim=0)
    score_all = torch.cat([p ** 2 for p in param_flats], dim=0) / knapsack_weight_all
    sorted_idx = torch.sort(score_all, descending=True)[1]
    cumsum = torch.cumsum(knapsack_weight_all[sorted_idx], dim=0)
    res_nnz = torch.nonzero(cumsum <= model_size).max().item()
    z_idx = sorted_idx[-(param_flats_all.numel() - res_nnz):]
    param_flats_all[z_idx] = 0.0
    # in-place zero-out
    i = 0
    res_weight_list = []
    for p in weight_list:
        p_temp = param_flats_all[i:i+p.numel()].view(p.shape)
        if in_place:
            p.data.copy_(p_temp)
        res_weight_list.append(p_temp)
        i += p.numel()
    num_nnz = [w.nonzero().shape[0] for w in res_weight_list]
    return res_weight_list, num_nnz


def prune_with_nnz(model, weight_nnz):
    i = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            z_idx = torch.topk(torch.abs(m.weight.data.view(-1)), m.weight.numel() - weight_nnz[i], largest=False)[1]
            m.weight.data.view(-1)[z_idx] = 0.0
            i += 1
    assert i == len(weight_nnz)


def mckp_greedy(profit, weight, group_size, budget, sorted_weights=True):
    """
    Greedy algorithm for Multi-Choice knapsack problem
    :param profit: items' profits
    :param weight: items' weights
    :param group_size: groups' size
    :param budget: weight budget
    :param sorted_weights: if each group's items are sorted by weights
    :return: binary solution of selected items
    """
    # get group offsets
    offset = [0] * len(group_size)
    temp = 0
    for i in range(len(group_size)):
        offset[i] = temp
        temp += group_size[i]

    if not sorted_weights:
        raw_sorted_idx = torch.zeros_like(profit, dtype=torch.long)
        for i in range(len(group_size)):
            if i + 1 < len(offset):
                indices = torch.argsort(weight[offset[i]:offset[i + 1]])
                raw_sorted_idx[offset[i]:offset[i+1]] = indices + offset[i]
            else:
                indices = torch.argsort(weight[offset[i]:])
                raw_sorted_idx[offset[i]:] = indices + offset[i]
        weight = weight[raw_sorted_idx].clone()
        profit = profit[raw_sorted_idx].clone()

    profit -= (profit.min() - 1e-6)
    # preprocess: remove the dominated items
    idx = torch.ones_like(profit, dtype=torch.bool)
    reduced_group_size = copy.deepcopy(group_size)
    for gi, gs in enumerate(group_size):
        if gs <= 1:
            continue
        go = offset[gi]
        temp = profit[go]
        for i in range(1, gs):
            if profit[go+i] <= temp:
                idx[go+i] = 0
                reduced_group_size[gi] -= 1
            else:
                temp = profit[go+i]

        if reduced_group_size[gi] <= 2:
            continue
        stack = [(go, None)]
        # print('idx1={}'.format(idx[go:go+gs]))
        for i in range(1, gs):
            cur_idx = go + i
            if bool(idx[cur_idx]):
                while True:
                    score = ((profit[cur_idx] - profit[stack[-1][0]]) / (weight[cur_idx] - weight[stack[-1][0]])).item()
                    if len(stack) <= 1 or score < stack[-1][1]:
                        stack.append((cur_idx, score))
                        break
                    else:
                        del_idx, del_score = stack.pop()
                        idx[del_idx] = 0
                        # print('profit={}, weight={}'.format(profit[del_idx], weight[del_idx]))
                        reduced_group_size[gi] -= 1
        # print('idx2={}'.format(idx[go:go+gs]))
    # greedy algorithm
    R_profit = profit[idx]
    R_d_profit = R_profit.clone()
    R_d_profit[1:] -= R_profit[:-1]
    R_weight = weight[idx]
    R_d_weight = R_weight.clone()
    R_d_weight[1:] -= R_weight[:-1]

    # print('profit={}'.format(profit))
    # print('weight={}'.format(weight))
    # print('gs={}'.format(reduced_group_size))

    R_score = R_d_profit / R_d_weight
    sorted_idx = sorted(range(len(R_score)), key=R_score.__getitem__, reverse=True)

    res = torch.zeros(len(R_score), dtype=torch.bool)
    res_profit = 0.0
    res_weight = 0.0
    group_idices = []
    offset = [0] * len(reduced_group_size)
    temp = 0
    for i in range(len(reduced_group_size)):
        offset[i] = temp
        # select the first item in each group
        res[offset[i]] = 1
        res_profit += R_profit[offset[i]].item()
        res_weight += R_weight[offset[i]].item()
        temp += reduced_group_size[i]
        group_idices += [i] * reduced_group_size[i]

    offset = set(offset)
    finished_group_indices = set()
    for i in sorted_idx:
        if i not in offset:
            if res_weight + R_d_weight[i].item() > budget:
                # break
                finished_group_indices.add(group_idices[i])
            if group_idices[i] not in finished_group_indices:
                if res[i-1] != 1:
                    print('idx={} is not selected, but {} is selecting'.format(i-1, i))
                assert res[i-1] == 1, 'sorted idx={}, offset={}, profit={}, weight={}'.format(sorted_idx, offset,
                                                                                              R_d_profit, R_d_weight)
                res[i-1] = 0
                res[i] = 1
                assert R_d_profit[i].item() >= 0
                res_profit += R_d_profit[i].item()
                res_weight += R_d_weight[i].item()

    raw_res = torch.zeros_like(profit, dtype=torch.bool)
    raw_res[idx] = res
    if not sorted_weights:
        raw_raw_res = torch.zeros_like(raw_res)
        raw_raw_res[raw_sorted_idx] = raw_res
        return raw_raw_res
    return raw_res


def zerok_means1D(X, n_clusters, niter=100, val_dict=None, dictnz=False):
    # n_clusters includes 0
    X = X.view(-1)
    if dictnz:
        if val_dict is None:
            val_dict = torch.linspace(X.min().item(), X.max().item(), steps=n_clusters, dtype=X.dtype, device=X.device)
        else:
            assert val_dict.numel() == n_clusters
    else:
        idx = torch.nonzero(X)
        if val_dict is None:
            if idx.shape[0] < n_clusters - 1:
                if idx.shape[0] == 0:
                    val_dict = torch.zeros(n_clusters, dtype=X.dtype, device=X.device)
                else:
                    val_dict = torch.cat([torch.zeros(n_clusters - idx.shape[0], dtype=X.dtype, device=X.device),
                                          torch.linspace(X[idx].min().item(), X[idx].max().item(), steps=idx.shape[0],
                                                         dtype=X.dtype, device=X.device)], dim=0)
                return val_dict
            else:
                val_dict = torch.cat([torch.zeros(1, dtype=X.dtype, device=X.device),
                                      torch.linspace(X[idx].min().item(), X[idx].max().item(), steps=n_clusters - 1,
                                                     dtype=X.dtype, device=X.device)], dim=0)
        else:
            assert val_dict.numel() == n_clusters
            assert val_dict[0].item() == 0

        X = X[idx].view(-1)

    if X.shape[0] > _MAX_KMEANS_N:
        mem_efficient = True
    else:
        mem_efficient = False

    torch.cuda.empty_cache()
    tol = 1e-3
    pre_dist = None
    start_idx = (0 if dictnz else 1)
    if not mem_efficient:
        one_hot = torch.zeros(X.shape[0], n_clusters, dtype=X.dtype, device=X.device)
        dist_buff = torch.zeros_like(one_hot)
    else:
        dist_buff = torch.zeros(_MAX_KMEANS_N, n_clusters, dtype=X.dtype, device=X.device)
    for t in range(niter):
        if mem_efficient:
            km_dist, km_code = mem_1ddist_min(X, val_dict, dist_buff=dist_buff)
        else:
            # assign codes
            torch.add(X.view(-1, 1), -1, val_dict, out=dist_buff)
            dist_buff.pow_(2)
            km_dist, km_code = torch.min(dist_buff, dim=1)
        cur_dist = km_dist.sum().item()
        # print(cur_dist)
        if pre_dist is not None and cur_dist > 0 and abs(pre_dist - cur_dist) / cur_dist < tol:
            return val_dict
        # print(t, cur_dist)
        pre_dist = cur_dist
        # update dictonary
        if mem_efficient:
            for c in range(start_idx, n_clusters):
                Xc = X[km_code == c]
                if Xc.numel() > 0:
                    val_dict[c] = Xc.mean()
        else:
            one_hot = one_hot.zero_()
            one_hot.scatter_(1, km_code.unsqueeze(1), 1)
            # print(one_hot.shape)
            # print(X.shape)
            Xp = (X.unsqueeze(0).mm(one_hot)).view(-1)

            Xsum = one_hot.sum(dim=0)
            idx = torch.nonzero(Xsum)
            Xsum[idx] = 1. / Xsum[idx]
            val_dict[start_idx:] = (Xp * Xsum)[start_idx:]
    if not dictnz:
        assert val_dict[0].item() == 0
    # del one_hot, dist_buff
    return val_dict


def get_optim_val_dict(input, nbits, niter=100, val_dict=None, dictnz=False):
    X = input.data
    if dictnz:
        nbins = 2 ** nbits
    else:
        nbins = 2 ** nbits + 1
    val_dict = zerok_means1D(X, nbins, niter=niter, val_dict=val_dict, dictnz=dictnz)
    return val_dict


def mem_1ddist_min(X, Y, dist_buff=None):
    X, Y = X.view(-1), Y.view(-1)
    if X.numel() <= _MAX_KMEANS_N:
        return torch.min((X.view(-1, 1) - Y).pow_(2), dim=1)
    else:
        if dist_buff is None:
            dist_buff = torch.zeros(_MAX_KMEANS_N, Y.numel(), dtype=X.dtype, device=X.device)
        dist, idx = [], []
        n = math.ceil(X.numel() / _MAX_KMEANS_N)
        for i in range(n):
            if i < n - 1:
                # print(X.shape, i, _MAX_KMEANS_N * i, _MAX_KMEANS_N * (i + 1))
                temp_dist, temp_idx = torch.min(torch.add(X[_MAX_KMEANS_N * i: _MAX_KMEANS_N * (i + 1)].view(-1, 1),
                                                          -1, Y, out=dist_buff).pow_(2), dim=1)
            else:
                temp_dist, temp_idx = torch.min(torch.add(X[_MAX_KMEANS_N * i:].view(-1, 1),
                                                          -1, Y, out=dist_buff).pow_(2), dim=1)
            dist.append(temp_dist)
            idx.append(temp_idx)

        return torch.cat(dist), torch.cat(idx)


def km_quantize_tensor(input, nbits, val_dict, dictnz=False):
    if dictnz:
        nbins = 2 ** nbits
    else:
        nbins = 2 ** nbits + 1
    assert nbins == val_dict.numel(), '{} != {}'.format(nbins, val_dict.numel())
    # km_dist, km_code = torch.min((input.view(-1, 1) - val_dict) ** 2, dim=1)
    if not dictnz:
        km_code = torch.zeros(input.numel(), dtype=torch.int64, device=input.device)
        km_dist = input.view(-1) ** 2
        nz_idx = input.abs() >= _NZ_PES
        input_nz = input[nz_idx]
        if input_nz.numel() > 0:
            # km_dist_nz, km_code_nz = torch.min((input_nz.view(-1, 1) - val_dict).pow_(2), dim=1)
            km_dist_nz, km_code_nz = mem_1ddist_min(input_nz, val_dict)
            km_code[nz_idx.view(-1)] = km_code_nz
            km_dist[nz_idx.view(-1)] = km_dist_nz
    else:
        km_dist, km_code = mem_1ddist_min(input, val_dict)
    res = val_dict[km_code].view(input.shape)
    return res, km_dist, km_code


def km_project_tensor(input, nbits, km_code, one_hot=None, dictnz=False):
    X = input.view(-1)
    if dictnz:
        nbins = 2 ** nbits
    else:
        nbins = 2 ** nbits + 1
    if one_hot is None:
        one_hot = torch.zeros(X.shape[0], nbins, dtype=X.dtype, device=X.device)
        one_hot.scatter_(1, km_code.unsqueeze(1), 1)

    Xp = (X.unsqueeze(0).mm(one_hot)).view(-1)
    Xsum = one_hot.sum(dim=0)
    idx = torch.nonzero(Xsum)
    Xsum[idx] = 1. / Xsum[idx]
    val_dict = torch.cat([torch.zeros(1, dtype=X.dtype, device=X.device), (Xp * Xsum)[1:]])
    assert nbins == val_dict.numel(), '{} != {}'.format(nbins, val_dict.numel())
    X_proj = one_hot.mm(val_dict.unsqueeze(1)).view(-1)
    # print('X-X_proj={}'.format(torch.dist(X, X_proj)))
    X.copy_(X_proj)
    return one_hot


def mem_km_project_tensor(input, km_code, code_set=None):
    X = input.data.view(-1)
    if code_set is None:
        code_set = km_code.unique().tolist()

    for c in code_set:
        idx = (km_code == c)
        if c == 0:
            X[idx] = 0
        else:
            mu = X[idx].mean()
            X[idx] = mu

    return code_set


def sparse_quantize(weight_list, num_nnz, model_size, include_dict=False, in_place=False, dictnz=False,
                    cons_nbits=False, bwlb=1, mckp_time=None):
    # default_nbits_dict = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
    default_nbits_dict = torch.tensor(list(range(bwlb, 9)), dtype=torch.float32)
    if cons_nbits:
        cons_nbits_dict = default_nbits_dict[4:]
    else:
        cons_nbits_dict = default_nbits_dict
    mckp_budget = model_size
    mckp_p = []
    mckp_w = []
    mckp_gs = []
    weight_bits = []
    res_weight_list = []
    val_dicts_list = []
    for i, p in enumerate(weight_list):
        if i == len(weight_list) - 1:
            nbits_dict = cons_nbits_dict
        else:
            nbits_dict = default_nbits_dict
        dist4nbits = []
        val_dicts = {}
        # pnorm = p.data.norm().item() ** 2
        p_thresholed = p.data.clone()
        if not dictnz:
            p_thresholed[p_thresholed.abs() < _NZ_PES] = 0
        for nbits in nbits_dict:
            nbits = int(nbits)
            val_dict = get_optim_val_dict(p_thresholed, nbits, niter=100, dictnz=dictnz)
            # print(p.data.min().item())
            val_dicts[nbits] = val_dict
            assert torch.isnan(p.data).sum().item() == 0
            dist = km_quantize_tensor(p.data, nbits, val_dict=val_dict, dictnz=dictnz)[1].sum().item()
            dist4nbits.append(dist)

        mckp_p.append(-torch.tensor(dist4nbits, dtype=torch.float))
        dict_size = (2.0 ** nbits_dict) * 32 if include_dict else 0.0
        mckp_w.append(nbits_dict * num_nnz[i] + dict_size)
        mckp_gs.append(len(dist4nbits))
        val_dicts_list.append(val_dicts)
        # print("{} th layer {}".format(i, [round(dist, 5) for dist in dist4nbits]))
        # if i == 3:
        #     print("\t")

    mckp_p = torch.cat(mckp_p, dim=0)
    mckp_w = torch.cat(mckp_w, dim=0)
    if mckp_time is not None:
        start_time = time.time()
    x = mckp_greedy(mckp_p, mckp_w, mckp_gs, mckp_budget, sorted_weights=True)
    if mckp_time is not None:
        mckp_time[0] = time.time() - start_time
    offset = 0
    offered_cluster = []
    for i in range(len(mckp_gs)):
        nbits_dict = cons_nbits_dict if i == len(weight_list) - 1 else default_nbits_dict
        nbits = int(nbits_dict[x[offset:offset + mckp_gs[i]]])
        val_dict = val_dicts_list[i][nbits]
        offered_cluster.append(val_dict)
        res_weight_list.append(km_quantize_tensor(weight_list[i].data, nbits, val_dict=val_dict, dictnz=dictnz)[0])
        offset += mckp_gs[i]
        weight_bits.append(nbits)
        if in_place:
            weight_list[i].data.copy_(res_weight_list[i])
    # print(num_nnz)
    # print(weight_bits)
    # exit()
    return res_weight_list, weight_bits, offered_cluster


def quantize_with_bits(conv_weights, weight_bits, in_place=True, dictnz=False):
    i = 0
    km_code_res = []
    res_weight_list = []
    for w in conv_weights:
        val_dict = get_optim_val_dict(w.data, int(weight_bits[i]), niter=100, dictnz=dictnz)
        weight_quant, km_dist, km_code = km_quantize_tensor(w.data, int(weight_bits[i]), val_dict, dictnz=dictnz)
        if in_place:
            w.data.copy_(weight_quant)
        else:
            res_weight_list.append(weight_quant)
        i += 1
        km_code_res.append(km_code)
    assert i == len(weight_bits)
    return res_weight_list, km_code_res


def tensor_round(t, n=0):
    if n == 0:
        return torch.round(t)
    factor = 10.0 ** n
    return torch.round(t * factor).div_(factor)