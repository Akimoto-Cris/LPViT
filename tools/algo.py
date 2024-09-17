import torch
import numpy as np
import scipy.io as io

from .common import *
from itertools import product


eps = 1e-12


def get_num_input_channels(tensor_weights):
    return tensor_weights.size()[1]

def get_ele_per_input_channel(tensor_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    return tensor_weights[:, 0, :, :].numel()

def get_input_channels(tensor_weights, st_id, ed_id):
    weights_copy = tensor_weights.clone()
    if len(weights_copy.shape) == 2:
        weights_copy = weights_copy[..., None, None]
    return weights_copy[:, st_id:ed_id, :, :]

def assign_input_channels(tensor_weights, st_id, ed_id, quant_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    tensor_weights[:, st_id:ed_id, :, :] = quant_weights
    return

def get_num_output_channels(tensor_weights):
    return tensor_weights.size()[0]

def get_ele_per_output_channel(tensor_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    return tensor_weights[0, :, :, :].numel()

def get_output_channels(tensor_weights, st_id, ed_id):
    weights_copy = tensor_weights.clone()
    if len(weights_copy.shape) == 2:
        weights_copy = weights_copy[..., None, None]
    return weights_copy[st_id:ed_id, :, :, :]

def get_output_channels_inds(tensor_weights, inds):
    weights_copy = tensor_weights.clone()
    return weights_copy[inds]

def assign_output_channels(tensor_weights, st_id, ed_id, quant_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    tensor_weights[st_id:ed_id, :, :, :] = quant_weights


def assign_output_channels_inds(tensor_weights, inds, quant_weights):
    tensor_weights[inds, ...] = quant_weights


#####################

def reshape_tensor_for_blockwise(tensor_weights):
    if len(tensor_weights.shape) == 2:
        return tensor_weights
    elif len(tensor_weights.shape) == 4:
        return tensor_weights.view(tensor_weights.shape[0], -1)

def tuple2tensor(block_size):
    return torch.tensor(block_size) if isinstance(block_size, (tuple, list)) else block_size

def get_num_blocks(tensor_weights, block_size=8):
    return torch.tensor(reshape_tensor_for_blockwise(tensor_weights.clone()).shape).float().div(tuple2tensor(block_size)).ceil().long()


def pad_weight_blockwise(tensor_weights, block_size=8):
    tensor_weights = reshape_tensor_for_blockwise(tensor_weights.clone())
    if (torch.tensor(tensor_weights.shape) % block_size == 0).bool().all():
        return tensor_weights
    
    n_blocks = get_num_blocks(tensor_weights, block_size)
    new_weight = torch.ones(tuple(n_blocks * tuple2tensor(block_size)), dtype=tensor_weights.dtype, device=tensor_weights.device)
    new_weight[:tensor_weights.shape[0], :tensor_weights.shape[1]] = tensor_weights
    return new_weight
    

def get_block(tensor_weights, st_id_x, ed_id_x, st_id_y, ed_id_y):
    weights_copy = reshape_tensor_for_blockwise(tensor_weights.clone())
    return weights_copy[st_id_x:ed_id_x, st_id_y:ed_id_y]

def assign_block(tensor_weights, st_id_x, ed_id_x, st_id_y, ed_id_y, quant_weights):
    reshape_tensor_for_blockwise(tensor_weights)[st_id_x:ed_id_x, st_id_y:ed_id_y, :, :] = quant_weights
    
    
def pruning(data, amount, blocksize=-1, mode='unstructured', rank="l1",grad=None):
    if amount <= 0:
        return data
    
    if mode in ["structured", "structured_approx"]:
        return data * get_mask_structured(data, amount, rank=rank, grad=grad)
    elif mode in ["blockwise", "blockwise_approx"] and blocksize != -1:
        mask = get_mask_blockwise_fast(data, amount, blocksize, rank=rank, grad=grad)
        return data * mask
    return data * get_mask(data, amount, rank=rank, grad=grad)

@torch.no_grad()
def get_mask(data, amount, rank="l1", grad=None):
    if rank == "l1":
        data = data.abs()
    elif rank == "taylor":
        data = (data * grad).abs_()
        
    mask = torch.ones_like(data).to(data.device)
    if amount <= 0:
        return mask
    assert amount <= 1
    k = int(amount * data.numel())
    if not k:
        print(k)
        return mask

    topk = torch.topk(torch.abs(data).view(-1), k=k, largest=False, sorted=False)
    mask.view(-1)[topk.indices] = 0
    return mask

@torch.no_grad()
def get_mask_blockwise_fast(data, amount, blocksize=8, rank="l1", grad=None):
        
    if rank == "l1":
        data = data.abs()
    elif rank == "taylor":
        data = (data * grad).abs_()
        
    if amount <= 0:
        return torch.ones_like(data).to(data.device)
    assert amount <= 1    
    n_blocks = get_num_blocks(data, blocksize)
    ori_shape = data.shape
    data = reshape_tensor_for_blockwise(data.clone())
    
    l1_norm_2d = torch.nn.AvgPool2d(blocksize, stride=blocksize, ceil_mode=True)(data[None, None, ...])
    l1_norm = l1_norm_2d.flatten()
    assert len(l1_norm) == n_blocks.prod()
    
    k = int((1-amount) * n_blocks.prod())
    mask = torch.zeros_like(l1_norm, device=l1_norm.device)
    topk = torch.topk(l1_norm, k=k, largest=True)
    slc = [slice(None)]
    slc[0] = topk.indices
    mask[slc] = 1.
    mask = torch.nn.functional.interpolate(mask.view(l1_norm_2d.shape), scale_factor=blocksize).view(*(n_blocks * tuple2tensor(blocksize)))[:data.shape[0], :data.shape[1]]
    if len(ori_shape) == 4:
        mask = mask.view(ori_shape)
    return mask
    

@torch.no_grad()
def get_mask_indices_blockwise_fast(data, amount, blocksize=8, rank="l1", grad=None):
    if rank == "l1":
        data = data.abs()
    elif rank == "taylor":
        data = (data * grad).abs_()

    if amount <= 0:
        return torch.ones_like(data).to(data.device)
    assert amount <= 1    
    n_blocks = get_num_blocks(data, blocksize)
    ori_shape = data.shape
    data = reshape_tensor_for_blockwise(data.clone())
    
    l1_norm_2d = torch.nn.AvgPool2d(blocksize, stride=blocksize, ceil_mode=True)(data[None, None, ...])
    l1_norm = l1_norm_2d.flatten()
    assert len(l1_norm) == n_blocks.prod()
    
    k = int((1-amount) * n_blocks.prod())
    mask = torch.zeros_like(l1_norm, device=l1_norm.device)
    topk_ind = torch.topk(l1_norm, k=k, largest=True).indices
    topk_ind = torch.stack([topk_ind // l1_norm_2d.shape[0], topk_ind % l1_norm_2d.shape[0]], 1)
    return topk_ind


    
@torch.no_grad()
def get_mask_structured(data, amount, rank="l1", grad=None):
    if rank == "l1":
        data = data.abs()
    elif rank == "taylor":
        data = (data * grad).abs_()

    if amount <= 0:
        return torch.ones_like(data).to(data.device)
    assert amount <= 1
    ori_shape = data.shape
    nchannels = ori_shape[0]
    if len(data.shape) == 2:
        data = data.clone()[..., None, None]
    
    l1_norm = torch.norm(data, p=1, dim=(1,2,3))
    k = int((1-amount) * nchannels)
    mask = torch.zeros_like(data, device=data.device)
    topk = torch.topk(l1_norm, k=k, largest=True)
    slc = [slice(None)] * 4
    slc[0] = topk.indices
    mask[slc] = 1
    
    return mask.view(ori_shape)

def get_mask_structured_origin(data, amount, rank="l1", grad=None):
    if rank == "l1":
        data = data.abs()
    elif rank == "taylor":
        data = (data * grad).abs_()

    if amount <= 0:
        return torch.ones_like(data).to(data.device)
    assert amount <= 1
    ori_shape = data.shape
    nchannels = ori_shape[1]
    if len(data.shape) == 2:
        data = data.clone()[..., None, None]
    
    l1_norm = torch.norm(data, p=1, dim=(0,2,3))
    k = int((1-amount) * nchannels)
    mask = torch.zeros_like(data, device=data.device)
    topk = torch.topk(l1_norm, k=k, largest=True)
    slc = [slice(None)] * 4
    slc[1] = topk.indices
    mask[slc] = 1
    
    return mask.view(ori_shape)


def load_rd_curve_batch(archname, layers, maxprunerates, datapath, nchannelbatch):
    nlayers = len(layers)
    rd_dist = []
    rd_phi = []

    for l in range(0, nlayers):
        nchannels = get_num_output_channels(layers[l].weight)
        nbatch = nchannels // nchannelbatch
        if (nchannels % nchannelbatch) != 0:
            nbatch += 1
        rd_dist_l = []
        rd_phi_l = []
        for f in range(0, nchannels, nchannelbatch):
            matpath = ('%s/%s_%03d_%04d.mat' % (datapath, archname, l, f))
            mat = io.loadmat(matpath)
            rd_dist_l.append(mat['rd_dist'][0])
            rd_phi_l.append(mat['rd_phi'][0])
        rd_dist.append(np.array(rd_dist_l))
        rd_phi.append(np.array(rd_phi_l))
    
    return rd_dist, rd_phi


def load_rd_curve(archname, layers, maxprunerates, datapath, y_tag="rd_dist"):
    nlayers = len(layers)
    rd_dist = []
    rd_phi = []

    for l in range(0, nlayers):
        rd_dist_l = []
        rd_phi_l = []
        
        matpath = ('%s/%s_%03d.mat' % (datapath, archname, l))
        mat = io.loadmat(matpath)
        rd_dist_l.append(mat[y_tag][0])
        rd_phi_l.append(mat['rd_amount'][0])
        rd_dist.append(np.array(rd_dist_l))
        rd_phi.append(np.array(rd_phi_l))
    
    return rd_dist, rd_phi


def cal_total_num_weights(layers):
    nweights = 0
    nlayers = len(layers)

    for l in range(0, nlayers):
        n_filter_elements = layers[l].weight.numel()
        nweights += n_filter_elements

    return nweights


def derivative_approx(xs, ys, h=10):
    results = []
    for i in range(len(xs)-1):
        h_ = min(h, min(i, len(xs)-i-1))
        if h_ < 2 and i < 2:
            results.append((ys[i+1]-ys[i])/(xs[i+1]-xs[i]))
        elif h_ < 2 and len(xs) - i < 3:
            results.append((ys[i+1]-ys[i-1])/(xs[i+1]-xs[i-1]))
        else:
            results.append((-ys[i+int(h_)]+8*ys[i+int(h_/2)]-8*ys[i-int(h_/2)]+ys[i-int(h_)])/(6*(xs[i+int(h_)]-xs[i])))
    # return (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    return np.array(results)



def pareto_condition_batch(layers, rd_dist, rd_phi, slope_lambda, nchannelbatch):
    nlayers = len(layers)
    pc_phi = [0] * nlayers
    pc_dist_sum = 0
    for l in range(0, nlayers):
        nchannels = get_num_output_channels(layers[l].weight)
        n_channel_elements = get_ele_per_output_channel(layers[l].weight)
        nbatch = nchannels // nchannelbatch
        if (nchannels % nchannelbatch) != 0:
            nbatch += 1
        pc_phi[l] = [0] * nbatch
        cnt = 0

        for f in range(0, nchannels, nchannelbatch):
            st_layer = f
            ed_layer = f + nchannelbatch
            if f + nchannelbatch > nchannels:
                ed_layer = nchannels
            pr = np.argmin(np.abs(derivative_approx(rd_phi[l][cnt, :], rd_dist[l][cnt, :]) - slope_lambda))
            # pr = 9
            pc_phi[l][cnt] = rd_phi[l][cnt, pr]
            pc_dist_sum += rd_dist[l][cnt, pr]
            cnt = cnt + 1

    return pc_phi


def pareto_condition(layers, rd_dist, rd_phi, slope_lambda, min_sp=0., h=10):
    nlayers = len(layers)
    pc_phi = [0] * nlayers
    for l in range(0, nlayers):
        pc_phi[l] = [0]
        cnt = 0
        # y_0 = -slope_lambda * rd_phi[l][cnt, :] + rd_dist[l][cnt, :]
        # pr = int(np.argmin(y_0))
        d = derivative_approx(rd_phi[l][cnt, :], rd_dist[l][cnt, :], h=h)
        pr = np.argmin(np.abs(d - slope_lambda))
        # pr = np.argsort(np.abs(d - slope_lambda))[:5].max()
        # if not max(d - slope_lambda) * min(d - slope_lambda) < 0:
        #    print("warning:", d, slope_lambda)
        pc_phi[l][cnt] = min(1-min_sp, rd_phi[l][cnt, pr])

    return pc_phi



def refine_curve(rd_dists, rd_amounts):
    allowed = []
    cur_min = 1e10
    for i in range(len(rd_amounts)):
        if rd_dists[i] <= cur_min and 0 <= rd_amounts[i] <= 1: 
            allowed.append(i)
            cur_min = rd_dists[i]
    
    return rd_dists[allowed], rd_amounts[allowed], allowed
    
def make_nondecreasing(rd_dists, rd_amounts):
    for i in range(1, len(rd_dists)):
        rd_dists[i] = max(rd_dists[i-1], rd_dists[i])
    return rd_dists, rd_amounts

def make_nonincreasing(rd_dists, rd_amounts):
    for i in reversed(range(len(rd_dists)-1)):
        rd_dists[i] = min(rd_dists[i+1], rd_dists[i])
    return rd_dists, rd_amounts


def smooth(rd_amounts, weight=0.9):
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = torch.zeros_like(rd_amounts, device=rd_amounts.device)
    num_acc = 0
    for next_val in rd_amounts:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - weight ** num_acc
        smoothed_val = last / debias_weight
        smoothed[num_acc-1] = smoothed_val

    return smoothed