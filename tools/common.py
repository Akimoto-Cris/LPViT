import torch
from . import algo
import scipy.io as io
import numpy as np
import timm.models.vision_transformer as vit
import warnings
warnings.simplefilter("ignore", UserWarning)

device = torch.device("cuda:0")

rgb_avg = [0.5, 0.5, 0.5] # [0.485, 0.456, 0.406]
rgb_std = [0.5, 0.5, 0.5] # [0.229, 0.224, 0.225]

PARAMETRIZED_MODULE_TYPES = (torch.nn.Linear,
                             torch.nn.Conv2d,)

NORM_MODULE_TYPES = (torch.nn.BatchNorm2d,
                     torch.nn.LayerNorm) # eventually didn't use this



def replaceconv(net,layers,includenorm=False):
    pushconv([layers],net,includenorm,direction=1)
    return net

def findconv(net,includenorm=False):
    layers = pushconv([[]],net,includenorm)
    return layers

def getdevice():
	global device
	return device


def pushattr(layers,container,attr,includenorm,direction, prefix=""):
    if isinstance(getattr(container,attr, None), PARAMETRIZED_MODULE_TYPES) or \
            (isinstance(getattr(container, attr, None), NORM_MODULE_TYPES) and includenorm):
        # setattr(container,attr,TimeWrapper(getattr(container,attr), prefix))

        if direction == 0:
            layers[0].append(getattr(container,attr))
        else:
            setattr(container,attr,layers[0][0])
            layers[0] = layers[0][1:len(layers[0])]
    # print(container.__class__.__name__, attr)

def pushlist(layers,container,attr,includenorm,direction, prefix=""):
    if isinstance(container[attr], PARAMETRIZED_MODULE_TYPES) or \
            (isinstance(container[attr], NORM_MODULE_TYPES) and includenorm):
        # container[attr] = TimeWrapper(container[attr], prefix)
        if direction == 0:
            layers[0].append(container[attr])
        else:
            container[attr] = layers[0][0]
            layers[0] = layers[0][1:len(layers[0])]
    else:
        pushconv(layers,container[attr],includenorm, direction, prefix=prefix)

def pushconv(layers, container, includenorm=True, direction=0, prefix="model"):
    if isinstance(container, vit.PatchEmbed):
        pushattr(layers,container,'proj',includenorm,direction, prefix=prefix+".proj")
        
    elif isinstance(container, vit.Block):
        # pushattr(layers, container, "norm1", includenorm, direction)
        pushconv(layers, container.attn, includenorm, direction, prefix=prefix+".attn")
        # pushattr(layers, container, "norm2", includenorm, direction)
        pushconv(layers, container.mlp, includenorm, direction, prefix=prefix+".mlp")
        
    elif isinstance(container, vit.Attention):
        pushattr(layers, container, "qkv", includenorm, direction, prefix=prefix+".qkv")
        pushattr(layers, container, "proj", includenorm, direction, prefix=prefix+".proj")
    
    elif isinstance(container, vit.Mlp):
        pushattr(layers, container, "fc1", includenorm, direction, prefix=prefix+".fc1")
        pushattr(layers, container, "fc2", includenorm, direction, prefix=prefix+".fc2")
    else:
        return [m for m in container.modules() if isinstance(m, PARAMETRIZED_MODULE_TYPES)]

    return layers[0]


def replacelayer(module, layers, classes):
    module_output = module
    # base case
    if isinstance(module, classes):
        module_output, layers[0] = layers[0][0], layers[0][1:]
    # recursive
    for name, child in module.named_children():
        module_output.add_module(name, replacelayer(child, layers, classes))
    del module
    return module_output


def loadvarstats(archname,testsize):
    mat = io.loadmat(('%s_stats_%d.mat' % (archname, testsize)))
    return np.array(mat['cov'])


def findrdpoints(y_sse,delta,coded,lam_or_bit, is_bit=False):
    # find the optimal quant step-size
    y_sse[np.isnan(y_sse)] = float('inf')
    ind1 = np.nanargmin(y_sse,1)
    ind0 = np.arange(ind1.shape[0]).reshape(-1,1).repeat(ind1.shape[1],1)
    ind2 = np.arange(ind1.shape[1]).reshape(1,-1).repeat(ind1.shape[0],0)
    inds = np.ravel_multi_index((ind0,ind1,ind2),y_sse.shape) # bit_depth x blocks
    y_sse = y_sse.reshape(-1)[inds]
    delta = delta.reshape(-1)[inds]
    coded = coded.reshape(-1)[inds]
    # mean = mean.reshape(-1)[inds]
    # find the minimum Lagrangian cost
    if is_bit:
        point = coded == lam_or_bit
    else:
        point = y_sse + lam_or_bit*coded == (y_sse + lam_or_bit*coded).min(0)
    return np.select(point, y_sse), np.select(point, delta), np.select(point, coded)#, np.select(point, mean)


def predict2(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in iter(loader):
            x = x.to(device)
            y_hat = torch.cat((y_hat,net(x)))
    return y_hat


def predict2_withgt(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    y_gt = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, y in iter(loader):
            x = x.to(device)
            y = y.to(device).float()    
            y_hat = torch.cat((y_hat,net(x)))
            y_gt = torch.cat((y_gt, y))
    return y_hat, y_gt

def predict2_activation(net, loader, layerhook):
    global device
    acts = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in iter(loader):
            x = x.to(device)
            _ = net(x)
            acts = torch.cat((acts, layerhook.output_tensor))
    return acts


def predict_dali(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            x = data[0]["data"]
            res = net(x)
            y_hat = torch.cat((y_hat,res))
    return y_hat


def predict_dali_withgt(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    y_gt = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            x = data[0]["data"]
            y_hat = torch.cat((y_hat,net(x)))
            y = data[0]["label"]
            y_gt = torch.cat((y_gt, y))
    return y_hat, y_gt


def predict_dali_activation(net, loader, layerhook):
    global device
    acts = torch.zeros(0, device=device)
    #loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            x = data[0]["data"]
            _ = net(x)
            acts = torch.cat((acts, layerhook.output_tensor))
    return acts


import math 
@torch.no_grad()
def predict_tensor(net, X, batchsize=128):
    global device
    y_hat = torch.zeros(0, device=device)
    for b in range(math.ceil(len(X) / batchsize)):
        y_hat = torch.cat((y_hat,net(X[b * batchsize: (b+1) * batchsize])))
    return y_hat

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #pred.reshape(pred.shape[0], -1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def ternary_search(min_val, max_val, target_func, max_iters=20):
    l = min_val
    r = max_val
    cnt = 0
    while l < r:
        y_l = target_func(l)
        y_r = target_func(r)
        if y_l > y_r:
            l = l + (r-l)/3
        else:
            r = r - (r-l)/3
        
        cnt += 1
        if cnt >= max_iters:
            break
    return l


def binary_search(min_val, max_val, target_func, target_val, epsilon=0.02, max_iters=40):
    l = min_val
    r = max_val
    cnt = 0
    while l < r:
        mid = (l + r) / 2
        y_mid = target_func(mid)

        if abs(y_mid - target_val) <= epsilon:
            return mid
        elif y_mid < target_val:
            l = mid
        elif y_mid > target_val:
            r = mid
        
        cnt += 1
        if cnt >= max_iters:
            y_l = target_func(l)
            y_r = target_func(r)
            if abs(y_mid - target_val) > abs(y_l - target_val) and abs(y_r - target_val) > abs(y_l - target_val):
                mid = l
            elif abs(y_mid - target_val) > abs(y_r - target_val) and abs(y_l - target_val) > abs(y_r - target_val):
                mid = r
            break
    return mid



def find_slope(model, target_sparsity, rd_dist, rd_amount, layers=None, prune_mode="unstructured", flop_budget=False, **kwargs):
    layers = layers or findconv(model, False)
    if flop_budget:
        layer_weights = [layer.weight.clone() for layer in layers]

    def target_func(slope):
        if not flop_budget:
            total_n_weights = 0
            survived_n_weights = 0
        else:
            layer_weights = [layer.weight.clone() for layer in layers]
        pc_amount = algo.pareto_condition(layers, rd_dist, rd_amount, 2 ** slope, min_sp=kwargs["min_sp"], h=kwargs.get("h", 10))
        # print(pc_amount, slope)

        for i in range(0, len(layers)):
            prune_weights = algo.pruning(layers[i].weight.clone(), pc_amount[i][0], blocksize=kwargs.get("blocksize", -1), mode=prune_mode, rank=kwargs.get("rank", "l1"), 
                                        grad=kwargs["grad"][i] if kwargs.get("rank", "l1") == "taylor" else None)
            if not flop_budget:
                total_n_weights += prune_weights.numel()
                survived_n_weights += (prune_weights != 0).sum().float()
            else:
                layers[i].weight.data = prune_weights

        if flop_budget:
            flops = get_model_flops(model, kwargs["dataset"]) / kwargs["dense_flops"]
            for layer, ori_weight in zip(layers, layer_weights):
                layer.weight.data = ori_weight
            return -flops
        ret = 1 - survived_n_weights / total_n_weights
        return ret
    
    return binary_search(-100, 100, target_func, -target_sparsity if flop_budget else target_sparsity)


def prune_layerbylayer(model, layer_id, amount):
    layers = findconv(model, False)
    layers[layer_id].amount = amount
    
def hooklayers(layers):
    return [Hook(layer) for layer in layers]

class Hook:
    def __init__(self, module, backward=False):
        self.backward = backward
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input_tensor = input[0]
        self.output_tensor = output if not self.backward else output[0]
        self.input = torch.tensor(self.input_tensor.shape[1:])
        self.output = torch.tensor(self.output_tensor.shape[1:])

    def close(self):
        self.hook.remove()


from tools.pruners import get_weights, get_modules

@torch.no_grad()
def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(m.weight.count_nonzero())
    return torch.FloatTensor(unmaskeds)

@torch.no_grad()
def _count_total_weights(model):
    """
    Return a 1-dimensional tensor of #total weights.
    """
    wlist = get_weights(model)
    numels = []
    for w in wlist:
        numels.append(w.numel())
    return torch.FloatTensor(numels)

@torch.no_grad()
def get_model_flops(net, dataset="cifar"):
    net.eval()
    if dataset == "cifar":
        dummy_input = torch.zeros((1, 3, 32, 32), device=next(net.parameters()).device)
    else:
        dummy_input = torch.zeros((1, 3, 224, 224), device=next(net.parameters()).device)

    layers = findconv(net, False)
    unmaskeds = _count_unmasked_weights(net)

    hookedlayers = hooklayers(layers)
    _ = net(dummy_input)
    fil = [hasattr(h, "output") for h in hookedlayers]
    if False in fil:
        layers = [layers[i] for i in range(len(layers)) if fil[i]]
        hookedlayers = [hookedlayers[i] for i in range(len(hookedlayers)) if fil[i]]
        unmaskeds = [unmaskeds[i] for i in range(len(unmaskeds)) if fil[i]]

    output_dimens = [hookedlayers[i].output for i in range(0, len(hookedlayers))]
    for l in hookedlayers:
        l.close()

    nom_flops = 0.0

    for o_dim, surv, m in zip(output_dimens, unmaskeds, layers):
        if isinstance(m, torch.nn.Conv2d):
            nom_flops += o_dim[-2:].prod() * surv + (0 if m.bias is None else o_dim.prod())
        elif isinstance(m, torch.nn.Linear):
            nom_flops += surv * o_dim[0] + (0 if m.bias is None else o_dim.prod())

    return nom_flops
