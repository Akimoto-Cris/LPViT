import torch
from torch.nn.utils import prune
from tools.utils import get_weights, get_modules
import numpy as np
import tools.common as common
import tools.algo as algo
import time
import os
import scipy.io as io
import tqdm
from collections import defaultdict

def weight_pruner_loader(pruner_string):
    """
    Gives you the pruning methods: LAMP, Glob, Unif, Unif+, and ERK
    """
    if pruner_string == 'lamp':
        return prune_weights_lamp
    elif pruner_string == 'glob':
        return prune_weights_global
    elif pruner_string == 'unif':
        return prune_weights_uniform
    elif pruner_string == 'unifplus':
        return prune_weights_unifplus
    elif pruner_string == 'erk':
        return prune_weights_erk
    elif pruner_string == 'rd':
        return RDPruner()
    else:
        raise ValueError('Unknown pruner')

"""
prune_weights_reparam: Allocate identity mask to every weight tensors.
prune_weights_l1predefined: Perform layerwise pruning w.r.t. given amounts.
"""


def prune_weights_reparam(model):
    module_list = get_modules(model)
    for m in module_list:
        prune.identity(m,name="weight")
        

def prune_weights_l1predefined(model,amounts):
    mlist = get_modules(model)
    for idx,m in enumerate(mlist):
        prune.l1_unstructured(m,name="weight",amount=float(amounts[idx]))
        
        
def prune_weights_l1structured(model,amounts):
    mlist = get_modules(model)
    for idx,m in enumerate(mlist):
        prune.ln_structured(m,name="weight",amount=float(amounts[idx]), n=1, dim=0) # change pruned channel to 0
                                     
    
def prune_weights_block_structured(model,amounts, blocksize=256, **kwargs):
    mlist = get_modules(model)
    for idx,m in enumerate(mlist):
        rank = kwargs.get("ranking", "l1")
        grad = kwargs["grad"][idx] if rank != "l1" else None
        mask = algo.get_mask_blockwise_fast(m.weight_orig.data, float(amounts[idx]), blocksize, rank=rank, grad=grad)
        prune.custom_from_mask(m, "weight", mask)


"""
Methods: All weights
"""    

def prune_weights_global(model,amount):
    parameters_to_prune = _extract_weight_tuples(model)
    prune.global_unstructured(parameters_to_prune,pruning_method = prune.L1Unstructured,amount=amount)

def prune_weights_lamp(model,amount):
    assert amount <= 1
    amounts = _compute_lamp_amounts(model,amount)
    print(amounts)
    prune_weights_l1predefined(model,amounts)

def prune_weights_uniform(model,amount):
    module_list = get_modules(model)
    assert amount <= 1 # Can be updated later to handle > 1.
    for m in module_list:
        prune.l1_unstructured(m,name="weight",amount=amount)

def prune_weights_unifplus(model,amount):
    assert amount <= 1
    amounts = _compute_unifplus_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)

def prune_weights_erk(model,amount):
    assert amount <= 1
    amounts = _compute_erk_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)

"""
These are not intended to be exported.
"""

def _extract_weight_tuples(model):
    """
    Gives you well-packed weight tensors for global pruning.
    """
    mlist = get_modules(model)
    return tuple([(m,'weight') for m in mlist])


@torch.no_grad()
def taylor_2nd_order_fischer_approx(delta_weight, gw, damp_lamda=1e-7, blocksize=2**13):
    flat_delta_w = delta_weight.view(-1)
    flat_gw = gw.view(-1)
    c = damp_lamda * torch.eye(blocksize,device="cuda")
    ret = torch.zeros([1],device="cuda")
    for i in range(len(flat_delta_w)//blocksize):
        dw_i = flat_delta_w[blocksize*i:blocksize*(i+1)]
        gw_i = flat_gw[blocksize*i:blocksize*(i+1)]
        for j in range(len(flat_delta_w)//blocksize):
            if i < j: continue
            dw_j = flat_delta_w[blocksize*j:blocksize*(j+1)]
            gw_j = flat_gw[blocksize*j:blocksize*(j+1)]
            ret += ((0.5 if i == j else 1) * dw_i.view(1, -1) @ ((c if i == j else 0) + torch.outer(gw_i, gw_j)) @ dw_j).squeeze()
            del dw_j, gw_j
    return ret.item()


@torch.no_grad()
def hessian_deltaw(delta_delta_weight, gw, damp_lamda=1e-7, blocksize=128):
    flat_delta_w = delta_delta_weight.view(-1)
    nonzero_idx = flat_delta_w.nonzero()
    ret = gw.view(-1, 1) @ (gw.view(-1)[nonzero_idx].view(1, -1) @ flat_delta_w[nonzero_idx])
    ret[nonzero_idx] += damp_lamda
    return ret


def gen_rd_curves_block_approx(net, loader, args, prefix=None, suffix=None):
    if prefix is None:
        path_output = ('./%s_ndz_%04d_rdcurves_block%s_opt_dist' % (args.model, args.format_blocksize, args.maxdeadzones))
    else:
        path_output = ('%s/%s_ndz_%04d_rdcurves_block%s_opt_dist/%s/' % (prefix, args.model, args.maxdeadzones, args.format_blocksize, suffix))
    
    layers = common.findconv(net, False)
    hookedlayers = common.hooklayers(layers)
    dummy_input = torch.zeros((1, 3, 224, 224)).cuda()
    _ = net(dummy_input)
    fil = [hasattr(h, "output") for h in hookedlayers]
    if False in fil:
        layers = [layers[i] for i in range(len(layers)) if fil[i]]
    if args.lambda_power > 0:
        act_inchans = [h.input[-1 if isinstance(layers[i], torch.nn.Linear) else 1] for i, h in enumerate(hookedlayers) if fil[i]]
    for l in hookedlayers:
        l.close()
        
    
    grad_list = []
    net.train()
    mlist = get_modules(net)
    for c, data in enumerate(loader):
        try:
            x = data[0]["data"]
            y = data[0]["label"]
        except:
            x, y = data
            x = x.cuda()
            y = y.cuda()
        # res = torch.mean((net(x).max(1)[0] - y) ** 2) 
        res = torch.mean(net(x) ** 2) 
        res.backward()
        for idx,m in enumerate(mlist):
            if len(grad_list) < len(mlist):
                grad_list.append(m.weight.grad.data / len(loader))
            else:
                grad_list[idx] += m.weight.grad.data / len(loader)
        for p in net.parameters():
            if p.grad is not None:
                torch.nn.init.zeros_(p.grad.data)

    print('total number of layers: %d' % (len(layers)))
    print(f'Generating RD curves to {path_output}')
    isExists=os.path.exists(path_output)
    if not isExists:
        os.makedirs(path_output)
    elif len(os.listdir(path_output)) == len(layers):
        print("found curves in", path_output)
        rd_dists, rd_amounts = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)
        if args.lambda_power > 0:
            pow_costs, _ = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output, y_tag="pow_costs_l")
            return rd_dists, pow_costs, rd_amounts, grad_list
        return rd_dists, rd_amounts, grad_list

    for l in range(0, len(layers)):
        layer_weights = layers[l].weight.clone()

    net.eval()
    rd_dists = []
    rd_amounts = []
    if args.lambda_power > 0:
        pow_costs = []
    
    if not hasattr(args, "num_parts"):
        args.num_parts = 1
        args.part_id = 0
    len_part = len(layers) // args.num_parts

    pbar = tqdm.tqdm(total=len_part)

    with torch.no_grad():
        for layerid in range(args.part_id * len_part, (args.part_id + 1) * len_part):
            layer_weights = layers[layerid].weight.clone()

            rst_amount = torch.ones(args.maxdeadzones + 1).cuda()
            rst_dist = torch.ones(args.maxdeadzones + 1).cuda()

            if args.lambda_power > 0:
                pow_costs_l = torch.ones(args.maxdeadzones + 1).cuda()

            min_amount = 0 if not args.change_curve_scale else (1 - layers[layerid].weight.count_nonzero() / layers[layerid].weight.numel())

            prev_prune_weights = None

            for d in range(args.maxdeadzones + 1):
                amount = (1. - min_amount) * d / args.maxdeadzones + min_amount
                rst_amount[d] = amount
                prune_weights = algo.pruning(layers[layerid].weight, amount, mode=args.prune_mode, blocksize=args.blocksize, rank=args.ranking, grad=grad_list[layerid].clone() if args.ranking == "taylor" else None)
                if d > 0:
                    delta_delta_weight = prune_weights - prev_prune_weights
                prev_prune_weights = prune_weights.clone()
                
                delta_weight = prune_weights - layer_weights
                gw = grad_list[layerid].clone()
                cur_dist = ((delta_weight * gw)**2).mean()

                if args.second_order:
                    if d == 0:
                        prev_second_term = taylor_2nd_order_fischer_approx(delta_weight.clone(), gw, blocksize=2**13) #2**13
                    else:
                        tmp = 0.5 * (delta_delta_weight + 2 * delta_weight).view(-1) @ hessian_deltaw(delta_delta_weight, gw, blocksize=-1)
                        prev_second_term += tmp.item()
                    cur_dist += prev_second_term

                rst_dist[d] = cur_dist

                if args.lambda_power > 0:
                    num_survived_w_blocks = amount * prune_weights.numel() / ((args.blocksize[0] * args.blocksize[1]) if isinstance(args.blocksize, (tuple, list)) else (args.blocksize ** 2))
                    pow_costs_l[d] = num_survived_w_blocks  * float(int(act_inchans[layerid] / 16))
            
            if args.smooth_curve:
                rst_dist = algo.smooth(rst_dist, args.smooth)
                # rst_dist, rst_amount, allowed = algo.refine_curve(rst_dist, rst_amount)
                # if args.lambda_power > 0:
                #     pow_costs_l = pow_costs_l[allowed]
                    
            rst_dist, rst_amount = rst_dist[None, ...], rst_amount[None, ...]
            
            save_dict = {'rd_amount': rst_amount.cpu().numpy(), 'rd_dist': rst_dist.cpu().numpy()}
            if args.lambda_power > 0:
                save_dict['pow_costs_l'] = pow_costs_l.cpu().numpy()
                
            io.savemat(('%s/%s_%03d.mat' % (path_output, args.model, layerid)), save_dict)

            rd_dists.append(rst_dist)
            rd_amounts.append(rst_amount)
            if args.lambda_power > 0:
                pow_costs.append(pow_costs_l)

            pbar.update(1)

    rd_dists, rd_amounts = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)
    if args.lambda_power > 0:
        pow_costs, _ = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output, y_tag="pow_costs_l")
        return rd_dists, pow_costs, rd_amounts, grad_list
    return rd_dists, rd_amounts, grad_list


class RDPruner:
    @staticmethod
    def recalc_lamda_power(rd_amounts, rd_dists, rd_powers):
        max_dists_derivative = np.array([algo.derivative_approx(rd_amount[0], rd_dist[0], h=10).max() for rd_amount, rd_dist in zip(rd_amounts, rd_dists)])
        max_powers_derivative = np.array([algo.derivative_approx(rd_amount[0], rd_power[0], h=10).max() for rd_amount, rd_power in zip(rd_amounts, rd_powers)])
        return max_dists_derivative.mean() / max_powers_derivative.mean()

    def __call__(self, model, amount, args, val_loader, container):
        if not hasattr(self, "amount"):    
            assert amount <= 1
            self.amount = amount
            
        sd = model.state_dict()
        new = sd.copy()
        for k, v in sd.items():
            if "weight_orig" in k:
                new[k.replace("weight_orig", "weight")] = v * sd[k.replace("weight_orig", "weight_mask")]

        container.load_state_dict(new, strict=False) #
        if not hasattr(self, "layers"):
            self.layers = common.findconv(container, False)
        target_sparsity = self.amount

        start_time_curve_gen = time.time()
        
        outputs = gen_rd_curves_block_approx(container, val_loader, args, prefix=f"./rd_curves/{args.seed}/ranking_{args.ranking}{'/second_order_approx/' if args.second_order else ''}", suffix=f'sp{target_sparsity}')
        if len(outputs) > 3:
            rd_dist, rd_power, rd_phi, grad_list = outputs
            args.lambda_power = self.recalc_lamda_power(rd_phi, rd_dist, rd_power)
            print("changing lamda_power to", args.lambda_power)
            rd_dist = [rd_l + args.lambda_power * pw_l for rd_l, pw_l in zip(rd_dist, rd_power)]
        else:
            rd_dist, rd_phi, grad_list = outputs
        
        end_time_curve_gen = time.time()
        print("Curve Generation took:", end_time_curve_gen - start_time_curve_gen, "s")
        
        min_sp = min(args.min_sp, (1 - target_sparsity) / 2)
        start_time_rd = time.time()

        args.slope = common.find_slope(container, target_sparsity, rd_dist, rd_phi, prune_mode=args.prune_mode, blocksize=args.blocksize, rank=args.ranking, grad=grad_list, flop_budget=args.flop_budget, dataset=args.dataset, min_sp=min_sp, dense_flops=args.dense_flops, h=args.h)
        print(f"Found {args.slope=}")
        
        pc_phi = algo.pareto_condition(self.layers, rd_dist, rd_phi, 2 ** args.slope, min_sp=min_sp, h=args.h)

        amounts = [min(1 - min_sp, p[0]) for p in pc_phi]
        print(amounts)

        end_time_rd = time.time()
        print("Solve RD took:", end_time_rd - start_time_rd, "s")
        print("CG & RD took:", end_time_curve_gen - start_time_curve_gen + end_time_rd - start_time_rd, "s")

        
        prune_weights_block_structured(model, amounts, blocksize=args.blocksize, rank=args.ranking, grad=grad_list if args.ranking == "taylor" else None)
        
        mask_save_path = f"./rd_curves" + \
            f"/{args.seed}" + \
            f"/sp{target_sparsity:.2f}_{args.model}_ndz_{args.maxdeadzones:04d}_rdcurves_block{args.format_blocksize}_ranking_{args.ranking}_{'secondorder' if args.second_order else ''}approx_opt_dist_mask_{'powertarget' if args.power_target else ''}.pt"
        to_save = {k: v for k, v in model.state_dict().items() if "weight_mask" in k}
        torch.save(to_save, mask_save_path)
        


def _compute_unifplus_amounts(model,amount):
    """
    Compute # of weights to prune in each layer.
    """
    amounts = []
    wlist = get_weights(model)
    unmaskeds = _count_unmasked_weights(model)
    totals = _count_total_weights(model)

    last_layer_minimum = np.round(totals[-1]*0.2) # Minimum number of last-layer weights to keep
    total_to_prune = np.round(unmaskeds.sum()*amount)
    
    if wlist[0].dim() == 4:
        amounts.append(0) # Leave the first layer unpruned.
        frac_to_prune = (total_to_prune*1.0)/(unmaskeds[1:].sum())
        if frac_to_prune > 1.0:
            raise ValueError("Cannot be pruned further by the Unif+ scheme! (first layer exception)")
        last_layer_to_surv_planned = np.round((1.0-frac_to_prune)*unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune-last_layer_to_prune)*1.0)/(unmaskeds[1:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (first+last layer exception)")
            amounts.extend([frac_to_prune_middle]*(unmaskeds.size(0)-2))
            amounts.append((last_layer_to_prune*1.0)/unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune]*(unmaskeds.size(0)-1))
    else:
        frac_to_prune = (total_to_prune*1.0)/(unmaskeds.sum())
        last_layer_to_surv_planned = np.round((1.0-frac_to_prune)*unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune-last_layer_to_prune)*1.0)/(unmaskeds[:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (last layer exception)")
            amounts.extend([frac_to_prune_middle]*(unmaskeds.size(0)-1))
            amounts.append((last_layer_to_prune*1.0)/unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune]*(unmaskeds.size(0)))
    return amounts

def _compute_erk_amounts(model,amount):
    unmaskeds = _count_unmasked_weights(model)
    erks = _compute_erks(model)

    return _amounts_from_eps(unmaskeds,erks,amount)

def _amounts_from_eps(unmaskeds,ers,amount):
    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0-amount)*unmaskeds.sum() # Total to keep.
    
    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds*(1-layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense*unmaskeds).sum()
        
        ers_of_prunables = ers*(1.0-layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables*ers_of_prunables/ers_of_prunables.sum())

        layer_to_make_dense = -1
        max_ratio = 1.0
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx]/unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx]/unmaskeds[idx]
        
        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1

    amounts = torch.zeros(num_layers)
    
    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx]/unmaskeds[idx])
    return amounts

def _compute_lamp_amounts(model,amount):
    """
    Compute normalization schemes.
    """
    unmaskeds = _count_unmasked_weights(model)
    num_surv = int(np.round(unmaskeds.sum()*(1.0-amount)))
    
    flattened_scores = [_normalize_scores(w**2).view(-1) for w in get_weights(model)]
    concat_scores = torch.cat(flattened_scores,dim=0)
    topks,_ = torch.topk(concat_scores,num_surv)
    threshold = topks[-1]
    
    # We don't care much about tiebreakers, for now.
    final_survs = [torch.ge(score,threshold*torch.ones(score.size()).to(score.device)).sum() for score in flattened_scores]
    amounts = []
    for idx,final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv/unmaskeds[idx]))
    
    return amounts

def _compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx,w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0)+w.size(1)+w.size(2)+w.size(3)
        else:
            erks[idx] = w.size(0)+w.size(1)
    return erks

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

def _normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
    new_scores[sorted_idx] = sorted_scores
    
    return new_scores.view(scores.shape)
