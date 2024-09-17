import torch,argparse,random
import torch.nn.functional as F
import numpy as np
from tools import *
import tools.common as common
import timm
from tools.pruners import prune_weights_reparam



""" ARGUMENT PARSING """
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--cuda', type=int, help='cuda number', default=0)
parser.add_argument('--model', type=str, help='network')
parser.add_argument('--pruner', type=str, help='pruning method', default="rd")
parser.add_argument('--dataset', type=str, choices=['cifar', 'imagenet'], default='imagenet')
parser.add_argument('--data_path', type=str, default='/imagenet/', help='path to data')
parser.add_argument('--maxdeadzones', type=int, default=100)
parser.add_argument('--amount', type=float, default=0.5, help='pruning target flops/sparsity rate')
parser.add_argument('--blocksize', type=int, nargs="+", default=-1)
parser.add_argument('--ranking', type=str, default="taylor", choices=["l1", "taylor"])
parser.add_argument('--prune_mode', '-pm', type=str, default='blockwise_approx', choices=['unstructured', 'structured', 'unstructured_approx', 'structured_approx', 'blockwise', 'blockwise_approx'])
parser.add_argument('--calib_size', type=int, default=2000)
parser.add_argument('--smooth_curve', action="store_true")
parser.add_argument('--bspline_smooth_curve', action="store_true")
parser.add_argument('--change_curve_scale', action="store_true")
parser.add_argument('--worst_case_curve', '-wcc', action="store_true")
parser.add_argument('--smooth', type=float, default=0.5)
parser.add_argument('--h', type=int, default=4)
parser.add_argument('--lambda_power', type=float, default=0., help="beta in the paper, to turn on, set to any value other than 0 and the program will automatically search for the true value for beta.")
parser.add_argument('--second_order', action="store_true")
parser.add_argument('--power_target', action="store_true", help="don't touch this")
parser.add_argument('--flop_budget', action="store_true", help="use flop as the targeting budget in binary search instead of sparsity. if true, `amounts` and `target_sparsity` variables in the codes will represent flops instead")
parser.add_argument('--min_sp', type=float, default=0.05)
args = parser.parse_args()

""" SET THE SEED """
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.blocksize != -1:
    if 'blockwise' not in args.prune_mode:
        args.prune_mode = 'blockwise'
    assert len(args.blocksize) < 3, "please give no longer than 2 blocksize dims for 2D matmul dimensions"
    args.format_blocksize = "x".join(map(str, args.blocksize))
    if len(args.blocksize) == 1:
        args.blocksize = args.blocksize[0]
else:
    args.format_blocksize = ""
DEVICE = args.cuda


def test(model,loader):
    model.eval()
    device = next(model.parameters()).device
    
    correct = 0
    loss    = 0
    total   = 0
    for i,data in enumerate(loader):
        try:
            x, y = data
            x = x.to(device)
            y = y.to(device)
        except:
            x = data[0]["data"]
            y = data[0]["label"].long()[:, 0]
        with torch.no_grad():
            yhat    = model(x)
            _,pred  = yhat.max(1)
        correct += pred.eq(y).sum().item()
        loss += F.cross_entropy(yhat,y)*len(x)
        total += len(x)
    
    if hasattr(loader, "reset"):
        loader.reset()

    acc     = correct/total * 100.0
    loss    = loss/total
    
    model.train()
    
    return [acc,loss]



def model_and_opt_loader(model_string, DEVICE, reparam=True):
    if DEVICE == None:
        raise ValueError('No cuda device!')
    if "vit" in model_string or "deit" in model_string or "swin" in model_string:
        model = timm.create_model(model_string, pretrained=True).to(DEVICE)
        batch_size = 64 
    else:
        raise ValueError(f'Unknown model: {model_string}')
    if reparam:
        prune_weights_reparam(model)
    return model, batch_size




""" IMPORT LOADERS/MODELS/PRUNERS/TRAINERS"""
model, batch_size = model_and_opt_loader(args.model, DEVICE, reparam=True)
_, test_loader, calib_loader = dataset_loader(args.model,batch_size=batch_size, args=args)


pruner = weight_pruner_loader(args.pruner)
container = model_and_opt_loader(args.model, DEVICE, False)[0]

dense_flops = common.get_model_flops(model, args.dataset)

print(f"Pruning method: {args.pruner}")
flops = common.get_model_flops(model, args.dataset) / dense_flops

print(f"Before prune: FLOPs: {flops}")
if args.flop_budget:
    args.dense_flops = dense_flops
if args.pruner == "rd":
    pruner(model, args.amount, args, calib_loader, container)
else:
    pruner(model, args.amount)    

flops = common.get_model_flops(model, args.dataset) / dense_flops
sparse = utils.get_model_sparsity(model)
print(f"sparsity: {sparse}")
print(f"FLOPs: {flops}")

# torch.save(model.state_dict(), os.path.join(DICT_PATH,args.pruner + '.pth.tar'))
print(f"Pruned test accuracy: {test(model, test_loader)[0]}")