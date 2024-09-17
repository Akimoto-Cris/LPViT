import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_attention_mask(n_tokens, stride_length=3, c=2, mode="fixed"):
    x = torch.arange(start=0, end=n_tokens).reshape(-1, 1).cuda()
    y = x.T
    z = torch.zeros((n_tokens,n_tokens)).cuda()
    Q = z + x
    K = z + y

    if mode == "fixed":
        fixed_mask_1 = (Q//stride_length == K//stride_length)
        fixed_mask_2 = (((K % stride_length) >= stride_length-c) & ((K % stride_length) <= stride_length))
        combined_mask = (fixed_mask_1 | fixed_mask_2)
    elif mode == "strided":
        stride_mask_1 = ((Q-K).abs() <= stride_length)
        stride_mask_2 = ((Q-K) % stride_length == 0)
        combined_mask =  (stride_mask_1 | stride_mask_2)

    return combined_mask

    
class SparseAttentionWrapper(nn.Module):
    def __init__(self, attn_mode, attn_module, N):
        super(SparseAttentionWrapper, self).__init__()
        self.attn_module = attn_module
        self.attn_mode = attn_mode
        self.mask = sparse_attention_mask(N, stride_length=14, mode=attn_mode)
        self.num_heads = self.attn_module.num_heads
        for name, module in reversed(attn_module._modules.items()):
            self._modules[name] = module
            
    def _fixed_factorized_attention(self, q, k, v, stride_length=14, c=2):
        B, H, N, C = q.shape
        pad_N = (N // stride_length) * stride_length - N
        q_padded = F.pad(q, (1, 0, 2, pad_N))
        k_padded = F.pad(k, (1, 0, 2, pad_N))
        v_padded = F.pad(v, (1, 0, 2, pad_N))
        q_blocked = q_padded.view(B, -1, stride_length, C)
        k_blocked = k_padded.view(B, -1, stride_length, C)
        v_blocked = v_padded.view(B, -1, stride_length, v.shape[-1])

        # out = 
        x_ = F.scaled_dot_product_attention(
            q_blocked, k_blocked[:, :, :-c, :], v_blocked[:, :, :-c, :],
            dropout_p=self.attn_module.attn_drop.p if self.training else 0.,
            scale=self.attn_module.scale)
        
        x_c = F.scaled_dot_product_attention(
            q_padded, k_blocked[:, :, -c:, :].view(B, H, -1, C), v_blocked[:, :, -c:, :].view(B, H, -1, v.shape[-1]),
            dropout_p=self.attn_module.attn_drop.p if self.training else 0.,
            scale=self.attn_module.scale)

        out = torch.cat([x, x_c[:, -1, c, v.shape[-1]]], dim=-2)
        return out.view(B, H, N, -1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.attn_module.qkv(x).reshape(B, N, 3, self.attn_module.num_heads, C // self.attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_module.attn_drop.p if self.training else 0.,
            attn_mask=self.mask
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.attn_module.proj(x)
        x = self.attn_module.proj_drop(x)
        return x



def get_spraseattention_vit_module(model, N, attn_mode):
    for name, module in reversed(model._modules.items()):
    
        if len(list(module.children())) > 0:
            model._modules[name] = get_spraseattention_vit_module(model=module, N=N, attn_mode=attn_mode)

        if hasattr(module, "qkv"):
            if attn_mode == "block":
                model._modules[name] = BlockSparseAttentionWrapper(module, N)
            else:
                model._modules[name] = SparseAttentionWrapper(attn_mode, module, N)

    return model

import math

def blocksparse_attention_mask(N, blocksize=32):
    mask = torch.zeros((N, N), device="cuda")
    for i in range(0, math.ceil(N / blocksize)):
        for j in range(0, math.ceil(N / blocksize)):
            if i == j:
                mask[i * blocksize: (i+1) * blocksize, j * blocksize: (j+1) * blocksize] = 1
    return mask


class BlockSparseAttentionWrapper(nn.Module):
    blocksize = 32
    def __init__(self, attn_module, N):
        super(BlockSparseAttentionWrapper, self).__init__()
        self.attn_module = attn_module
        self.mask = blocksparse_attention_mask(N, blocksize=self.blocksize)
        self.num_heads = self.attn_module.num_heads
        for name, module in reversed(attn_module._modules.items()):
            self._modules[name] = module
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.attn_module.qkv(x).reshape(B, N, 3, self.attn_module.num_heads, C // self.attn_module.num_heads).permute(2, 0, 3, 1, 4) # [3, B, H, N, C]
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_module.attn_drop.p if self.training else 0.,
            attn_mask=self.mask
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.attn_module.proj(x)
        x = self.attn_module.proj_drop(x)
        return x
    