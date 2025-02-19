import torch
import torch.nn as nn


__all__ = ['MultiDilatelocalAttention']

class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3, 4]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        # x = x.permute(0, 3, 1, 2)# B, C, H, W
        y = x.clone()
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # num_dilation,3,B,C//num_dilation,H,W
        y1 = y.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            y1[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
        y2 = y1.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        y3 = self.proj(y2)
        y4 = self.proj_drop(y3).permute(0, 3, 1, 2)
        return y4


#改进代码（加入门控）
# import torch
# import torch.nn as nn
#
# class AdaptiveGating(nn.Module):
#     def __init__(self,dim, num_classes=4, num_heads=8,head_dim=4):
#         super().__init__()
#         self.num_classes = num_classes
#         self.head_dim = head_dim
#         self.gate_weights = nn.Parameter(torch.ones(num_classes, head_dim))
#
#     def forward(self, x, labels):
#         # x: [B, head_dim, H, W]
#         # labels: [B]
#         B, d, H, W = x.shape
#         gate = self.gate_weights[labels].view(B, d, 1, 1)
#         return x * gate
#
# class DilateAttention(nn.Module):
#     "Implementation of Dilate-attention"
#
#     def __init__(self, head_dim, num_classes=4, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
#         super().__init__()
#         self.head_dim = head_dim
#         self.scale = qk_scale or head_dim ** -0.5
#         self.kernel_size = kernel_size
#         self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.adaptive_gating = AdaptiveGating(num_classes, head_dim)
#
#     def forward(self, q, k, v, labels):
#         # B, C//3, H, W
#         B, d, H, W = q.shape
#         q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
#         k = self.unfold(k).reshape(
#             [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
#                                                                                                         3)  # B,h,N,d,k*k
#         attn = (q @ k) * self.scale  # B,h,N,1,k*k
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         v = self.unfold(v).reshape(
#             [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
#                                                                                                         2)  # B,h,N,k*k,d
#         x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
#         x = self.adaptive_gating(x, labels)
#         return x
#
# class MultiDilatelocalAttention(nn.Module):
#     "Implementation of Dilate-attention"
#
#     def __init__(self, dim, num_classes=4, num_heads=8, qkv_bias=True, qk_scale=None,
#                  attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3, 4]):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_dilation = len(dilation)
#         assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
#         self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
#         self.dilate_attention = nn.ModuleList(
#             [DilateAttention(head_dim, num_classes, qk_scale, attn_drop, kernel_size, dilation[i])
#              for i in range(self.num_dilation)])
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x, labels):
#         B, C, H, W = x.shape
#         y = x.clone()
#         qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
#         y1 = y.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
#         for i in range(self.num_dilation):
#             y1[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2],4)   # B, H, W,C//num_dilation
#         y2 = y1.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
#         y3 = self.proj(y2)
#         y4 = self.proj_drop(y3).permute(0, 3, 1, 2)
#         return y4






