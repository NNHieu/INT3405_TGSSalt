from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
# from .general import LinearWithChannel
import math

from model.activation import Mish

class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class FPA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPA, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     Mish())
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     Mish())

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     Mish())
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     Mish())

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   Mish())

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(x_glob, scale_factor=8, mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x

class ScaleDotProductAttention(nn.Module):
    """ 
    """
    def __init__(self, input_size, identity=False, key_dim_reduction=2):
        """
        
        """
        super(ScaleDotProductAttention, self).__init__()
        self.input_size = input_size
        self.identity = identity
        self.dk = input_size // key_dim_reduction
        if not identity:
            self.linear_Q = nn.Linear(input_size, self.dk)
            self.linear_K = nn.Linear(input_size, self.dk)
            # self.linear_V = nn.Linear(input_size, input_size)

    def forward(self, Q, K, V, K_mask=None):
        """
        Args:
            Q: batch * len1 * hdim
            K: batch * len2 * hdim
            V: batch * len2 * hv_dim
            K_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        bs, len_q, dqk = Q.size()
        len_k = K.size(1)
        if not self.identity:
            Q = self.linear_Q(Q.view(-1, Q.size(-1))).view(bs, len_q, self.dk)
            K = self.linear_K(K.view(-1, K.size(-1))).view(bs, len_k, self.dk)
            # V = self.linear_V(V.view(-1, K.size(-1))).view(V.size())
        # Compute scores and scale
        scores = Q.bmm(K.transpose(2, 1)) / math.sqrt(self.input_size)
        # Mask padding
        if K_mask is not None:
            K_mask = K_mask.unsqueeze(1).expand(scores.size()) # shape(batch, len1, len2)
            scores.data.masked_fill_(K_mask.data.eq(1), -float('inf'))
        # Normalize
        alpha_flat = torch.sigmoid(scores.view(-1, K.size(1))) #shape(batch*len1, len2)
        alpha = alpha_flat.view(-1, Q.size(1), K.size(1)) # shape(batch, len1, len2)
        # Tính attn vô value
        matched_seq = alpha.bmm(V) # shape(batch, len1, hv_dim)
        return matched_seq

# class Multihead(nn.Module):
#     def __init__(self, num_head, qk_feature_size, v_feature_size, out_size):
#         super(Multihead, self).__init__()
#         self.num_head = num_head
#         self.qk_feature_size = qk_feature_size
#         self.v_feature_size = v_feature_size
#         self.proj_Q = LinearWithChannel(qk_feature_size, qk_feature_size, num_head)
#         self.proj_K = LinearWithChannel(qk_feature_size, qk_feature_size, num_head)
#         self.proj_V = LinearWithChannel(v_feature_size, v_feature_size, num_head)
#         self.linear_out = nn.Linear(num_head*v_feature_size, out_size)
#     def forward(self, Q, K, V, K_mask):
#         assert K.size(1) == V.size(1)
#         num_batch = Q.size(0)
#         dq    = Q.size(2)
#         dk    = K.size(2)
#         dv    = V.size(2)
#         qlen  = Q.size(1)
#         klen  = vlen = K.size(1)
#         assert dq == dk and self.qk_feature_size == dq and self.v_feature_size == dv
#         assert len(K_mask.size()) == 2 and K_mask.size(0) == num_batch and K_mask.size(1) == klen


#         Q_proj = Q.view(-1, Q.size(-1))
#         Q_proj = Q_proj.unsqueeze(0).expand(self.num_head, *Q_proj.size())
#         assert Q_proj.size(0) == self.num_head and Q_proj.size(1) == num_batch*qlen and Q_proj.size(2) == dq
#         Q_proj = self.proj_Q(Q_proj)
#         assert Q_proj.size(0) == self.num_head and Q_proj.size(1) == num_batch*qlen and Q_proj.size(2) == dq


#         K_proj = K.view(-1, K.size(-1))
#         K_proj = K_proj.unsqueeze(0).expand(self.num_head, *K_proj.size())
#         assert K_proj.size(0) == self.num_head and K_proj.size(1) == num_batch*klen and K_proj.size(2) == dk
#         K_proj = self.proj_K(K_proj)
#         assert K_proj.size(0) == self.num_head and K_proj.size(1) == num_batch*klen and K_proj.size(2) == dk

#         V_proj = V.view(-1, V.size(-1))
#         V_proj = V_proj.unsqueeze(0).expand(self.num_head, *V_proj.size())
#         assert V_proj.size(0) == self.num_head and V_proj.size(1) == num_batch*vlen and V_proj.size(2) == dv
#         V_proj = self.proj_V(V_proj)
#         assert V_proj.size(0) == self.num_head and V_proj.size(1) == num_batch*vlen and V_proj.size(2) == dv


#         Q_proj = Q_proj.view(-1, qlen, Q_proj.size(-1))
#         assert Q_proj.size(0) == self.num_head*num_batch and Q_proj.size(1) == qlen and Q_proj.size(2) == dq
      
#         K_proj = K_proj.view(-1, klen, K_proj.size(-1))
#         assert K_proj.size(0) == self.num_head*num_batch and K_proj.size(1) == klen and K_proj.size(2) == dk

#         V_proj = V_proj.view(-1, vlen, V_proj.size(-1))
#         assert V_proj.size(0) == self.num_head*num_batch and V_proj.size(1) == vlen and V_proj.size(2) == dv

#         # Compute scores and scale
#         scores = Q_proj.bmm(K_proj.transpose(2, 1)) / math.sqrt(self.qk_feature_size)
#         assert scores.size(0) == self.num_head*num_batch and scores.size(1) == qlen and scores.size(2) == klen
#         # Mask padding
#         K_mask = K_mask.unsqueeze(1).expand(K.size(0), qlen, klen).repeat(self.num_head, 1, 1)
#         assert K_mask.size() == scores.size()
        
#         scores.data.masked_fill_(K_mask.data.eq(1), -float('inf'))
#         # Normalize
#         alpha_flat = F.softmax(scores.view(-1, scores.size(-1)), dim=-1)
#         alpha = alpha_flat.view(-1, qlen, klen) 
#         assert alpha.size(0) == self.num_head*num_batch and alpha.size(1) == qlen and alpha.size(2) == klen

#         # Tính attn vô value
#         heads = alpha.bmm(V_proj).view(self.num_head, -1, qlen, dv) # shape(num_head, batch, qlen, dv)
        
#         head_cat = torch.cat([heads[i] for i in range(self.num_head)], dim=-1)
#         out = self.linear_out(head_cat.view(-1, head_cat.size(-1)))
#         return out.view(num_batch, qlen, dv)

# class SelfAttnMultihead(Multihead):
#     def __init__(self, num_head, feature_size, out_size=None):
#         super().__init__(num_head, feature_size, feature_size, out_size if out_size is not None else feature_size)

# class EncodeModule(nn.Module):
#     def __init__(self, hidden_size, value_size, out_size, num_head, dropout_rate=0.1) -> None:
#         super(EncodeModule, self).__init__()
#         self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
#         self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
#         self.norm1 = nn.LayerNorm((hidden_size, ))
#         self.norm2 = nn.LayerNorm((hidden_size, ))
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.mul_head = Multihead(num_head, hidden_size, value_size, out_size)

#     def forward(self, Q, K, V, K_mask):
#         heads = self.mul_head(Q, K, V, K_mask)
#         heads = self.norm1(Q + heads)
#         self.dropout(heads)
#         heads2 = self.linear1(heads.view(-1, heads.size(-1)))
#         heads2 = F.gelu(heads2)
#         heads2 = self.linear2(heads2).view(heads.size())
#         heads2 = F.gelu(heads2)
#         heads2 = self.norm2(heads2 + heads)
#         return self.dropout(heads2)