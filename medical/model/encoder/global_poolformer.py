from typing import Sequence, Union
import ml_collections
import torch
import torch.nn as nn
from medical.model.fusion.layers import get_config
from einops import rearrange
import torch.nn.functional as F

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        super(Convolution, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride_size, padding_size),
                                    nn.InstanceNorm3d(out_channels),
                                    nn.ReLU())
    def forward(self, x):
        x = self.conv_1(x)
        return x

class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        super(TwoConv, self).__init__()
        self.conv_1 = Convolution(in_channels, out_channels, kernel_size, stride_size, padding_size)
        self.conv_2 = Convolution(out_channels, out_channels, kernel_size, stride_size, padding_size)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
            self,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            pool_size = (2, 2, 2)
    ):

        super().__init__()

        up_chns = in_chns // 2
        self.upsample = torch.nn.ConvTranspose3d(in_chns, up_chns, kernel_size=pool_size, stride=pool_size, padding=0)

        self.convs = TwoConv(cat_chns + up_chns, out_chns, 3, 1, 1)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        x_0 = self.upsample(x)

        x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x

###################################
class MlpChannel(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Conv3d(config.hidden_size, config.mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(config.mlp_dim, config.hidden_size, 1)
        self.drop = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
#####################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention_weights = self.sigmoid(out)
        return attention_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        attention_weights = self.sigmoid(x)
        return attention_weights

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        ca_weights = self.ca(x)
        sa_weights = self.sa(x * ca_weights)
        return sa_weights  # sa_weights를 반환

class moe_gating(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=6):
        super(moe_gating, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Conv3d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, kernel_size=1),
        ) for _ in range(num_experts)])

        self.gate = nn.Conv3d(dim, num_experts, kernel_size=1)

    def forward(self, x, attention_weights):
        batch_size, channels, depth, height, width = x.size()
        gate_values = self.gate(x).view(batch_size, self.num_experts, -1)
        
        # attention_weights를 gate_values에 곱하여 반영
        gate_values = gate_values * attention_weights.view(batch_size, 1, -1)
        
        gate_values = torch.softmax(gate_values, dim=1)
        top_k_values, top_k_indices = torch.topk(gate_values, k=3, dim=1)
        mask = torch.zeros_like(gate_values).scatter_(1, top_k_indices, top_k_values)
        gate_values = mask
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs_flat = expert_outputs.view(batch_size, self.num_experts, channels, -1)
        combined_output = torch.sum(gate_values.unsqueeze(2) * expert_outputs_flat, dim=1)
        combined_output = combined_output.view(batch_size, channels, depth, height, width)
        return combined_output

######################################
class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        return x

class Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        in_channels = config.in_channels
        patch_size = config.patch_size

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.norm = LayerNormChannel(num_channels=config.hidden_size)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = self.norm(x)
        return x

class GlobalPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        all_size = self.img_size[0] * self.img_size[1] * self.img_size[2]
        self.global_layer = nn.Linear(1, all_size)

    def forward(self, x):
        x = rearrange(x, "b c d w h -> b c (d w h)")
        x = x.mean(dim=-1, keepdims=True)
        x = self.global_layer(x)
        x = rearrange(x, "b c (d w h) -> b c d w h", d=self.img_size[0], w=self.img_size[1], h=self.img_size[2])
        return x

class BlockPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNormChannel(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNormChannel(config.hidden_size, eps=1e-6)
        self.ffn = moe_gating(dim=config.hidden_size, hidden_dim=config.mlp_dim, num_experts=6)
        self.attn = nn.AvgPool3d(3, 1, padding=1)

    def forward(self, x, attention_weights):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x) + x
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x * attention_weights, attention_weights)  # sa_weights를 이용
        x = x + h

        return x

class Poolformer(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, num_experts, patch_size, mlp_size=256, num_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = get_config(in_channels=in_channels, hidden_size=out_channels, num_experts=num_experts, patch_size=patch_size, img_size=img_size)
        self.block_list = nn.ModuleList([BlockPool(self.config) for i in range(num_layers)])
        self.embeddings = Embeddings(self.config)
        self.cbam = CBAM(out_channels)

    def forward(self, x, out_hidden=False):
        x = self.embeddings(x)
        hidden_state = []
        attention_weights = self.cbam(x)
        for l in self.block_list:
            x = l(x, attention_weights)
            hidden_state.append(x)
        if out_hidden:
            return x, hidden_state
        return x

class PoolformerEncoder(nn.Module):
    def __init__(
            self,
            img_size,
            in_channels,
            features: Sequence[int],
            pool_size,
            num_experts,
            mlp_ratio=4.

    ):
        super().__init__()
        fea = features
        self.drop = nn.Dropout()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.features = features
        self.img_size = img_size
        self.conv_0 = TwoConv(in_channels, features[0], 3, 1, 1)
        self.down_1 = Poolformer(fea[0], fea[1], img_size=img_size[0], patch_size=pool_size[0], mlp_size=fea[1]*2, num_layers=2, num_experts=num_experts)
        self.down_2 = Poolformer(fea[1], fea[2], img_size=img_size[1], patch_size=pool_size[1], mlp_size=fea[2]*2, num_layers=2, num_experts=num_experts)
        self.down_3 = Poolformer(fea[2], fea[3], img_size=img_size[2], patch_size=pool_size[2], mlp_size=fea[3]*2, num_layers=2, num_experts=num_experts)
        self.down_4 = Poolformer(fea[3], fea[4], img_size=img_size[3], patch_size=pool_size[3], mlp_size=fea[4]*2, num_layers=2, num_experts=num_experts)

    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        return x4, x3, x2, x1, x0

class Encoder(nn.Module):

    def __init__(self, model_num,
                 img_size,
                 fea,
                 pool_size,
                 num_experts
                 ):

        super().__init__()
        self.model_num = model_num
        self.encoders = nn.ModuleList([])
        for i in range(model_num):
            encoder = PoolformerEncoder(
                                               img_size=img_size,
                                               in_channels=1,
                                               pool_size=pool_size,
                                               features=fea,
                                               num_experts=num_experts)

            self.encoders.append(encoder)

    def forward(self, x):
        encoder_out = []
        x = x.unsqueeze(dim=2)
        for i in range(self.model_num):
            encoder_out.append(self.encoders[i](x[:, i]))

        return encoder_out

#################### multimodal fusion transformer ###############
class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class FeedForwardMoE(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate, num_experts=6, top_k=3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(p=dropout_rate)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        gate_scores = self.gate(x)
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        batch_size, seq_len, dim = x.size()
        expert_outputs = torch.zeros(batch_size, seq_len, dim, device=x.device)

        for i in range(self.top_k):
            expert_idx = topk_indices[:, :, i]
            for j in range(self.num_experts):
                mask = (expert_idx == j).float().unsqueeze(-1).expand(batch_size, seq_len, dim)
                if mask.sum() > 0:
                    expert_output = self.experts[j](x * mask)
                    expert_outputs += expert_output * mask

        topk_scores = F.softmax(topk_scores, dim=-1)
        # topk_scores를 (batch_size, seq_len, top_k)에서 (batch_size, seq_len, top_k, 1)로 확장
        topk_scores = topk_scores.unsqueeze(-1)
        # expert_outputs를 (batch_size, seq_len, dim)에서 (batch_size, seq_len, 1, dim)로 확장
        expert_outputs = expert_outputs.unsqueeze(2)
        # topk_scores와 expert_outputs를 곱한 후 (batch_size, seq_len, top_k, dim) 형태로 만듦
        final_output = (topk_scores * expert_outputs).sum(dim=2)

        return final_output
               

class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.01):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForwardMoE(embedding_dim, mlp_dim, dropout_rate, num_experts=6, top_k=3))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)

    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x

