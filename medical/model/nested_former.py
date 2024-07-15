import torch.nn as nn
from .encoder.global_poolformer import Encoder, Convolution, TwoConv, UpCat, Transformer

from .fusion.nmafa import NMaFaLayer
import torch
from einops import rearrange

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
patch_size = 8
depth = 1

class NestedFormer(nn.Module):
    def __init__(self, model_num,
                 out_channels,
                 image_size,
                 fea=(16, 16, 32, 64, 128, 16),
                 num_experts=4,
                 top_k=3,
                 window_size=(2, 4, 4),
                 pool_size=((2,2,2), (2,2,2), (2,2,2), (2,2,2)),
                 self_num_layer=2,
                 token_mixer_size=32,
                 token_learner=True):

        super().__init__()
        self.out_channels = out_channels
        self.model_num = model_num
        self.num_experts = num_experts
        self.pool_size = pool_size

        pool_size_all = [1, 1, 1]
        image_size_s = [image_size]
        for p in pool_size:
            pool_size_all = [pool_size_all[i] * p[i] for i in range(len(p))]
            image_size_s.append((image_size_s[-1][0] // p[0], image_size_s[-1][1] // p[1], image_size_s[-1][2] // p[2]))
        new_image_size = [image_size[i] // pool_size_all[i] for i in range(3)]

        self.encoder = Encoder(model_num=model_num,
                               img_size=image_size_s[1:],
                               fea=fea, pool_size=pool_size, num_experts=num_experts)
        
        self.enconv = nn.Conv3d(fea[4] * model_num, transformer_basic_dims, kernel_size=1, stride=1, padding=0)  # 수정된 부분
        self.pos = nn.Parameter(torch.zeros(1, 512, 512))  # encoder_5 크기에 맞게 조정
        self.visiontransformer = Transformer(embedding_dim=transformer_basic_dims,depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.fusion_conv_5 = TwoConv(model_num * fea[4], fea[4], 3, 1, 1)
        self.fusion_conv_1 = TwoConv(model_num * fea[0], fea[0], 3, 1, 1)
        self.fusion_conv_2 = TwoConv(model_num * fea[1], fea[1], 3, 1, 1)
        self.fusion_conv_3 = TwoConv(model_num * fea[2], fea[2], 3, 1, 1)
        self.fusion_conv_4 = TwoConv(model_num * fea[3], fea[3], 3, 1, 1)
        self.fusion_conv_6 = TwoConv(fea[4], fea[3], 1, 1, 1)

        self.upcat_4 = UpCat(fea[4], fea[3], fea[3], pool_size=pool_size[3])
        self.upcat_3 = UpCat(fea[3], fea[2], fea[2], pool_size=pool_size[2])
        self.upcat_2 = UpCat(fea[2], fea[1], fea[1], pool_size=pool_size[1])
        self.upcat_1 = UpCat(fea[1], fea[0], fea[5], pool_size=pool_size[0])

        self.final_conv = nn.Conv3d(fea[5], out_channels, 1, 1)

    def forward(self, x):
        assert x.shape[1] == self.model_num
        encoder_x = self.encoder(x)

        encoder_1 = torch.stack([encoder_x[i][4] for i in range(self.model_num)], dim=1)  # level 4
        encoder_2 = torch.stack([encoder_x[i][3] for i in range(self.model_num)], dim=1)
        encoder_3 = torch.stack([encoder_x[i][2] for i in range(self.model_num)], dim=1)
        encoder_4 = torch.stack([encoder_x[i][1] for i in range(self.model_num)], dim=1)
        encoder_5 = torch.stack([encoder_x[i][0] for i in range(self.model_num)], dim=1)

        encoder_5 = rearrange(encoder_5, "b n c d w h -> b (n c) d w h")
        fusion_out= self.enconv(encoder_5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        fusion_out = self.visiontransformer(fusion_out, self.pos)
        fusion_out = self.fusion_conv_5(encoder_5)


        
        encoder_1 = rearrange(encoder_1 , "b n c d w h -> b (n c) d w h")
        encoder_2 = rearrange(encoder_2 , "b n c d w h -> b (n c) d w h")
        encoder_3 = rearrange(encoder_3 , "b n c d w h -> b (n c) d w h")
        encoder_4 = rearrange(encoder_4 , "b n c d w h -> b (n c) d w h")

        encoder_1_cnn = self.fusion_conv_1(encoder_1)
        encoder_2_cnn = self.fusion_conv_2(encoder_2)
        encoder_3_cnn = self.fusion_conv_3(encoder_3)
        encoder_4_cnn = self.fusion_conv_4(encoder_4)



        u4 = self.upcat_4(fusion_out, encoder_4_cnn)
        u3 = self.upcat_3(u4, encoder_3_cnn)
        u2 = self.upcat_2(u3, encoder_2_cnn)
        u1 = self.upcat_1(u2, encoder_1_cnn)

       

        logits = self.final_conv(u1)
        return logits
         
