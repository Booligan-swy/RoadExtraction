import logging
import math
import os
import numpy as np 

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
# from SETR.transformer_model import TransModel2d, TransConfig
from SETR.transformer_block import TransModel2d, TransConfig
import math 

class Encoder2D(nn.Module):
    def __init__(self, config: TransConfig, is_segmentation=True):
        super().__init__()
        self.config = config
        self.out_channels = config.out_channels
        self.bert_model = TransModel2d(config)
        sample_rate = config.sample_rate
        sample_v = int(math.pow(2, sample_rate))
        assert config.patch_size[0] * config.patch_size[1] * config.hidden_size % (sample_v**2) == 0, "不能除尽"
        self.final_dense = nn.Linear(config.hidden_size, config.patch_size[0] * config.patch_size[1] * config.hidden_size // (sample_v**2))
        self.patch_size = config.patch_size
        self.hh = self.patch_size[0] // sample_v
        self.ww = self.patch_size[1] // sample_v

        self.is_segmentation = is_segmentation
    def forward(self, x):
        ## x:(b, c, w, h)
        b, c, h, w = x.shape
        # print("输入图像",x.shape)
        assert self.config.in_channels == c, "in_channels != 输入图像channel"
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        # print("encode2d p1:",p1)
        # print("encode2d p2:",p2)
        # print("*************************************************")
        # print("hidden_size",self.config.hidden_size)
        # print("",self.config.patch_size[0]," ",self.config.patch_size[1])
        # print(self.config.sample_rate)
        # print("sample_v:",int(math.pow(2, self.config.sample_rate)))

        # print("*************************************************")
        if h % p1 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        if w % p2 != 0:
            print("请重新输入img size 参数 必须整除")
            os._exit(0)
        hh = h // p1 
        ww = w // p2 
        # print("hh:",hh," ww:",ww)

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p1, p2 = p2)
        
        out_layer_1, out_layer_2, out_layer_3, encode_x = self.bert_model(x) # 取出来编码器四层的输出
        # print(encode_x.size())
        # if not self.is_segmentation:
        #     return encode_x
        out_layer_1 = self.final_dense(out_layer_1)
        out_layer_2 = self.final_dense(out_layer_2)
        out_layer_3 = self.final_dense(out_layer_3)

        x = self.final_dense(encode_x)
        # print("final_dense:",x.size())
        # print("self.hh:",self.hh)
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = self.hh, p2 = self.ww, h = hh, w = ww, c = self.config.hidden_size)
        # print("encode output x size is :",x.size())
        out_layer_1 = rearrange(out_layer_1, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = self.hh, p2 = self.ww, h = hh, w = ww, c = self.config.hidden_size)
        out_layer_2 = rearrange(out_layer_2, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = self.hh, p2 = self.ww, h = hh, w = ww, c = self.config.hidden_size)
        out_layer_3 = rearrange(out_layer_3, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1 = self.hh, p2 = self.ww, h = hh, w = ww, c = self.config.hidden_size)

        return out_layer_1, out_layer_2, out_layer_3, x 

class InterBlock(nn.Module):
    def __init__(self, in_num_channels=[1024,1024,1024,1024], out_num_channels=[128,256,512,1024], enlarge_factor=[8,4,2]):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_num_channels[0], out_num_channels[0], 3, padding=1),
            nn.BatchNorm2d(out_num_channels[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=enlarge_factor[0], mode="bilinear", align_corners=True)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_num_channels[1], out_num_channels[1], 3, padding=1),
            nn.BatchNorm2d(out_num_channels[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=enlarge_factor[1], mode="bilinear", align_corners=True)
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_num_channels[2], out_num_channels[2], 3, padding=1),
            nn.BatchNorm2d(out_num_channels[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=enlarge_factor[2], mode="bilinear", align_corners=True)
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_num_channels[3], out_num_channels[3], 1),
            nn.BatchNorm2d(out_num_channels[3]),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
    def forward(self, x_1, x_2, x_3, x_4):
        x_1 = self.block_1(x_1)
        x_2 = self.block_2(x_2)
        x_3 = self.block_3(x_3)
        x_4 = self.block_4(x_4)

        return x_1, x_2, x_3, x_4


class Decoder2D(nn.Module):
    def __init__(self, in_channels=[1024,512,256,128], out_channels=1, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels[0], features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(in_channels[1], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(in_channels[2], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(in_channels[3], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x_1, x_2, x_3,x):
        d1 = self.decoder_1(x)
        x_3 = x_3 + d1
        # print("decode 1 x:",d1.size())
        d2 = self.decoder_2(x_3)
        x_2 = x_2 + d2
        # print("decode 2 x:",d2.size())
        d3 = self.decoder_3(x_2)
        x_1 = x_1 + d3
        # print("decode 3 x:",d3.size())
        d4 = self.decoder_4(x_1)
        # print("decode 4 x:",d4.size())
        output = self.final_out(d4)
        return output

class DeeplabTRNet(nn.Module):
    def __init__(self, patch_size=(32, 32), 
                        in_channels=3, 
                        out_channels=1, 
                        hidden_size=1024, 
                        num_hidden_layers=8, 
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 64],
                        sample_rate=4,):
        super().__init__()
        config = TransConfig(patch_size=patch_size, 
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            sample_rate=sample_rate,
                            hidden_size=hidden_size, 
                            num_hidden_layers=num_hidden_layers, 
                            num_attention_heads=num_attention_heads)
        self.encoder_2d = Encoder2D(config)
        self.interblock = InterBlock()
        self.decoder_2d = Decoder2D()

    def forward(self, x):
        x_1, x_2, x_3, final_x = self.encoder_2d(x)
        x_1, x_2, x_3, x_4 = self.interblock(x_1, x_2, x_3, final_x)
        
        x = self.decoder_2d(x_1, x_2, x_3, x_4)
        return x 

   

