from torch import mode, tensor
import torch
import torch.nn as nn
from IntmdSequential import IntermediateSequential
from torch.utils.tensorboard import summaryWriter

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
        print("transformer output : ",x.size())
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        output = self.fn(x) + x
        print(output.size())
        return output
# 原始convmixer的实现
def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

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

# 原始的SETR中的transformer的MLP部分
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout_rate):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(p=dropout_rate),
#         )

#     def forward(self, x):
#         return self.net(x)

class ConvMixerLayer(nn.Module):
    def __init__(self,dim,hidden_dim,kernel_size = 3):
        super().__init__()
        self.Resnet =  nn.Sequential(
            # nn.Conv2d(dim,dim,kernel_size=kernel_size,groups=dim,padding=1),
            nn.Conv2d(4,dim//4,kernel_size=kernel_size,groups=4,padding=1),
            nn.GELU(),
            nn.BatchNorm2d(dim//4)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(4 + dim//4,4,kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(4)
        )
    def forward(self,x):
        # print("conv_mixer input:",x.size())
        x = x.unsqueeze(dim=0)
        # print("x unsqueeze :",x.size())
        out = self.Resnet(x)
        # print("conv_mixer out:",out.size())
        # x = x + out
        x = torch.cat((x,out),dim=1)
        x = self.Conv_1x1(x)
        x = x.squeeze(dim=0)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = ConvMixerLayer(dim, hidden_dim)

    def forward(self, x):
        return self.net(x)

class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = IntermediateSequential(*layers)

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = ConvMixerLayer(1,1)
    x = torch.randn(1,1,1024,1024)
    output = model(x)
    print(output)
