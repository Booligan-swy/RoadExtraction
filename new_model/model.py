from scipy.spatial import transform
import torch
from torch import diag, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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
# 原始vit版本的FeedForward
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)

############################################
# FeedForward实现
###########################################
class TransLayerNorm(nn.Module):
    def __init__(self, hidden_size = 1024, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(TransLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
       

    def forward(self, x):

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class TransIntermediate(nn.Module):
    def __init__(self, hidden_size = 1024, intermediate_size = 1024):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # self.intermediate_act_fn = ACT2FN[config.hidden_act] ## relu 
        # self.intermediate_act_fn = F.

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class TransOutput(nn.Module):
    def __init__(self, intermediate_size = 1024, hidden_size = 1024, hidden_dropout_prob = 0.1,layer_norm_eps = 1e-12):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = TransLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# 替换FeedForward的实现过程（测试）
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.intermediate = TransIntermediate(dim, hidden_dim)
        self.output = TransOutput(hidden_dim, dim)
        
    def forward(self, x):
        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)
        return layer_output
############################################

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout):
#         super().__init__()
#         self.net = ConvMixerLayer(dim, hidden_dim)

#     def forward(self, x):
#         return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print("x_atten size:",x.size())
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # print("x_atten q:",qkv[0].size())
        # print("x_atten k:",qkv[1].size())
        # print("x_atten v:",qkv[2].size())
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # print("q ",q.size())
        # print("k ",k.size())
        # print("v ",v.size())
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print("dots ",dots.size())
        attn = self.attend(dots)
        # print("attn ",attn.size())

        out = torch.matmul(attn, v)
        # print("out 1 ",out.size())
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print("out 2 ",out.size())
        out = self.to_out(out)
        # print("attention out : ",out.size())
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            # print("transformer inter:", x.size())
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # print("img.size:",img.size())
        x = self.to_patch_embedding(img)
        # print("x self.to_patch_embedding(img):",x.size())
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # print("cls_tokens : ", cls_tokens.size())
        # x = torch.cat((cls_tokens, x), dim=1)
        # print("x cls_tokens cat size:",x.size())
        x += self.pos_embedding[:, :(n + 1)]
        # print("x pos_embedding :",x.size())
        x = self.dropout(x)
        # print("x dropout:",x.size())
        x = self.transformer(x)
        # print("x2 transformer :",x.size())
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ASP(nn.Module):
    def __init__(self,in_channel, depth):
        super(ASP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.atrous_block1 = nn.Conv2d(in_channel,depth, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn = nn.BatchNorm2d(depth)
        self.atrous_block2 = nn.Conv2d(in_channel,depth,kernel_size=3,stride=1,padding=4, dilation=4)
        self.atrous_block3 = nn.Conv2d(in_channel,depth,kernel_size=3, stride=1, padding=6, dilation=6)
    def forward(self, x):
        # x = self.mean(x)
        print("ASP x size ",x.size())
        x1 = self.atrous_block1(x)
        x1 = self.bn(x1)
        print("ASP x1 size ",x1.size())
        x2 = self.atrous_block2(x)
        x2 = self.bn(x2)
        print("ASP x2 size ",x2.size())
        x3 = self.atrous_block3(x)
        x3 = self.bn(x3)
        print("ASP x3 size ",x3.size())
        return x1+x2+x3

def conv1x1_stage(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels)
    )

def inter_conv_stage(in_channels=1, out_channels=256):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels)
    )

class Encoder2D(nn.Module):
    def __init__(self,image_size,
                patch_size, 
                dim, 
                depth, 
                heads, 
                mlp_dim, 
                channels = 3,  
                dim_head = 64, 
                dropout = 0., 
                emb_dropout = 0.,
                sample_rate = 5):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)
        self.dim = dim
        assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0, "Image dimensions must be divisible by the patch size."

        num_patches = int((image_height // self.patch_height)**2)
        patch_dim = channels * self.patch_height * self.patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.sample_rate = sample_rate
        sample_v = int(math.pow(2, self.sample_rate))
        assert self.patch_height * self.patch_width * dim % (sample_v**2) == 0, "Can't be divisible."
        self.final_dense = nn.Linear(dim, self.patch_height * self.patch_width * dim // (sample_v**2))

        self.layer_1 = Transformer(dim = dim, depth = depth, heads = heads, dim_head=dim_head,mlp_dim=mlp_dim, dropout=dropout)
        # self.layer_2 = Transformer(dim, depth = 4, heads = 8, dim_head=64,mlp_dim=2, dropout=0.2)
        # self.layer_3 = Transformer(dim, depth = 6, heads = 8, dim_head=64,mlp_dim=2, dropout=0.1)
        # self.layer_4 = Transformer(dim, depth = 3, heads = 8, dim_head=64,mlp_dim=2, dropout=0.2)

    def forward(self, img):
        # print(img.size())
        x = self.to_patch_embedding(img)
        # print("to_patch_embedding: ",x.size())
        b,n,_ = x.shape
        # print("b:",b," n:",n)
        x += self.pos_embedding[:,:n]
        # print("pos_embedding: ",x.size())
        x = self.dropout(x)
        encode_x = self.layer_1(x)
        # print("trans out 1 layer_1: ",encode_x.size())

        x = self.final_dense(encode_x)
        # print("final dense: ", x.size())
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",h = self.patch_height//2, w = self.patch_width//2, p1 = 1, p2 = 1, c = self.dim)
        # output = F.interpolate(, size=img.size()[2:], mode='bilinear', align_corners=True)
        # output = F.sigmoid(output)
        return encode_x,x


class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        # print("decode 1 x:",x.size())
        x = self.decoder_2(x)
        # print("decode 2 x:",x.size())
        x = self.decoder_3(x)
        # print("decode 3 x:",x.size())
        x = self.decoder_4(x)
        # print("decode 4 x:",x.size())
        x = self.final_out(x)
        x = F.sigmoid(x)
        return x

class DeeplabTRNet(nn.Module):
    def __init__(self,image_size,
                patch_size, 
                dim, 
                depth, 
                heads, 
                mlp_dim, 
                channels = 3,  
                dim_head = 64, 
                dropout = 0., 
                emb_dropout = 0.,
                sample_rate = 5):
        super().__init__()
        self.decode_features = [512, 256, 128, 64]

        self.encode_2d = Encoder2D(image_size,
                patch_size, 
                dim, 
                depth, 
                heads, 
                mlp_dim, 
                channels = 3,  
                dim_head = 64, 
                dropout = 0., 
                emb_dropout = 0.,
                sample_rate = 5)
        self.decode_2d = Decoder2D(in_channels=dim, out_channels=1, features=self.decode_features)
    def forward(self,x):
        _, final_x = self.encode_2d(x)
        x = self.decode_2d(final_x)
        return x

if __name__ == "__main__":
    x = torch.randn((4,3,512,512))
    model = DeeplabTRNet(image_size = 512, patch_size=32, dim=1024, depth=24, heads=8, mlp_dim=2, channels = 3, dim_head = 64, dropout = 0.2, emb_dropout = 0.2)
    output = model(x)
    # print(output)
    print(output.size())
# if __name__ == "__main__":
#     x = torch.randn((4,3,512,512))
#     model = Transformer(dim=512, depth=24, heads=12, dim_head=64, mlp_dim=2, dropout = 0.2)
#     output = model(x)
#     print(output)
#     print(output.size())