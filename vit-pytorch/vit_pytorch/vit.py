from scipy.spatial import transform
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

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
        print("x_atten size:",x.size())
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        print("x_atten q:",qkv[0].size())
        print("x_atten k:",qkv[1].size())
        print("x_atten v:",qkv[2].size())
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        print("q ",q.size())
        print("k ",k.size())
        print("v ",v.size())
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        print("dots ",dots.size())
        attn = self.attend(dots)
        print("attn ",attn.size())

        out = torch.matmul(attn, v)
        print("out 1 ",out.size())
        out = rearrange(out, 'b h n d -> b n (h d)')
        print("out 2 ",out.size())
        out = self.to_out(out)
        print("attention out : ",out.size())
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
            print("transformer inter:", x.size())
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
        print("img.size:",img.size())
        x = self.to_patch_embedding(img)
        print("x self.to_patch_embedding(img):",x.size())
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # print("cls_tokens : ", cls_tokens.size())
        # x = torch.cat((cls_tokens, x), dim=1)
        # print("x cls_tokens cat size:",x.size())
        x += self.pos_embedding[:, :(n + 1)]
        print("x pos_embedding :",x.size())
        x = self.dropout(x)
        print("x dropout:",x.size())
        x = self.transformer(x)
        print("x2 transformer :",x.size())
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == "__main__":
    x = torch.randn((4,3,512,512))
    model = ViT(image_size = 512, patch_size=32, num_classes=2, dim=256, depth=2, heads=2, mlp_dim=2, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.2, emb_dropout = 0.2)
    output = model(x)
    print(output)
    print(output.size())
# if __name__ == "__main__":
#     x = torch.randn((4,3,512,512))
#     model = Transformer(dim=512, depth=24, heads=12, dim_head=64, mlp_dim=2, dropout = 0.2)
#     output = model(x)
#     print(output)
#     print(output.size())