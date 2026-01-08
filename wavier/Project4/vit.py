import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        
        """
        Attention类中参数的含义：
        - dim 输入特征的维度
        - heads 多头注意力模块中头的个数
        - dim_head 每个头的维度
        """ 

        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        head =self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, head, -1).transpose(1, 2), qkv)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, -1)

        return self.to_out(out)
        """
        TODO
        实现多头注意力机制的前向传播过程，大约10行代码。
        """
        pass

class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, depth, dim_feedforward):
        """
        Transformer类中参数的含义：
        - dim 输入特征的维度
        - heads 多头注意力模块中头的个数
        - dim_head 每个头的维度
        - depth Transformer的层数
        - dim_feedforward FeedForward模块中隐藏层的维度
        """ 
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, dim_feedforward),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]))
                  
    def forward(self, x):
        for attn, ff, norm1, norm2 in self.layers:
            x_attn = attn(norm1(x))
            x = x + x_attn
            x = x + ff(norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, num_classes=10, dim=64, depth=2, heads=2, dim_head=64, mlp_dim=128):

        """
        ViT类中参数的含义：
        - img_size 输入图像的尺寸
        - patch_size 小块的尺寸
        - num_classes 全部类的个数
        - dim 输入特征的维度
        - depth Transformer的层数
        - heads 多头注意力模块中头的个数
        - dim_head 每个头的维度
        - mlp_dim FeedForward模块中隐藏层的维度
        """ 

        super(ViT, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by the patch size."
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        
        # 可学习位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 0, dim))
        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
        self.transformer = Transformer(dim, heads, dim_head, depth, dim_feedforward=mlp_dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(x.size(0), -1, self.patch_dim)
        x = self.patch_to_embedding(x)
        
        b, n, _ = x.shape
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.transformer(x)
        x = torch.mean(x,dim=1)

        return self.sigmoid(self.mlp_head(x))