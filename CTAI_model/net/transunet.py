"""
TransUNet: Transformer + UNet 融合架构
基于论文: TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation (2021)

架构设计:
- Encoder: Vision Transformer (ViT) 提取全局特征
- Decoder: CNN Decoder (UNet风格) 还原分辨率
- Skip Connection: 融合多尺度特征 + Attention Gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    """将图像分割为patches并嵌入到高维空间"""
    def __init__(self, img_size=512, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 32x32 = 1024
        
        # 使用卷积实现patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, 1, 512, 512)
        x = self.proj(x)  # (B, 768, 32, 32)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, 1024, 768)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 残差连接 + 注意力
        x = x + self.attn(self.norm1(x))
        # 残差连接 + MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器 (堆叠多个TransformerBlock)"""
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class AttentionGate(nn.Module):
    """注意力门控机制 - 优化跳跃连接"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """解码器块: 上采样 + 卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class HybridCNNEncoder(nn.Module):
    """混合CNN编码器 - 提取低层特征用于跳跃连接"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 256x256
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # 128x128
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 不再池化,保持128x128用于与Transformer特征融合
        
    def forward(self, x):
        # 返回多尺度特征供跳跃连接使用
        c1 = self.conv1(x)      # (B, 64, 512, 512)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)     # (B, 128, 256, 256)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)     # (B, 256, 128, 128)
        
        return c1, c2, c3


class TransUNet(nn.Module):
    """
    TransUNet主模型
    
    参数说明:
    - img_size: 输入图像尺寸 (默认512)
    - patch_size: Transformer patch大小 (默认16)
    - in_channels: 输入通道数 (CT图像为1)
    - out_channels: 输出通道数 (二分类为1)
    - embed_dim: Transformer嵌入维度 (默认768, ViT-Base标准)
    - depth: Transformer层数 (默认12, 可减少到6以节省显存)
    - num_heads: 注意力头数 (默认12)
    """
    def __init__(
        self, 
        img_size=512, 
        patch_size=16, 
        in_channels=1, 
        out_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        # 1. CNN编码器 (提取低层特征)
        self.cnn_encoder = HybridCNNEncoder()
        
        # 2. Patch Embedding (将图像转为序列)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # 3. 位置编码 (可学习)
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 4. Transformer编码器
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
        
        # 5. 桥接层: 将Transformer输出(1024, 768)重塑为特征图(768, 32, 32)
        self.bridge = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 6. 解码器
        self.decoder4 = DecoderBlock(512, 256)   # 32x32 -> 64x64
        self.att4 = AttentionGate(256, 256, 128)
        
        self.decoder3 = DecoderBlock(512, 128)   # 64x64 -> 128x128 (256+256=512 -> 128)
        self.att3 = AttentionGate(128, 256, 128)  # g=128, x=256
        
        self.decoder2 = DecoderBlock(384, 64)    # 128x128 -> 256x256 (128+256=384 -> 64)
        self.att2 = AttentionGate(64, 128, 64)    # g=64, x=128
        
        self.decoder1 = DecoderBlock(192, 64)    # 256x256 -> 512x512 (64+128=192 -> 64)
        
        # 7. 输出层
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B = x.shape[0]
        
        # CNN编码器提取多尺度特征
        c1, c2, c3 = self.cnn_encoder(x)  # (64, 512, 512), (128, 256, 256), (256, 128, 128)
        
        # Transformer编码器
        x_embed = self.patch_embed(x)     # (B, 1024, 768)
        x_embed = x_embed + self.pos_embed
        x_trans = self.transformer(x_embed)  # (B, 1024, 768)
        
        # 重塑为特征图
        x_trans = rearrange(x_trans, 'b (h w) c -> b c h w', h=32, w=32)  # (B, 768, 32, 32)
        x_trans = self.bridge(x_trans)  # (B, 512, 32, 32)
        
        # 解码器 + 注意力门控跳跃连接
        d4 = self.decoder4(x_trans)  # (B, 256, 64, 64)
        
        # 第一个跳跃: c3是(256, 128, 128),需要下采样到64x64并应用注意力
        c3_down = F.interpolate(c3, size=(64, 64), mode='bilinear', align_corners=False)  # (256, 64, 64)
        c3_att = self.att4(g=d4, x=c3_down)  # (256, 64, 64)
        d3 = self.decoder3(torch.cat([d4, c3_att], dim=1))  # cat[(256, 64, 64), (256, 64, 64)] -> (512, 64, 64) -> (128, 128, 128)
        
        # 第二个跳跃: c3是(256, 128, 128)
        c3_att = self.att3(g=d3, x=c3)  # (256, 128, 128)
        d2 = self.decoder2(torch.cat([d3, c3_att], dim=1))  # cat[(128, 128, 128), (256, 128, 128)] -> (384, 128, 128) -> (64, 256, 256)
        
        # 第三个跳跃: c2是(128, 256, 256)
        c2_att = self.att2(g=d2, x=c2)  # (128, 256, 256)
        d1 = self.decoder1(torch.cat([d2, c2_att], dim=1))  # cat[(64, 256, 256), (128, 256, 256)] -> (192, 256, 256) -> (64, 512, 512)
        
        # 输出
        out = self.final(d1)  # (B, 1, 512, 512)
        
        return out


# ==================== 轻量级版本 (显存不足时使用) ====================

class TransUNetLite(nn.Module):
    """
    TransUNet轻量级版本
    - Transformer层数: 12 -> 6
    - 嵌入维度: 768 -> 384
    - 注意力头数: 12 -> 6
    - 参数量降低约60%
    """
    def __init__(self, img_size=512, patch_size=16, in_channels=1, out_channels=1):
        super().__init__()
        
        embed_dim = 384
        depth = 6
        num_heads = 6
        
        self.cnn_encoder = HybridCNNEncoder()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads)
        
        self.bridge = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder4 = DecoderBlock(256, 128)  # 32x32 -> 64x64
        self.att4 = AttentionGate(128, 256, 64)
        
        self.decoder3 = DecoderBlock(384, 64)  # (128+256=384) -> 64, 64x64 -> 128x128
        self.att3 = AttentionGate(64, 256, 64)
        
        self.decoder2 = DecoderBlock(320, 64)  # (64+256=320) -> 64, 128x128 -> 256x256
        self.att2 = AttentionGate(64, 128, 64)
        
        self.decoder1 = DecoderBlock(192, 64)  # (64+128=192) -> 64, 256x256 -> 512x512
        
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B = x.shape[0]
        
        c1, c2, c3 = self.cnn_encoder(x)  # c1:(64,512,512), c2:(128,256,256), c3:(256,128,128)
        
        x_embed = self.patch_embed(x)
        x_embed = x_embed + self.pos_embed
        x_trans = self.transformer(x_embed)
        
        x_trans = rearrange(x_trans, 'b (h w) c -> b c h w', h=32, w=32)  # (B, 384, 32, 32)
        x_trans = self.bridge(x_trans)  # (B, 256, 32, 32)
        
        d4 = self.decoder4(x_trans)  # (B, 128, 64, 64)
        c3_down = F.interpolate(c3, size=(64, 64), mode='bilinear', align_corners=False)  # (B, 256, 64, 64)
        c3_att = self.att4(g=d4, x=c3_down)  # (B, 256, 64, 64)
        
        d3 = self.decoder3(torch.cat([d4, c3_att], dim=1))  # cat[(128, 64, 64), (256, 64, 64)] -> (384, 64, 64) -> (64, 128, 128)
        c3_att = self.att3(g=d3, x=c3)  # g:(64, 128, 128), x:(256, 128, 128) -> (256, 128, 128)
        
        d2 = self.decoder2(torch.cat([d3, c3_att], dim=1))  # cat[(64, 128, 128), (256, 128, 128)] -> (320, 128, 128) -> (64, 256, 256)
        c2_att = self.att2(g=d2, x=c2)  # g:(64, 256, 256), x:(128, 256, 256) -> (128, 256, 256)
        
        d1 = self.decoder1(torch.cat([d2, c2_att], dim=1))  # cat[(64, 256, 256), (128, 256, 256)] -> (192, 256, 256) -> (64, 512, 512)
        
        out = self.final(d1)  # (B, 1, 512, 512)
        
        return out


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试完整版
    model = TransUNet(
        img_size=512,
        patch_size=16,
        in_channels=1,
        out_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=12
    ).to(device)
    
    x = torch.randn(1, 1, 512, 512).to(device)
    y = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试轻量版
    model_lite = TransUNetLite().to(device)
    y_lite = model_lite(x)
    print(f"\n轻量版输出形状: {y_lite.shape}")
    print(f"轻量版参数量: {sum(p.numel() for p in model_lite.parameters()) / 1e6:.2f}M")
