"""
Day: 11: Multi-Head Attention + Transformer Block
"""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    多头自注意力. 

    输入: (batch, seq_len, embed_dim)
    输出: (batch, seq_len, embed_dim)
    """
    def __init__(self, embed_dim: int, num_heads:int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性投影, 把输入映射到 Q/K/V (合并成一个大的Linear, 效率更高)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) or None
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影 -> (batch, seq, 3 * embed_dim)
        qkv = self.qkv_proj(x)

        # 拆分
        q, k, v = qkv.chunk(3, dim = -1)

        # 分头: 把embed_dim 维度拆成 (num_heads, head_dim)
        # 目标形状: (batch, num_heads, seq_len, head_dim)
        # 这样每个头可以做独立乘法
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在每个 q, k, v 都是 (batch, num_heads, seq_len, head_dim)

        # 4. 计算注意力分散, 并Scale
        #   (batch, heads, seq, head_dim) @ (batch, heads, head_dim,seq) -> (batch, heads, seq, seq)
        scores = q @ k.transpose(-2, -1)
        scores = scores / self.scale

        # 5. Mask 
        if mask is not None:
            # mask: (batch, seq) -> 拓展到 (batch, 1, 1, seq) 以便广播
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 6. Softmax
        attn_weights = torch.softmax(scores, dim = -1)

        # 7. 加权求和
        #   (batch, heads, seq, seq) @ (batch, heads, seq, head_dim) -> (batch, heads, seq, head_dim)
        attn_output = attn_weights @ v
        

        # 8. 拼接多头: 把 heads 维度合并回 embed_dim
        # (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim) -> (batch, seq, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # 9. 输出投影
        output = self.out_proj(attn_output)
        return output, attn_weights


# ================ Transformer Block ======================
class TransformerBlock(nn.Module):
    """
    标准 Transformer Encode Block:
    Attention -> Add&Norm -> FFN -> Add&Norm
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Args: 
            embed_dim: 模型维度(如256)
            numheads: 注意力头数(如8)
            ff_dim: FFN中间层维度 (如 512, 通常是模型维度的 2-4 倍)
            dropout: Dropout 概率
        """
        super().__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads)

        #FFN: 两个 Linear + ReLU
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        关键: 残差连接(residual Connection)
        防止梯度丢失, 让深层网络可训练
        """

        # 子层 1: Multi-Head Attention + 残差 + LayerNorm
        attn_out, _ = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 子层 2: FFN + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

if __name__ == "__main__":
    batch, seq, embed = 2, 12, 256
    x = torch.rand(batch, seq, embed)

    # 测试 MultiHeadAttention
    mha = MultiHeadAttention(embed_dim = embed, num_heads = 8)
    out, weights = mha(x)

    print(f"输入: {x.shape}")           #[2, 12, 256]
    print(f"输出: {out.shape}")         #[2, 12, 256]
    print(f"权重: {weights.shape}")     #[2, 8, 12, 12]

    # 测试 TransformerBlock
    block = TransformerBlock(embed_dim = embed, num_heads = 8,ff_dim = 512)
    out = block(x)
    print(f"Block 输出: {out.shape}")   #[2, 12, 256]