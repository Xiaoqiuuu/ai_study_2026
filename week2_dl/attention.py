"""
Day10: Scaled Dot-Product Attention
"""

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    单头 Self-Attention。

    输入 x: (batch_size, seq_len, embed_dim)
    输出:   (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int, qkv_dim: int):
        """
        Args:
            embed_dim: 输入词向量的维度
            qkv_dim: Q/K/V 的维度, 通常为 embed_dim 的 1/2 或 1/4
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim

        #三个线性投影: 共享输入，分别生成 Q/K/V
        #注意: 没有bias(在Transformer 原始论文中不用bias)

        self.W_q = nn.Linear(embed_dim, qkv_dim, bias = False)
        self.W_k = nn.Linear(embed_dim, qkv_dim, bias = False)
        self.W_v = nn.Linear(embed_dim, qkv_dim, bias = False)

        # 最后的输出投影, 把qkv_dim 映射回 embed_dim
        self.W_o = nn.Linear(qkv_dim, embed_dim, bias = False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len, embed_dim) 或 None. 用于填充屏蔽位置
            
        Returns:
            output: (batch, seq_len, embed_dim)
            attn_weights: (batch, seq_len, seq_len) 用于可视化
        
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影得到Q, K, V
        # x @ W_q: (batch, seq, embed) @ (embed, qkv) -> (batch, seq, qkv)

        Q = x @ self.W_q.weight.t()
        K = x @ self.W_k.weight.t()
        V = x @ self.W_v.weight.t()

        # 2. 计算注意力分数: Q @ K^T
        # (batch, seq, qkv) @ (batch, qkv, seq) -> (batch, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 3. Scale
        scores = scores / math.sqrt(self.qkv_dim)

        # 4. Mask: 把填充位置的分数设置为 -inf, softmax 后变为 0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. Softmax: 每行归一化, 得到注意力权重
        # dim = -1 表示对最后一位(seq_len) 做 softmax
        attn_weights = torch.softmax(scores, dim = -1)

        # 6. 加权求和: 权重 @ V
        # (batch, seq, seq) @ (batch, seq, qkv) -> (batch, seq, qkv)
        attn_output = attn_weights @ V

        # 7. 输出投影
        output = self.W_o(attn_output)

        return output, attn_weights


# ============== 验证维度 ==============
if __name__ == "__main__":
    batch_size, seq_len, embed_dim, qkv_dim = 2, 4, 8, 4

    x = torch.rand(batch_size, seq_len, embed_dim)
    attn = SelfAttention(embed_dim, qkv_dim)

    output, weights = attn(x)

    print(f"输入 X 形状:     {x.shape}")           # [2, 4, 8]
    print(f"输出形状:        {output.shape}")       # 应为 [2, 4, 8]
    print(f"注意力权重形状:  {weights.shape}")      # 应为 [2, 4, 4]
    print(f"权重每行和:      {weights[0].sum(dim=-1)}")  # 应为 [1, 1, 1, 1]

    # 验证：自注意力后，每个位置的输出都融合了全句信息
    # 但输出形状和输入一样，这就是"置换等变性"