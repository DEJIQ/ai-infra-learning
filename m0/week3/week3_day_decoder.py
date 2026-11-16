import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 切换题目，只把一个置为 True ==========
RUN_T1 = True
RUN_T2 = False
RUN_T3 = False
RUN_T4 = False


if RUN_T1:
    """
    T1：实现缩放点积注意力（Scaled Dot-Product Attention）
    需求：
    1. 实现函数 scaled_dot_product_attention(q, k, v, mask=None)
    2. mask 如果不为 None，要把 mask == 0 的位置填 -inf，再 softmax
    3. 返回注意力输出、注意力权重
    测试：
        batch = 1
        seq_len = 4
        d_model = 8
        q = torch.randn(batch, seq_len, d_model)
        k = torch.randn(batch, seq_len, d_model)
        v = torch.randn(batch, seq_len, d_model)
        out, attn_weight = scaled_dot_product_attention(q, k, v)
        print(out.shape, attn_weight.shape)
    """
    def scaled_dot_product_attention(q, k, v, mask=None):
        # ========== 在这里写你的代码 ==========

        # 提示：
        # d_k = q.size(-1)
        # attn_scores = q @ k.mT / sqrt(d_k)
        # mask 处理、softmax、乘 v
        output = None
        attn_weight = None
        attn_scores = q @ k.transpose(-2,-1) / (q.size(-1) ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weight = F.softmax(attn_scores, dim=-1)
        output = attn_weight @ v
        return output, attn_weight

    def main():
        batch, seq_len, d_model = 1, 4, 8
        q = torch.randn(batch, seq_len, d_model)
        k = torch.randn(batch, seq_len, d_model)
        v = torch.randn(batch, seq_len, d_model)
        out, attn_weight = scaled_dot_product_attention(q, k, v)
        print("T1 out shape:", out.shape)
        print("T1 attn weight shape:", attn_weight.shape)

    if __name__ == "__main__":
        main()


if RUN_T2:
    """
    T2：多头注意力（Multi-Head Attention）
    需求：
    1. 手写 MultiHeadAttention，不要直接调用 nn.MultiheadAttention
    2. 内部切头、分头算注意力、拼接、最后投影
    3. 构造参数：d_model, num_heads；保证 d_model 可以被 num_heads 整除
    测试：
        mha = MultiHeadAttention(d_model=16, num_heads=2)
        x = torch.randn(1, 5, 16)
        out = mha(x, x, x)
        print(out.shape)
    """
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, num_heads: int):
            super().__init__()
            assert d_model % num_heads == 0
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            # ========== 在这里定义参数：w_q, w_k, w_v, w_o ==========
            

        def split_heads(self, x):
            # ========== 在这里实现切头：
            # view + permute，输出 shape: [B, num_heads, S, d_k]
            return x

        def forward(self, q, k, v, mask=None):
            b = q.size(0)
            # 1. 投影、分头
            # 2. 缩放点积注意力
            # 3. 拼接头、投影输出
            out = None
            return out

    def main():
        mha = MultiHeadAttention(d_model=16, num_heads=2)
        x = torch.randn(1, 5, 16)
        out = mha(x, x, x)
        print("T2 out shape:", out.shape)

    if __name__ == "__main__":
        main()


if RUN_T3:
    """
    T3：Decoder 前馈网络（Feed Forward Network）
    需求：
    1. 实现 Pre-LN 的前馈块：LN → Linear → GELU → Linear
    2. 内部带残差，输入加上本模块输出
    测试：
        ffn = FeedForward(d_model=16, hidden_dim=32)
        x = torch.randn(1, 5, 16)
        out = ffn(x)
        print(out.shape)
    """
    class FeedForward(nn.Module):
        def __init__(self, d_model: int, hidden_dim: int):
            super().__init__()
            # ========== 在这里定义 LayerNorm、两个 Linear ==========

        def forward(self, x):
            residual = x
            # ========== 在这里完成前向计算：LN -> fc1 -> gelu -> fc2 + 残差 ==========
            out = None
            return out

    def main():
        ffn = FeedForward(d_model=16, hidden_dim=32)
        x = torch.randn(1, 5, 16)
        out = ffn(x)
        print("T3 out shape:", out.shape)

    if __name__ == "__main__":
        main()


if RUN_T4:
    """
    T4：完整 Pre-LN Decoder Block
    需求：
    1. 顺序：
       - Pre-LN → Masked MultiHeadAttention + 残差
       - Pre-LN → FeedForward + 残差
    2. 内部复用上面的 MultiHeadAttention、FeedForward（你自己实现）
    3. 支持传入因果掩码（causal mask）
    测试：
        block = DecoderBlock(d_model=16, num_heads=2, ffn_hidden=32)
        x = torch.randn(1, 6, 16)
        out = block(x)
        print(out.shape)
    """
    # 你需要把 T2、T3 写好的 MultiHeadAttention / FeedForward 挪进来或者直接实现
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, num_heads: int):
            super().__init__()
            assert d_model % num_heads == 0
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            # ========== 同 T2，自己补 ==========

        def split_heads(self, x):
            # ========== 同 T2，自己补 ==========
            return x

        def forward(self, q, k, v, mask=None):
            # ========== 同 T2，自己补 ==========
            return None

    class FeedForward(nn.Module):
        def __init__(self, d_model: int, hidden_dim: int):
            super().__init__()
            # ========== 同 T3，自己补 ==========

        def forward(self, x):
            # ========== 同 T3，自己补 ==========
            return None

    class DecoderBlock(nn.Module):
        def __init__(self, d_model: int, num_heads: int, ffn_hidden: int):
            super().__init__()
            # ========== 在这里定义 ln1, attn, ln2, ffn ==========

        def forward(self, x, mask=None):
            # ========== 按 Pre-LN 顺序实现，带上两处残差 ==========
            out = None
            return out

    def main():
        block = DecoderBlock(d_model=16, num_heads=2, ffn_hidden=32)
        x = torch.randn(1, 6, 16)
        out = block(x)
        print("T4 out shape:", out.shape)

    if __name__ == "__main__":
        main()


"""
思考题（做完再看）：
Q1：Pre-LN 和 Post-LN 的区别是什么？为什么现在大模型普遍用 Pre-LN？
Q2：因果掩码（causal mask）长什么样子？它是阻止什么？
Q3：如果 d_model 不能被 num_heads 整除，会发生什么？
Q4：残差连接的核心作用是什么，梯度层面怎么理解？
"""
