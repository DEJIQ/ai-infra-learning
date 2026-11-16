# M0 Week3 Day6 复盘笔记：Transformer Decoder 基础

> 配套：PyTorch 自注意力、多头注意力、Decoder Block；前置：张量算子、广播、SVD、矩阵线性变换 术语对照（中文｜英文）
>
> - 词嵌入｜embedding
> - 查询｜Query (Q)
> - 键｜Key (K)
> - 值｜Value (V)
> - 缩放点积注意力｜Scaled Dot-Product Attention
> - 多头注意力｜Multi-Head Attention
> - 掩码 / 因果掩码｜Mask / Causal Mask
> - 残差连接｜Residual Connection
> - 层归一化｜Layer Normalization（LN）
> - 前馈网络｜Feed Forward Network（FFN）
> - Pre-LN：先归一化，再做注意力 / 前馈（GPT 系列标准结构）

## 一、核心公式回顾

### 1. 缩放点积注意力

\(\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\)

- \(d_k\)：单头 Q/K 的维度，除以 \(\sqrt{d_k}\) 是**缩放**，防止点积数值太大，softmax 梯度饱和
- 因果掩码（causal mask）：上三角填充 `-inf`，softmax 之后变成 0，保证每个 token **只能看到前面的 token，看不到未来 token**（GPT 自回归的核心）

### 2. 多头注意力

不是算一次大注意力，而是把 Q/K/V 切分成多个头，各自独立算注意力，最后拼接投影：

\(\begin{align*} \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\ \text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O \end{align*}\)

- 好处：多组独立的注意力子空间，捕捉不同类型依赖（语法、指代、长距离关联）
- shape 变化重点（你手动推导的那组）： `[B, L, D]` → split heads → `[B, h, L, d_k]`，做完注意力再合并回 `[B, L, D]`

### 3. Pre-LN Decoder Block（GPT 用的版本）

```
x = x + MHA(LN(x))
x = x + FFN(LN(x))
```

> 区别：Post-LN 是先算模块再加残差、最后归一化；GPT 统一 Pre-LN FFN（前馈网络，GPT 里是 GELU 版本）：
>
> \(\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1)+b_2\) 一般中间维度是 4× 隐藏维度

## 二、高频坑点汇总（你踩过的）

1. 高维张量转置：`[B, h, L, dk]` 不能用 `.T`，要用 `.mT`，`.T` 只适合二维矩阵，高维会报警告、维度错乱
2. 因果掩码：mask 加在 \(QK^\top\) 之后、softmax 之前；不要加在 softmax 输出上
3. 残差：**残差两边维度必须一模一样**，不然无法相加
4. LayerNorm：是对最后一维（特征维度）归一化，不是对序列长度维度，和 BatchNorm 完全不一样
5. 多头切分：切完之后一定要交换维度，把头放到序列前面，才能批量矩阵相乘

## 三、你本次的练习清单

- T1：手写缩放点积注意力（带因果掩码）
- T2：手写多头注意力模块
- T3：手写 GPT 风格 FFN（GELU 激活）
- T4：组装 Pre-LN Decoder Block

## 四、思考题答案简记

1. 缩放 \(\sqrt{d_k}\) 的作用：控制点积方差，避免 softmax 进入饱和区、梯度消失
2. 因果掩码作用：自回归，保证生成时看不到未来 token
3. Pre-LN 对比 Post-LN：训练更稳定，GPT、LLaMA 全部采用 Pre-LN