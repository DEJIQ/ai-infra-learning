# Transformer 从零到实现：给只有线性代数基础的小白

> 目标读者：会一点矩阵乘法、向量点积，但对深度学习 / PyTorch 还陌生。
> 读完本文你将能够：
> 1. 说出 Transformer 为什么被发明出来，解决了什么问题；
> 2. 亲手推导 Attention（注意力）的每一步数学公式，并理解每个符号的含义；
> 3. 理解 Multi-Head Attention、Positional Encoding、残差连接、LayerNorm、Mask 分别做什么、为什么需要；
> 4. 看懂并运行一份完整的 mini-Transformer 代码。

---

## 目录

1. [学习路线与心态](#1-学习路线与心态)
2. [前置知识温习：你要用到的数学工具](#2-前置知识温习你要用到的数学工具)
3. [为什么要发明 Transformer](#3-为什么要发明-transformer)
4. [注意力机制：一切的核心](#4-注意力机制一切的核心)
5. [缩放点积注意力：从直觉到公式](#5-缩放点积注意力从直觉到公式)
6. [多头注意力 Multi-Head Attention](#6-多头注意力-multi-head-attention)
7. [位置编码 Positional Encoding](#7-位置编码-positional-encoding)
8. [残差连接与 LayerNorm](#8-残差连接与-layernorm)
9. [前馈网络 Feed-Forward Network](#9-前馈网络-feed-forward-network)
10. [掩码 Mask：padding 掩码与因果掩码](#10-掩码-maskpadding-掩码与因果掩码)
11. [Transformer 整体架构](#11-transformer-整体架构)
12. [完整代码实现：一个可运行的 mini-Transformer](#12-完整代码实现一个可运行的-mini-transformer)
13. [训练与推理](#13-训练与推理)
14. [常见问题与面试题](#14-常见问题与面试题)
15. [进一步学习资源](#15-进一步学习资源)

---

## 1. 学习路线与心态

Transformer 是 2017 年论文《Attention Is All You Need》提出的架构，现在是几乎所有大语言模型（GPT、BERT、LLaMA、Claude……）的底座。

它的名字容易唬人，但**核心其实只有一个公式**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

你的学习顺序应该是：

```
矩阵乘法 → softmax → 这个公式 → 多头 → 位置编码 → 残差+归一化 → 整体拼起来
```

**别急着背代码，先把每个张量的形状（shape）搞清楚**。Transformer 90% 的困惑都来自"这个张量现在是什么形状"。本文每一步都会标注形状。

---

## 2. 前置知识温习：你要用到的数学工具

这一节只需"看懂"，不用推导。我们只用到四种工具。

### 2.1 向量与点积（Dot Product）

向量就是一列数。两个向量 **a** 和 **b** 的点积：

$$
\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n
$$

**直觉**：点积衡量两个向量"方向有多一致"。
- 方向越接近，点积越大；
- 正交（垂直），点积 ≈ 0；
- 方向相反，点积为负。

例：`[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32`

> 在注意力机制里，点积用来衡量"两个词向量有多相关"。

### 2.2 矩阵乘法

矩阵 `A`（形状 `m × n`）乘矩阵 `B`（形状 `n × p`）得到 `C`（形状 `m × p`）：

$$
C_{ij} = \sum_k A_{ik} B_{kj}
$$

**记形状的口诀**：`(m × n) @ (n × p) = (m × p)`，**中间的 n 必须相等**，结果取两边的 m 和 p。

`C` 的第 i 行第 j 列，等于 `A` 的第 i 行 与 `B` 的第 j 列 做点积。

**一个重要观察**：矩阵乘法 `A @ B` 可以看成"用 A 的每一行，去和 B 的每一列做点积"。这正对应后面注意力的 `Q @ K^T`。

在 PyTorch 里：
- `A @ B` = 矩阵乘法（`torch.matmul`）
- `A * B` = 逐元素相乘（元素一一对应相乘），**完全不同，别搞混**
- `K.T` 或 `K.transpose(0, 1)` = 转置（行变列）

### 2.3 softmax：把一堆数变成"和为 1 的概率"

给定一组数 `z = [z_1, z_2, ..., z_n]`，softmax 定义为：

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

**性质**：
- 每个输出都在 (0, 1) 之间；
- **所有输出加起来等于 1**，所以可以当"概率"看；
- 输入里越大的数，输出越接近 1（被"放大"了）。

**具体例子**，输入 `[1, 2, 3]`：

```
e^1 = 2.718,  e^2 = 7.389,  e^3 = 20.086
分母 = 2.718 + 7.389 + 20.086 = 30.193
输出 = [2.718/30.193, 7.389/30.193, 20.086/30.193]
     = [0.09, 0.24, 0.67]   # 加起来 = 1
```

> 注意：softmax 里用的是 `e`（自然常数 ≈ 2.718），不是随便选的。它让"最大值的差距"被拉大。

### 2.4 可学习参数 与 梯度下降（只需建立概念）

- **可学习参数（parameter）**：模型里那些"从数据里学出来的数字"。比如线性层的权重矩阵 W 和偏置 b。
- **前向传播（forward）**：输入 → 经过一堆运算 → 得到输出。
- **损失函数（loss）**：衡量"输出和正确答案差多少"，越小越好。
- **反向传播（backward）**：从 loss 出发，算出"每个参数往哪个方向微调能让 loss 变小"（这就是**梯度**，即导数）。
- **梯度下降**：沿梯度反方向，一点点更新参数。

本文你不需要手推梯度，只要知道：**PyTorch 会替你自动算梯度，你只需定义好前向传播**。

---

## 3. 为什么要发明 Transformer

### 3.1 任务背景：序列建模

很多数据是"一串"东西：一句话是词的序列、一段 DNA 是碱基序列、一段音乐是音符序列。

我们的目标是：**给定一个序列，让模型"理解"它，并产出新的序列**（翻译、续写、问答……）。

### 3.2 老办法：RNN / LSTM，以及它的两个硬伤

Transformer 之前，主流是 **RNN（循环神经网络）**：一个词一个词地读，每个时刻把"前面读到的信息"压缩成一个隐藏状态 `h_t`，传给下一步。

```
x_1 → x_2 → x_3 → ... → x_n
 h_1 → h_2 → h_3 → ... → h_n   （h 是逐步传递的记忆）
```

它有两个致命问题：

1. **串行，不能并行**：必须等第 t 步算完才能算第 t+1 步。句子一长，训练极慢，GPU 的优势发挥不出来。
2. **长距离依赖丢失**：第一个词的信息要穿过几十个"记忆更新"才能影响到最后一个词，中间会被稀释、遗忘。

### 3.3 Transformer 的思路：不要"一步步读"，要"所有词两两直接看"

Transformer 的核心想法：**放弃逐步传递记忆，让序列里的任意两个位置直接建立联系**。

具体做法就是"注意力"：对于每个位置，一次性算出"它应该关注其他所有位置的多少程度"。

- ✅ 完全并行：所有位置同时算，GPU 直接起飞。
- ✅ 任意距离的依赖，一步直达，不会"遗忘"。

> 论文标题《Attention Is All You Need》字面意思就是"有注意力就足够了"——连 RNN 那种循环结构都不需要了。

---

## 4. 注意力机制：一切的核心

### 4.1 一个生活化类比：查数据库

想象你在图书馆查资料：

- **Query（查询 Q）**：你想找什么。比如"我想找关于猫的资料"。
- **Key（键 K）**：每本书贴的标签。比如"狗"、"猫"、"汽车"。
- **Value（值 V）**：每本书的实际内容。

检索过程：
1. 拿你的 Query 去和每一本书的 Key **比较相似度**（点积）；
2. 把相似度**归一化成权重**（softmax），相似度越高权重越大；
3. 按权重**加权平均**所有书的 Value，得到你要的结果——"跟猫相关的书内容占大头"。

```
权重 = softmax(Query 与每个 Key 的相似度)
输出 = 权重 · Value 的加权和
```

这就是注意力。**Q、K、V 三个符号全部来自这个类比**。

### 4.2 在自然语言里，类比对应什么

一句话："The cat sat on the mat."

对于单词 "cat"，模型想知道：理解 "cat" 时，应该多看句中哪些词？

- `Q` = "cat" 的查询向量（我是什么，我要找谁）
- `K` = 每个词的键向量（我能提供什么信息供别人查询）
- `V` = 每个词的值向量（我实际携带的内容）

"cat" 的 Q 与 "sat"、"mat" 的 K 点积会比较大 → 权重高 → 最终 "cat" 的新表示里融入了这些词的信息。

### 4.3 Q、K、V 从哪来

同一个输入 `X`（每个词一个向量），乘以三个**不同的可学习矩阵**得到：

```
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
```

`W_Q, W_K, W_V` 是模型要学的参数。同一个词，作为查询者（Q）、被查者（K）、内容提供者（V）时，会被投影成三个不同的向量——**角色不同，表示也不同**。

> 这就是你之前代码里看到的 `Q = X @ Wq`、`K = X @ Wk`、`V = X @ Wv`。

---

## 5. 缩放点积注意力：从直觉到公式

### 5.1 完整公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 `d_k` 是 Q、K 的向量维度。

### 5.2 一步步推导（带形状）

设序列长度 `L = 4`（4 个词），每个词向量维度 `d_k = 8`。

**第 1 步：算相似度 `Q @ K^T`**

```
Q: [L, d_k] = [4, 8]
K: [L, d_k] = [4, 8]
K^T: [d_k, L] = [8, 4]
Q @ K^T: [4, 4]   （4 个 query × 4 个 key，两两相似度）
```

结果矩阵第 (i, j) 个元素 = 第 i 个词的 Q 与第 j 个词的 K 的**点积**，衡量"词 i 该关注词 j 多少"。

**第 2 步：缩放 `/ sqrt(d_k)`**

除以 $\sqrt{d_k}$。**为什么？**

假设 Q、K 每个元素独立、均值 0 方差 1。两个 `d_k` 维向量点积 = `d_k` 项相加，**方差会变成 `d_k`**（方差可加）。

`d_k` 越大 → 点积数值越大 → softmax 会变得"极端"（最大值被压到接近 1，其他接近 0）→ 梯度消失，训练不动。

除以 $\sqrt{d_k}$ 恰好把方差拉回 1，让 softmax 保持"温和"。**这就是"缩放（scaled）"的由来。**

**第 3 步：softmax 归一化**

```python
attn_weight = softmax(scores, dim=-1)   # 沿最后一个维度，即每一行
```

对每一行（每个 query）做 softmax，使**该 query 对 4 个 key 的权重加起来 = 1**。

**第 4 步：加权求和 `@ V`**

```
attn_weight: [4, 4]
V: [4, d_k] = [4, 8]
out = attn_weight @ V : [4, 8]
```

每个词的输出 = 所有词的 V 按权重的加权平均。至此，每个词的新表示里"融入了它关注的那些词的信息"。

### 5.3 一个手算例子（帮你建立直觉）

序列 2 个词，`d_k = 2`。设：

```
Q = [[1, 0],      # 词 0 的 query
     [0, 1]]      # 词 1 的 query
K = [[1, 0],      # 词 0 的 key
     [0, 1]]      # 词 1 的 key
V = [[10, 20],    # 词 0 的 value
     [30, 40]]    # 词 1 的 value
```

**第 1 步** `Q @ K^T`：

```
Q @ K^T = [[1, 0],
           [0, 1]]
```

词 0 的 Q 和词 0 的 K 点积 = 1，和词 1 的 K 点积 = 0。所以词 0 "只关注自己"。

**第 2 步** 假设 `d_k = 2`，除以 sqrt(2) ≈ 1.414：

```
scores = [[0.707, 0],
          [0, 0.707]]
```

**第 3 步** softmax（沿每行）：

```
softmax(0.707, 0) = [e^0.707, e^0] / (e^0.707 + e^0)
                  ≈ [2.028, 1] / 3.028
                  ≈ [0.67, 0.33]
```

所以权重矩阵 ≈ `[[0.67, 0.33], [0.33, 0.67]]`。

**第 4 步** 加权求和 `attn @ V`：

```
out = [[0.67, 0.33],   [10, 20],
       [0.33, 0.67]] @ [30, 40]
    = [[0.67*10+0.33*30, 0.67*20+0.33*40],   # [16.6, 26.6]
       [0.33*10+0.67*30, 0.33*20+0.67*40]]   # [23.4, 33.4]
```

每个词的输出，都是"自己 + 别的词"的加权混合。**这就是注意力在做什么。**

### 5.4 对应代码

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V: [L, d]
    d_k = Q.size(-1)                        # 取最后一维 = d_k
    # 1. 相似度 + 缩放
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # [L, L]
    # 2. 可选：掩码（后面细讲）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # 3. softmax 归一化
    attn = F.softmax(scores, dim=-1)        # [L, L]，每行和为 1
    # 4. 加权求和
    out = attn @ V                           # [L, d]
    return out, attn
```

> 说明：`K.transpose(-2, -1)` 是"交换最后两个维度"，对 2D 张量就是转置。用 `-2, -1` 而不是 `0, 1`，是为了兼容后面 3D、4D 张量。

---

## 6. 多头注意力 Multi-Head Attention

### 6.1 为什么需要多头

单个注意力只能学到"一种"关注模式。但语言里一个词可能需要**同时**关注很多不同的东西：

- "他 和 她 去了 学校" 中，"他"要关注"她"（语法关系），也要关注"学校"（语义关系）。
- 只用一个 head，这些关注会混在一起、互相干扰。

**多头**的思路：把 Q、K、V 的维度**切成 h 份**，每一份（一个"头"）独立做一次注意力，最后把结果拼起来。这样不同头可以学到**不同的关注模式**。

### 6.2 数学描述

```
head_i = Attention(Q @ W_i^Q,  K @ W_i^K,  V @ W_i^V)      # 每个头自己的投影
MultiHead = Concat(head_1, head_2, ..., head_h) @ W^O     # 拼起来再过一层
```

设 `d_model = 8`，`h = 2` 个头，则每个头 `d_k = 8/2 = 4`。

### 6.3 张量形状（重点）

多头最绕的地方就是形状变换。设 `batch=1`，序列 `L=4`，`d_model=8`，`h=2`：

```
X:        [batch, L, d_model] = [1, 4, 8]
投影后 Q: [1, 4, 8]
切成 h 个头:  [1, 4, h, d_k] = [1, 4, 2, 4]
交换维度:     [1, h, L, d_k] = [1, 2, 4, 4]   # 把"头"挪到前面，方便批量矩阵乘法
做注意力:     [1, h, L, d_k] = [1, 2, 4, 4]   # 每个头独立算，互不影响
拼回来:       [1, L, d_model] = [1, 4, 8]
再过 W^O:     [1, 4, 8]
```

关键操作：`.view()` 改变形状，`.transpose(1, 2)` 交换"头"和"序列"两个维度。

### 6.4 对应代码

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 四个可学习的线性层（Q、K、V 的投影 + 输出的投影）
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch, seq_len, _ = Q.shape

        # 投影 + 切成多头: [batch, L, d_model] -> [batch, L, h, d_k] -> [batch, h, L, d_k]
        Q = self.Wq(Q).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(K).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(V).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 每个头独立做缩放点积注意力
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)   # [batch, h, L, L]
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V                                          # [batch, h, L, d_k]

        # 拼回头: [batch, h, L, d_k] -> [batch, L, h, d_k] -> [batch, L, d_model]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.Wo(out), attn
```

> `nn.Linear(d_in, d_out)` 是一个线性层，内部就是 `x @ W^T + b`，其中 W、b 是可学习参数。`nn.Module` 是所有网络层的基类，`super().__init__()` 是调用父类构造器（必须写）。

---

## 7. 位置编码 Positional Encoding

### 7.1 为什么需要

注意力机制本身**对顺序不敏感**：把句子里的词顺序打乱，`Q @ K^T` 算出来的相似度完全一样（因为点积不在乎位置）。

但语言里顺序极其重要："狗咬人" ≠ "人咬狗"。

所以必须在词向量里**注入位置信息**。做法：给每个位置的词向量，加一个"位置向量"：

```
X_with_pos = X + PE        # PE[pos] 只和位置 pos 有关
```

### 7.2 论文用的正弦位置编码

位置 `pos` 的编码向量的第 `i` 个维度：

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

- `pos`：位置（0, 1, 2, ...）
- `i`：维度索引（0, 1, ...）
- 偶数维用 sin，奇数维用 cos。

**直觉**：不同的维度使用**不同频率**的正弦波。低维度变化慢（区分"远距离位置关系"），高维度变化快（区分"相邻位置"）。就像用不同粗细的尺子去量位置。

**为什么用 sin/cos 而不是直接学一个随机表？**

- 可以对任意长度外推（训练时最长 50，测试时 100 也能算）；
- 相邻位置的编码是"平滑变化"的，模型容易学到相对位置关系。

（现代模型如 GPT 很多改用**可学习的位置编码**或 **RoPE**，但理解 sin/cos 是基础。）

### 7.3 对应代码

```python
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)          # [max_len, d_model]
        pos = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]，列向量
        # 分母：10000^(2i/d_model)，按维度 i 计算
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)   # 偶数维：sin
        pe[:, 1::2] = torch.cos(pos * div)   # 奇数维：cos
        pe = pe.unsqueeze(0)                 # [1, max_len, d_model]，加 batch 维
        self.register_buffer('pe', pe)       # 不是参数，不参与梯度更新，但要随模型保存

    def forward(self, x):
        # x: [batch, L, d_model]，直接加上前 L 个位置编码
        return x + self.pe[:, :x.size(1)]
```

> `register_buffer`：把张量存进模型，但**不作为可学习参数**（不更新梯度），只在前向时随模型一起移动/保存。

---

## 8. 残差连接与 LayerNorm

Transformer 的每个子层外面都包着这两样东西。**它们不是注意力，但对训练成功至关重要。**

### 8.1 残差连接（Residual Connection）

```
输出 = 输入 + 子层(输入)      # 让梯度多一条"高速公路"
```

```python
x = x + self.attention(x)   # 而不是 x = self.attention(x)
```

**为什么？** 网络很深时，梯度要穿过几十层，乘来乘去容易**梯度消失**（趋于 0）或**梯度爆炸**（趋于无穷）。残差让梯度可以"抄近路"直接传回前面，训练就稳定了。

### 8.2 LayerNorm（层归一化）

对**每个样本内部的所有维度**做归一化：减均值、除以标准差，再做可学习的缩放平移：

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

其中 $\mu, \sigma$ 是 x 在该样本所有维度上的均值和标准差，$\gamma, \beta$ 是可学习参数。

**为什么？** 让每一层的数值保持稳定分布，训练更快更稳。

> 对比：BatchNorm 是对"不同样本的同一维度"归一化；LayerNorm 是对"同一样本的所有维度"归一化。NLP 里序列长度不定，所以用 LayerNorm。

### 8.3 两种拼接顺序：Post-LN 与 Pre-LN

论文原始写法是 **Post-LN**：

```
x = LayerNorm(x + Sublayer(x))
```

现代很多模型（GPT 等）改用 **Pre-LN**，更稳定：

```
x = x + Sublayer(LayerNorm(x))
```

本文代码用 Pre-LN（更易训练）。

---

## 9. 前馈网络 Feed-Forward Network

每个注意力层后面跟一个简单的前馈网络 FFN，对**每个位置独立**作用（不同位置之间不交流）：

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$

即：升维 → ReLU 激活 → 降维。

- 通常 `W_1` 把 `d_model` 升到 4 倍（如 512 → 2048），`W_2` 再降回来。
- 注意力负责"词之间交流信息"，FFN 负责"每个词内部做非线性变换/存储知识"。二者分工。

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 升维
        self.linear2 = nn.Linear(d_ff, d_model)  # 降维

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
```

---

## 10. 掩码 Mask：padding 掩码与因果掩码

### 10.1 为什么要 mask

softmax 会把每个位置都分配一些权重。但有些位置**不应该被关注**：

1. **padding 位置**：句子长度不一，短句会被补 `<pad>`。模型不该去关注这些填充位。
2. **未来位置**（生成时）：预测第 t 个词时，不能"偷看"第 t+1 个及之后的词，否则就是作弊。

### 10.2 做法：把要屏蔽的位置填 `-inf`

```python
scores = scores.masked_fill(mask == 0, float('-inf'))
```

因为 $\text{softmax}(-\infty) = 0$，被屏蔽位置的权重就变成 0。

- `masked_fill(条件, 值)`：条件为 True 的地方填成给定值。
- `mask == 0`：得到一个布尔张量，True 表示"要屏蔽"。

### 10.3 因果掩码（上三角掩码）

生成任务里，位置 i 只能看位置 0..i（含自己）。构造一个下三角为 1、上三角为 0 的矩阵：

```python
def causal_mask(L):
    return torch.tril(torch.ones(L, L))   # 下三角 = 1，上三角 = 0
```

```
L=4 时：
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

第 i 行只有前 i+1 列是 1 → 位置 i 只能看到 0..i。

---

## 11. Transformer 整体架构

### 11.1 总览图

原始 Transformer 是 **Encoder-Decoder** 结构（用于翻译）。

```
输入序列
   │
Embedding + Positional Encoding
   │
┌─────────────────────────┐
│  Encoder × N 层          │
│   ├ Multi-Head Attention │  (self-attention)
│   ├ Add & Norm           │
│   ├ Feed-Forward         │
│   └ Add & Norm           │
└─────────────────────────┘
   │  (编码好的"语义表示")
   ▼
┌─────────────────────────┐
│  Decoder × N 层          │
│   ├ Masked MHA           │  (带因果掩码的 self-attention)
│   ├ Add & Norm           │
│   ├ Cross-Attention      │  (Q 来自 decoder，K/V 来自 encoder)
│   ├ Add & Norm           │
│   ├ Feed-Forward         │
│   └ Add & Norm           │
└─────────────────────────┘
   │
Linear + Softmax → 输出每个位置的下一个词概率
```

### 11.2 三种注意力

| 类型 | Q 来自 | K/V 来自 | 作用 |
|------|--------|----------|------|
| Self-Attention（Encoder 里） | 输入自己 | 输入自己 | 序列内部词之间交流 |
| Masked Self-Attention（Decoder 里） | 输出自己 | 输出自己 | 同上，但加因果掩码 |
| Cross-Attention（Decoder 里） | 输出 | Encoder 输出 | 让 decoder 能"看"源语言信息 |

### 11.3 现代变体（重要）

- **Encoder-only**：BERT 系列，适合"理解"任务（分类、问答）。
- **Decoder-only**：GPT 系列，只有 Masked Self-Attention + FFN，适合"生成"任务。**现在的大语言模型基本都是这种**。
- **Encoder-Decoder**：原始 Transformer、T5、BART，适合"序列到序列"（翻译、摘要）。

你只要理解了上面的模块，三种变体就是"组合/删减"而已。

---

## 12. 完整代码实现：一个可运行的 mini-Transformer

下面是一份**完整、可运行**的 mini-Transformer（decoder-only，类似 GPT），从头用 PyTorch 实现，没有调用任何现成 Transformer 库。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. 位置编码 ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()   # [max_len, 1]
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))       # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))           # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---------- 2. 多头注意力 ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch, seq_len, _ = Q.shape
        Q = self.Wq(Q).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(K).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(V).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)   # [b, h, L, L]
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V                                           # [b, h, L, d_k]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.Wo(out), attn


# ---------- 3. 前馈网络 ----------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


# ---------- 4. 一个 Decoder 层（Pre-LN） ----------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 残差 + 注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)[0]
        # 残差 + FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ---------- 5. 完整的 Decoder-only Transformer ----------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)   # 词嵌入表
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)        # 输出每个词的分数

    def forward(self, x, mask=None):
        x = self.embedding(x)            # [b, L] -> [b, L, d_model]
        x = self.pos_enc(x)              # 加位置编码
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)           # [b, L, vocab_size]


# ---------- 6. 测试：前向传播能否跑通 ----------
if __name__ == '__main__':
    vocab_size = 100
    d_model = 64
    model = MiniGPT(vocab_size=vocab_size, d_model=d_model,
                    n_heads=4, n_layers=2, d_ff=128)

    batch, seq_len = 2, 8
    x = torch.randint(0, vocab_size, (batch, seq_len))      # 随机输入 token
    mask = torch.tril(torch.ones(seq_len, seq_len))         # 因果掩码

    logits = model(x, mask)                                 # [2, 8, 100]
    print("输入 shape:", x.shape)
    print("输出 logits shape:", logits.shape)               # torch.Size([2, 8, 100])

    # 算一个假的 loss，验证能反向传播
    loss = logits.sum()
    loss.backward()
    print("反向传播成功，loss =", loss.item())
```

### 12.1 逐模块说明

- `nn.Embedding(vocab_size, d_model)`：词嵌入表。输入是词 id（整数），输出对应词向量。本质是一张 `[vocab_size, d_model]` 的查找表。
- `nn.ModuleList`：把若干子层装进一个列表，这样它们才能被 PyTorch 正确识别为模型的一部分。
- `lm_head = nn.Linear(d_model, vocab_size)`：把每个位置的向量映射成"下一个词是每个候选词"的分数（logits），之后 softmax 得到概率。
- 前向流程：`token id → 词向量 → +位置编码 → N 个 DecoderLayer → LayerNorm → logits`。

---

## 13. 训练与推理

### 13.1 训练（Teacher Forcing）

训练时，给模型输入 `[w_1, w_2, ..., w_t]`，让它预测 `[w_2, w_3, ..., w_{t+1}]`（即**每个位置预测下一个词**）。因为"标准答案"已知，且用了因果掩码，所以**整个序列可以一次性并行前向**。

损失函数：`CrossEntropyLoss`（交叉熵），衡量"预测的词分布"和"真实的下一个词"差距。

```python
criterion = nn.CrossEntropyLoss()
logits = model(x, mask)              # [b, L, vocab]
logits = logits[:, :-1, :]           # 预测的是"下一个词"
targets = x[:, 1:]                   # 真实的下一个词
loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
loss.backward()
optimizer.step()                     # 更新参数
```

### 13.2 推理（自回归生成）

推理时没有标准答案，只能**一个词一个词地生成**：生成第 t 个词后，把它拼回输入，再预测第 t+1 个词，直到遇到结束符。

```python
model.eval()
tokens = [bos_id]                    # 从起始符开始
for _ in range(max_new_tokens):
    x = torch.tensor([tokens])
    logits = model(x)[0, -1, :]      # 只看最后一个位置的输出
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1).item()  # 按概率采样
    tokens.append(next_token)
    if next_token == eos_id:
        break
```

> 为什么推理必须"串行"？因为第 t+1 个词依赖第 t 个词，没法并行。这正是为什么"训练快、推理慢"。

---

## 14. 常见问题与面试题

**Q1：为什么要除以 sqrt(d_k)？**
防止 d_k 增大时点积方差变大，导致 softmax 进入饱和区、梯度消失。

**Q2：softmax 的 dim 为什么是 -1？**
因为要对"每个 query 对应的那排 key 权重"归一化（每行和为 1），也就是最后一个维度。

**Q3：多头和单头的本质区别？**
单头只有一个关注模式；多头把维度切分，让模型同时学多种关注模式，再拼接融合。

**Q4：self-attention 和 cross-attention 区别？**
self：Q、K、V 来自同一序列；cross：Q 来自一个序列，K/V 来自另一个序列（如翻译中 decoder 看 encoder）。

**Q5：位置编码为什么用 sin/cos？**
无参数、可外推任意长度、相邻位置平滑变化，模型易学相对位置。

**Q6：LayerNorm 和 BatchNorm 区别？**
LN 对同一样本所有维度归一化（适合变长序列）；BN 对不同样本同一维度归一化（适合图像）。

**Q7：为什么需要残差连接？**
让梯度能直接流回浅层，缓解深层网络的梯度消失。

**Q8：masked_fill 里为什么填 -inf 而不是 0？**
因为 softmax(-inf) = 0；若填 0，softmax(0) = 1/n ≠ 0，屏蔽不掉。

**Q9：`@` 和 `*` 的区别？**
`@` 矩阵乘法，`*` 逐元素相乘。形状规则完全不同。

**Q10：训练为什么要用因果掩码？**
防止模型在预测位置 t 时"偷看"未来的词，保证训练和推理时行为一致。

---

## 15. 进一步学习资源

1. **原始论文**：Vaswani et al., *Attention Is All You Need* (2017)。建议读完本文后再读，会豁然开朗。
2. **图解 Transformer**：Jay Alammar 的《The Illustrated Transformer》（有大量中文翻译版）。
3. **动手实现**：Andrej Karpathy 的《Let's build GPT: from scratch, in code, spelled out》（视频 + nanoGPT 代码）。
4. **nanoGPT**：一个极简、干净的 GPT 实现，适合读完本文后精读。

---

## 附：一页纸速查

```
注意力核心公式：
  Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V

形状速记（L=序列长度，d=d_model，h=头数）：
  X            [b, L, d]
  Q=K=V 投影   [b, L, d] → 切成 [b, h, L, d/h]
  scores       [b, h, L, L]
  attn         [b, h, L, L]   (最后一维和为1)
  out          [b, h, L, d/h] → 拼回 [b, L, d]

一个 Decoder 层：
  x = x + MHA(LayerNorm(x))     # 带因果掩码
  x = x + FFN(LayerNorm(x))

FFN：
  FFN(x) = W2(ReLU(W1 x + b1)) + b2    (d → 4d → d)

位置编码：
  PE(pos,2i)   = sin(pos / 10000^(2i/d))
  PE(pos,2i+1) = cos(pos / 10000^(2i/d))
```

> 建议阅读顺序：先看懂第 2、4、5 节（数学），再跑通第 12 节的代码，最后回头精读第 6~11 节的每个模块。**动手跑一遍代码，比读十遍文字更管用。**
