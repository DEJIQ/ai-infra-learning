import torch

# ========== 切换题目：仅保留一个 True，其余 False ==========
RUN_T1 = True
RUN_T2 = False
RUN_T3 = False
RUN_T4 = False
RUN_T5 = False
# ==========================================================


if RUN_T1:
    """
    T1：cat / stack
    需求
    1. a = torch.arange(0,6).reshape(2,3)
       b = torch.arange(6,12).reshape(2,3)
    2. 使用 cat，沿着 dim=0 拼接，打印 shape
    3. 使用 cat，沿着 dim=1 拼接，打印 shape
    4. 使用 stack，把 a,b 在 dim=0 打包，打印 shape
    """
    def main():
        # 在这里写你的代码
        a = torch.arange(0,6).reshape(2,3)
        b = torch.arange(6,12).reshape(2,3)
        c =torch.cat([a,b],dim = 0)
        print(c.shape)
        d = torch.cat([a,b],dim = 1)
        print(d.shape)
        e = torch.stack([a,b],dim = 0)
        print(e.shape)
        pass

    if __name__ == "__main__":
        main()


if RUN_T2:
    """
    T2：reduce 求和、max、keepdim
    M = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=torch.float32)
    需求
    1. 沿着 dim=0 求和，不保留维度，打印结果
    2. 沿着 dim=1 求最大值，keepdim=True，打印 value 和 index
    3. 计算整张矩阵所有数字总和
    """
    def main():
        # 在这里写你的代码
        M = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=torch.float32)
        N = M.sum(dim=0)
        print(N)
        val,idx = M.max(dim=1,keepdim=True)
        print(val)
        print(idx)
        total_sum = M.sum()
        print(total_sum)
        pass

    if __name__ == "__main__":
        main()


if RUN_T3:
    """
    T3：布尔掩码
    x = torch.tensor([2, -1, 5, -10, 8, 0])
    需求
    1. 取出所有大于 0 的数字并打印
    2. 把所有小于 0 的数字原地改成 0
    3. 统计大于 0 的元素一共有多少个
    """
    def main():
        # 在这里写你的代码
        x = torch.tensor([2, -1, 5, -10, 8, 0])
        print(x[x>0])
        mask = x<0
        x[mask] = 0
        print(x)
        count = (x>0).sum()
        print(count)
        pass

    if __name__ == "__main__":
        main()


if RUN_T4:
    """
    T4：逐元素乘 vs 矩阵乘法
    A = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
    B = torch.tensor([[0,1],[1,0]], dtype=torch.float32)
    需求
    1. A * B（逐元素）打印
    2. A @ B（矩阵乘法）打印
    3. 构造一维向量 v = torch.tensor([1, 2])，用 mv 做 A.mv(v)
    """
    def main():
        # 在这里写你的代码
        A = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
        B = torch.tensor([[0,1],[1,0]], dtype=torch.float32)
        print(A*B)
        print(A@B)
        v = torch.tensor([1, 2], dtype=torch.float32)
        print(A.mv(v))
        pass

    if __name__ == "__main__":
        main()


if RUN_T5:
    """
    T5：permute + clamp
    x = torch.randn(2, 4, 8)
    需求
    1. permute，把维度顺序改成 (4, 8, 2)，打印 shape
    2. 将里面所有元素 clamp 到 [-1, 1]
    3. 打印修改之后张量的最大值、最小值
    """
    def main():
        # 在这里写你的代码
        x = torch.randn(2, 4, 8)
        y = x.permute(2,0,1)
        print(y.shape)
        z = torch.clamp(y, min=-1, max=1)
        print(z.max())
        print(z.min())
        pass

    if __name__ == "__main__":
        main()


"""
===== 思考题（不参与编译）
Q1：cat 和 stack 核心区别，如果两个 shape=(3,4) 的张量 stack(dim=0)，输出 shape？
cat 和 stack 的核心区别在于，cat 是沿着指定的维度拼接张量，而 stack 是在新的维度上堆叠张量。对于两个 shape=(3,4) 的张量，如果使用 stack(dim=0)，输出的 shape 将是 (2, 3, 4)，因为 stack 会在第一个维度上创建一个新的维度来堆叠这两个张量。      
Q2：M.sum(dim=1, keepdim=True) 和 M.sum(dim=1) 在后续广播的时候会有什么差别？
M.sum(dim=1, keepdim=True) 会保留原来的维度，使得输出的张量在求和的维度上仍然有一个维度，形状为 (3, 1)。而 M.sum(dim=1) 会去掉求和的维度，输出的张量形状为 (3,)。在后续广播操作中，如果需要与原始张量进行操作，使用 keepdim=True 可以确保形状匹配，从而避免广播错误。   
Q3：掩码取出元素之后得到的张量，和原张量共享内存吗？修改它会不会影响原来？
掩码取出元素之后得到的张量通常不会与原张量共享内存，而是创建了一个新的张量。因此，修改这个新张量不会影响原来的张量。只有在使用某些特定的操作（如 view 或者直接索引）时，才可能会共享内存，但一般情况下，掩码操作会返回一个新的张量。    
Q4：什么时候优先用 permute，什么时候只用 transpose？
permute 可以同时改变多个维度的顺序，而 transpose 只能交换两个维度。因此，当需要重新排列多个维度时，应该使用 permute；而当只需要交换两个维度时，可以使用 transpose。
Q5：in-place 操作（add_）为什么在 autograd 里面很容易翻车？
in-place 操作（如 add_）会直接修改原始张量的内容，这可能会破坏计算图的结构，导致梯度计算出现问题。在 autograd 中，计算图依赖于张量的历史操作来正确地计算梯度，如果原地修改了张量，可能会丢失必要的信息，从而导致梯度计算错误或无法计算。因此，在使用 autograd 时，通常建议避免使用 in-place 操作。
Q6：torch.clamp 有什么典型用途？举一个 LLM 推理场景例子。
torch.clamp 的典型用途是限制张量的值在一个指定的范围内，防止数值过大或过小。在 LLM（大型语言模型）推理场景中，clamp 可以用于限制模型输出的 logits 值，以避免数值溢出或梯度爆炸。例如，在生成文本时，可以将 logits 限制在一个合理的范围内，以确保 softmax 操作后的概率分布不会出现极端值，从而提高生成文本的稳定性和质量。
"""
