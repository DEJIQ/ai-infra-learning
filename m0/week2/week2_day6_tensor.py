import torch

# ========== 切换题目：仅保留一个 True，其余 False ==========
RUN_T1 = False
RUN_T2 = False
RUN_T3 = False
RUN_T4 = True
# ==========================================================


if RUN_T1:
    """
    T1：逐元素运算
    需求：
    1. a = torch.tensor([1., 2., 3., 4.])
    2. b = torch.tensor([10., 20., 30., 40.])
    3. 使用 torch.add，alpha=0.2，计算 a + 0.2 * b，打印结果
    4. 使用 torch.mul，计算 a 和 b 的逐元素乘积，打印
    5. 计算 a 的平方（torch.pow），打印
    6. 计算 torch.exp(a)，再取 torch.log，打印
    """
    def main():
        # 在这里写你的代码
        a = torch.tensor([1., 2., 3., 4.])
        b = torch.tensor([10., 20., 30., 40.])
        result_add = torch.add(a,b, alpha=0.2)
        print("Result of a + 0.2 * b:\n", result_add)
        result_mul = torch.mul(a,b)
        print("Result of a * b:\n", result_mul)
        result_pow = torch.pow(a,2)
        print("Result of a squared:\n", result_pow)
        result_exp_log = torch.log(torch.exp(a))
        print("Result of log(exp(a)):\n", result_exp_log)
        pass


if RUN_T2:
    """
    T2：归约算子 sum / mean / max / argmax
    需求：
    1. x = torch.tensor([[1, 9, 3],
                         [4, 0, 7]], dtype=torch.float32)
    2. dim=0 求和，打印结果 + shape
    3. dim=1 求和，keepdim=True，打印结果 + shape
    4. dim=1 求均值，打印
    5. dim=1 求最大值与对应下标（torch.max），分别打印 values, indices
    6. 单独用 torch.argmax 在 dim=1，打印下标，和上面对照
    """
    def main():
        # 在这里写你的代码
        x = torch.tensor([[1, 9, 3],
                          [4, 0, 7]], dtype=torch.float32)
        x_sum_dim0 = torch.sum(x, dim=0)
        print("Sum along dim=0:\n", x_sum_dim0, "Shape:", x_sum_dim0.shape)
        x_sum_dim1_keepdim = torch.sum(x, dim=1, keepdim=True)
        print("Sum along dim=1 with keepdim:\n", x_sum_dim1_keepdim, "Shape:", x_sum_dim1_keepdim.shape)
        x_mean_dim1 = torch.mean(x, dim=1)
        print("Mean along dim=1:\n", x_mean_dim1)
        val,idx = torch.max(x, dim=1)
        print("Max values along dim=1:\n", val)
        print("Indices of max values along dim=1:\n", idx)
        x_argmax_dim1 = torch.argmax(x, dim=1)  
        print("Argmax along dim=1:\n", x_argmax_dim1)
        pass


if RUN_T3:
    """
    T3：掩码 + torch.where
    需求：
    1. x = torch.tensor([[5, -1, 8],
                         [-3, 10, 2]])
    2. 生成掩码 mask：x > 0
    3. 使用 torch.where：大于 0 保留原值，小于等于 0 替换成 0
    4. 打印 mask 和最终结果
    """
    def main():
        # 在这里写你的代码
        x = torch.tensor([[5, -1, 8],
                          [-3, 10, 2]])
        mask = x > 0
        result = torch.where(mask, x, torch.tensor(0))
        print("Mask:\n", mask)
        print("Result after applying mask:\n", result)
        pass


if RUN_T4:
    """
    T4：gather
    需求：
    1. src = torch.tensor([[1, 2, 3],
                           [4, 5, 6]])
    2. index = torch.tensor([[2, 1, 0],
                            [0, 2, 1]], dtype=torch.long)
    3. dim=1，调用 torch.gather，取出对应元素
    4. 打印 gather 的结果
    5. 额外：只取第 0 行，index = torch.tensor([[2,0]], long)，dim=1，再跑一次 gather
    """
    def main():
        # 在这里写你的代码
        src = torch.tensor([[1, 2, 3],
                            [4, 5, 6]])
        index = torch.tensor([[2, 1, 0],
                              [0, 2, 1]], dtype=torch.long)
        result = torch.gather(src,dim = 1,index = index)
        print("Gather result:\n", result)
        # 额外：只取第 0 行
        src_row0 = src[0:1]  # 或者 src[0].unsqueeze(0)
        index_row0 = torch.tensor([[2, 0]], dtype=torch.long)
        result_row0 = torch.gather(src_row0, dim=1, index=index_row0)
        print("Gather result for row 0:\n", result_row0)
        pass


if __name__ == "__main__":
    if RUN_T1:
        main()
    elif RUN_T2:
        main()
    elif RUN_T3:
        main()
    elif RUN_T4:
        main()


"""
# ========== 思考题（不用写代码） ==========
Q1：torch.mul 是逐元素乘；矩阵乘法用哪几个函数？
矩阵乘法可以使用 torch.mm、torch.matmul 或者 @ 运算符。
Q2：torch.sum(x, dim=None) 的效果是什么？
torch.sum(x, dim=None) 会计算张量中所有元素的和。
Q3：torch.where 的三个参数如果形状不一样，会发生什么？
如果三个参数的形状不一致，PyTorch 会抛出一个错误。
Q4：gather 的 index 如果不是 long 类型，会怎么样？
如果 index 不是 long 类型，PyTorch 会抛出一个错误。
"""
