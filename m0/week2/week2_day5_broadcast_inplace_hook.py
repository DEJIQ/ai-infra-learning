import torch

# ========== 切换题目：仅保留一个 True，其余 False ==========
RUN_T1 = False
RUN_T2 = False
RUN_T3 = True
RUN_T4 = False
# ==========================================================


if RUN_T1:
    """
    T1：广播基础运算
    需求：
    1. 创建 a: shape [3, 1]，值为 [ [10], [20], [30] ]，float
    2. 创建 b: shape [1, 4]，值为 [ 1, 2, 3, 4 ]，float
    3. a + b，打印结果张量和它的 shape
    4. 使用 torch.broadcast_tensors，拿到广播对齐后的两个张量，分别打印 shape
    """
    def main():
        # 在这里写你的代码
        a = torch.tensor([[10.0], [20.0], [30.0]])
        b = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = a + b
        print("Result of a + b:\n", result)
        aa,bb = torch.broadcast_tensors(a, b)
        print("Shape of broadcasted a:", aa.shape)
        print("Shape of broadcasted b:", bb.shape)
        pass


if RUN_T2:
    """
    T2：原地操作与梯度报错
    需求：
    1. 创建 x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    2. 尝试使用 x.add_(1.0)，捕获报错（不需要修复，只要观察理解）
    3. 再创建 y = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
    4. 对 y 使用 add_(5.0)，打印 y
    """
    def main():
        # 在这里写你的代码
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        try:
            x.add_(1.0)
        except RuntimeError as e:
            print("Caught an error when trying to perform in-place operation on x:", e)
        y = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
        y.add_(5.0)
        print("Value of y after in-place operation:", y)
        pass


if RUN_T3:
    """
    T3：梯度钩子：打印中间梯度
    需求：
    1. x = torch.tensor([3.0], requires_grad=True)
    2. y = x ** 3
    3. z = 2 * y
    4. 在 y 上注册钩子：打印钩子收到的梯度
    5. z.backward()
    6. 打印 x.grad
    7. 移除钩子
    手动验算：
       dz/dy = 2
       dy/dx = 3*x^2
       dz/dx = 2 * 3 * x^2
    """
    def main():
        # 在这里写你的代码
        x = torch.tensor([3.0], requires_grad=True)
        y = x ** 3
        z = 2 * y
        def hook(grad):
            print("Gradient received by hook:", grad)
            return grad
        handle = y.register_hook(hook)
        z.backward()
        print("Gradient of x:", x.grad)
        handle.remove()
        pass


if RUN_T4:
    """
    T4：钩子篡改梯度
    需求：
    1. x = torch.tensor([2.0], requires_grad=True)
    2. y = x ** 2
    3. 在 y 的钩子里面，把传入梯度 × 0.5 之后返回
    4. y.backward()
    5. 打印 x.grad
    预期：原本梯度是 4，经过钩子缩放之后变成 2
    """
    def main():
        # 在这里写你的代码
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2
        def hook(grad):
            return grad * 0.5
        handle = y.register_hook(hook)
        y.backward()
        print("Gradient of x after hook modification:", x.grad)
        handle.remove()
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
# ========== 思考题（不用写进代码） ==========
# Q1：如果两个张量 shape [2, 3, 4] 和 [3, 1]，能不能广播？为什么？
可以，从右往左对齐，第二个张量的 shape 可以通过在前面补 1 来对齐第一个张量的 shape，具体如下：
第一个张量的 shape: [2, 3, 4]
第二个张量的 shape: [1, 3, 1]（在前面补 1）
对齐后：
第一个张量的 shape: [2, 3, 4]
第二个张量的 shape: [1, 3, 1]
然后按照广播规则：
- 对于第一个维度，2 和 1 可以广播为 2
- 对于第二个维度，3 和 3 可以广播为 3
- 对于第三个维度，4 和 1 可以广播为 4   
# Q2：为什么带 requires_grad=True 的叶子张量，很多原地操作不允许？
因为原地操作可能会破坏梯度计算的连贯性，导致梯度信息丢失或不准确。
# Q3：钩子拿到的梯度，是局部导数，还是上游传进来的梯度？
钩子拿到的梯度是上游传进来的梯度。
# Q4：如果忘记调用 handle.remove()，在循环里面反复 backward，会发生什么？
可能会导致内存泄漏或意外的行为，因为钩子会持续接收梯度信息。
"""