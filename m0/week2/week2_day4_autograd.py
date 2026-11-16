import torch

# ========== 切换题目：仅保留一个 True，其余 False ==========
RUN_T1 = False
RUN_T2 = False
RUN_T3 = False
RUN_T4 = True
# ==========================================================


if RUN_T1:
    """
    T1：基础标量求导
    需求：
    1. 创建 x = torch.tensor([2.0], requires_grad=True)
    2. 计算 y = x ** 2 + 3 * x + 1
    3. y.backward() 反向传播
    4. 打印 x.grad
    5. 手动验证：导数 dy/dx = 2x + 3；代入 x=2，结果应该是 7
    """
    def main():
        # 在这里写你的代码
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2 + 3 * x + 1
        y.backward()
        print(x.grad)
        pass

    if __name__ == "__main__":
        main()


if RUN_T2:
    """
    T2：梯度累加、清零
    需求：
    1. x = torch.tensor([1.0], requires_grad=True)
    2. 第一轮：y = x * 5；y.backward()；打印 x.grad
    3. 第二轮：y = x * 5；y.backward()；打印 x.grad（观察累加）
    4. x.grad.zero_() 清零
    5. 再跑一次 y = x * 5；y.backward()；打印 x.grad
    """
    def main():
        # 在这里写你的代码
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 5
        y.backward()
        print(x.grad)
        y = x * 5
        y.backward()
        print(x.grad)
        x.grad.zero_()
        y = x * 5
        y.backward()
        print(x.grad)
        pass

    if __name__ == "__main__":
        main()


if RUN_T3:
    """
    T3：向量求导 + gradient 参数
    需求：
    1. w = torch.tensor([1.0, 2.0], requires_grad=True)
    2. y = w * w
    3. y 不是标量，调用 backward 的时候传入 gradient = torch.ones_like(y)
    4. 打印 w.grad
    """
    def main():
        # 在这里写你的代码
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        y = w * w
        y.backward(gradient=torch.ones_like(y))
        print(w.grad)
        pass

    if __name__ == "__main__":
        main()


if RUN_T4:
    """
    T4：no_grad 与 detach
    需求：
    1. a = torch.tensor([3.0], requires_grad=True)
    2. b = a * 10
    3. 用 detach 拿到 b_detach
    4. 在 with torch.no_grad() 里面计算 c = a + 100
    5. 打印 b.requires_grad，b_detach.requires_grad，c.requires_grad
    """
    def main():
        # 在这里写你的代码
        a = torch.tensor([3.0], requires_grad=True)
        b = a * 10
        b_detach = b.detach()
        with torch.no_grad():
            c = a + 100
        print(b.requires_grad)
        print(b_detach.requires_grad)
        print(c.requires_grad)
        pass

    if __name__ == "__main__":
        main()


"""
===== 思考题（不参与编译）
Q1：为什么整数张量设置 requires_grad=True 不会报错，但是无法求导？
因为整数张量在计算图中不支持梯度计算，PyTorch 的自动求导机制主要针对浮点数张量设计，整数类型无法进行微分运算，因此即使设置了 requires_grad=True，也不会报错，但在反向传播时无法计算梯度。
Q2：如果忘记 zero_grad，反复 backward，梯度会出现什么现象？训练的时候会带来什么后果？
梯度会累加，导致梯度值越来越大，影响模型的训练效果。
Q3：什么时候 backward 需要传入 gradient？这个 gradient 的含义是什么？
当 y 不是标量时，需要传入 gradient 参数。这个 gradient 的含义是 y 对于每个元素的梯度。
Q4：detach 和 torch.no_grad() 的区别，举一个适合各自使用的场景。
detach 用于从计算图中分离张量，而 torch.no_grad() 用于在不计算梯度的情况下进行前向传播。适合 detach 的场景是需要将张量从计算图中分离出来进行后续处理，而适合 torch.no_grad() 的场景是推理阶段不需要计算梯度的情况。
Q5：为什么推理阶段一定要套 with torch.no_grad()？除了不计算梯度，还有什么收益？
除了不计算梯度外，with torch.no_grad() 还可以减少内存使用和提高计算速度。
Q6：leaf tensor 的判定规则是什么？为什么中间算子输出的张量默认不会保留 grad？
leaf tensor 是指在计算图中没有父节点的张量。中间算子输出的张量默认不会保留 grad 是为了节省内存和提高计算效率。
"""
