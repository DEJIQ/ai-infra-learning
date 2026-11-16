# week2_day2_tensor.py
import torch
import numpy as np

# ========== 切换题目：仅保留一个 True，其余 False ==========
RUN_T1 = True
RUN_T2 = False
RUN_T3 = False
RUN_T4 = False
# ==========================================================


if RUN_T1:
    """
    T1：张量创建、属性查看
    需求：
    1. 使用 torch.tensor 创建一维张量 [9,8,7,6,5]，dtype=torch.float32
    2. 创建 shape=(2,5) 的全 1 张量
    3. 打印两个张量，以及各自的 shape、dtype、device
    4. 把第一个张量转成 numpy array，打印出来
    """
    def main():
        # 在这里写你的代码
        tensor_1 = torch.tensor([9,8,7,6,5],dtype = torch.float32)
        tensor_2 = torch.ones((2,5))
        print("tensor1:",tensor_1)
        print("tensor1 shape:",tensor_1.shape,"tensor1 dtype:",tensor_1.dtype,"tensor1 device:",tensor_1.device)
        print(tensor_2)
        print("tensor2 shape:",tensor_2.shape,"tensor2 dtype:",tensor_2.dtype,"tensor2 device:",tensor_2.device)
        print("tensor1 as numpy:",tensor_1.numpy())
        pass

    if __name__ == "__main__":
        main()


if RUN_T2:
    """
    T2：view / reshape + 切片
    需求：
    1. torch.arange(0, 32) 生成一维张量
    2. reshape 成 (4, 8)
    3. 取出所有行，下标为 0,2,4,6 的列，打印
    4. 取出第 2～4 行，前 3 列，打印
    5. 使用 view，把这个矩阵变成 (8,4)
    """
    def main():
        # 在这里写你的代码
        tensor_3 = torch.arange(0,32)
        print(tensor_3)
        tensor_3 = tensor_3.reshape((4,8))
        print(tensor_3)
        print(tensor_3[:,:8:2])
        print(tensor_3[1:4,:3])
        tensor_3 = tensor_3.view((8,4))
        print(tensor_3)
        pass

    if __name__ == "__main__":
        main()


if RUN_T3:
    """
    T3：广播与张量运算
    给定：
        M = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float32)
        v = torch.tensor([0.1, 0.2, 0.3])
    需求：
    1. M + v，打印结果
    2. 构造向量 u，实现给 M 的每一列加上 u
    3. 计算 M 和 M 的矩阵乘法（@）
    """
    def main():
        # 在这里写你的代码
        M = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float32)
        v = torch.tensor([0.1, 0.2, 0.3])
        print(M + v)
        u = torch.tensor([[1],[2],[3]])
        print(M + u)
        print(M @ M)
        pass

    if __name__ == "__main__":
        main()


if RUN_T4:
    """
    T4：clone / detach 区分练习
    需求：
    1. 创建 x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    2. y = x.detach()
    3. z = x.clone()
    4. 修改 y[0] = 999，打印 x, y, z，观察 x 是否跟着变
    5. 修改 z[1] = 888，打印 x, y, z，观察 x 是否跟着变
    """
    def main():
        # 在这里写你的代码
        x = torch.tensor([1.0,2.0,3.0],requires_grad=True)
        y = x.detach().clone()
        z = x.clone()
        y[0] = 999
        print("x:",x)
        print("y:",y)
        print("z:",z)
        z[1] = 888
        print("x:",x)
        print("y:",y)
        print("z:",z)
        pass

    if __name__ == "__main__":
        main()


"""
===== 思考题（放在文件末尾，不参与编译）
Q1：view 和 reshape 的区别？什么时候 view 会报错？
view 和 reshape 的区别在于：
1. view 只能在张量是连续的情况下使用，而 reshape 可以处理非连续的张量。
2. view 返回的是原始张量的视图，而 reshape 可能返回一个新的张量。
3. 当张量的内存布局不连续时，使用 view 会报错，而 reshape 会尝试返回一个新的张量。
Q2：torch.from_numpy 得到的张量，如果修改张量，会不会影响原来的 numpy 数组？为什么？
会影响。因为 torch.from_numpy 创建的张量与原 numpy 数组共享内存。
Q3：detach 和 clone 的区别；什么时候要用 detach().clone()？
detach 返回的是一个不参与梯度计算的视图，而 clone 返回的是一个全新的张量。当需要
在不参与梯度计算的情况下复制张量时，可以使用 detach().clone()。
Q4：requires_grad=True 的张量，为什么不能直接调用 .numpy()？
requires_grad=True 的张量在计算图中，不能直接调用 .numpy()，因为这会破坏计算图的结构。
Q5：如果一个张量在 GPU 上，能不能直接 from_numpy / .numpy()？
不能。from_numpy 和 .numpy() 只能在 CPU 上操作，如果张量在 GPU 上，需要先将其移动到 CPU 上再进行操作。
"""
