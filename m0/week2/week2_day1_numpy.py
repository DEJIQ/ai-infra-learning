# week2_day1_numpy.py
import numpy as np

# ========== 切换题目：仅保留一个 True，其余 False ==========
RUN_T1 = False
RUN_T2 = True
RUN_T3 = False
RUN_T4 = False
# ==========================================================


if RUN_T1:
    """
    T1：数组构造与属性
    需求：
    1. 创建一维数组：[5, 4, 3, 2, 1]
    2. 创建 shape=(3,4) 的全零数组，dtype=np.float32
    3. 打印这两个数组，以及各自的 shape、dtype、size
    """
    def main():
        # 在这里写你的代码
        arry1 = np.arange(5,0,-1)
        arry2 = np.zeros((3,4),dtype = np.float32)
        arry2_1 = arry2.reshape(3,4)
        print("arry1:\n",arry1)
        print("arry1 shape:",arry1.shape,"arry1 dtype:",arry1.dtype,"array1 size:",arry1.size)
        print("arry2:\n",arry2)
        print("arry2 shape:",arry2.shape,"arry2 dtype:",arry2.dtype,"array2 size:",arry2.size)
        pass

    if __name__ == "__main__":
        main()


if RUN_T2:
    """
    T2：reshape + 切片提取
    需求：
    1. 使用 np.arange(24) 生成 0~23 的一维数组
    2. reshape 成 (4,6) 的二维矩阵
    3. 提取：所有行，第 1、3、5 列，打印结果
    4. 再提取第 1~3 行，前两列，打印结果
    """
    def main():
        # 在这里写你的代码
        arry3 = np.arange(24)
        arry4 = arry3.reshape(-1,6)
        print(arry4[:,[1,3,5]])
        print(arry4[1:3,:2])
        pass

    if __name__ == "__main__":
        main()


if RUN_T3:
    """
    T3：广播计算
    需求：
    给定：
        A = np.array([[1,2,3],[4,5,6]])
        b = np.array([10, 20, 30])
    计算 A + b，打印结果；
    再构造一个向量 c，使得 c + A 可以广播，并且每一列都加上 c
    """
    def main():
        # 在这里写你的代码
        A = np.array([[1,2,3],[4,5,6]])
        b = np.array([10, 20, 30])
        print(A+b)
        c = np.array([[1],[2]])
        print(c+A)
        pass

    if __name__ == "__main__":
        main()


if RUN_T4:
    """
    T4：拼接与点积
    需求：
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    1. 使用 hstack 拼接 a,b 得到 [1,2,3,4,5,6]
    2. 计算 a 和 b 的点积并打印
    3. 将 a 变成列向量，b 变成行向量，矩阵相乘
    """
    def main():
        # 在这里写你的代码
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.hstack([a,b])
        print(c)
        print(a @ b)
        a = a.reshape(-1,1)
        b = b.reshape(1,-1)
        print(a @ b)
        pass

    if __name__ == "__main__":
        main()


"""
===== 思考题（只看，不跑）
Q1：reshape 会不会一定拷贝内存？什么时候会拷贝？
不会，只是改变看待这块内存的方式，只是改变了逻辑形状
Q2：广播的时候，是真的把数组复制出来一份吗？
不是
Q3：ndarray 的切片返回的是视图还是拷贝？如果想强制拷贝怎么做？
不知道
Q4：np.array([1,2,3]) @ np.array([[1],[2],[3]]) 结果是什么？
14
"""
