# week2_day1｜NumPy 完整汇总笔记

> 面向 AI Infra，不堆冷门函数，重点抓后面 PyTorch、算子、张量布局会复用的底层逻辑

## 一、ndarray 核心本质

普通 Python `list`：

- 元素可以类型混杂
- 内存不连续，存的是一堆对象指针
- 循环慢，不适合大规模数值计算

`np.ndarray`：

- **一块连续堆内存**，所有元素固定为同一个 dtype
- 配套元信息：`data`（起始指针）、`shape`、`strides`（跨步）、`dtype`、`flags`
- `strides`：每一维往下走一步，需要跳过多少字节；这就是为什么切片、reshape 很多时候不用拷贝内存
- `.size`：元素总个数，等于所有 shape 相乘
- `.nbytes`：总字节数 = size × 单个元素字节

```
arr = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
print(arr.strides)
```

## 二、数组创建

1. `np.array(列表)`：从 Python 列表构造

2. ```
   np.arange(start, stop, step)
  ```

   ：左闭右开！

   ```
   stop
   ```

    取不到

   - `np.arange(5,0,-1)` → `[5,4,3,2,1]`

3. ```
   np.zeros(shape, dtype=...)
  ```

    / 

   ```
   np.ones
   ```

    / 

   ```
   np.eye
   ```

   - 默认 dtype 是 `float64`；AI 里面经常手动指定 `float32`

4. `np.linspace(start, end, num)`：等分，首尾都包含，适合生成坐标

## 三、形状操作：reshape /ravel/flatten

- `reshape(new_shape)`：尽可能返回 **视图（view）**，共享底层内存；只有当内存布局不连续的时候，才强制拷贝
- `-1`：自动推导该维度大小，`reshape(-1,6)` 就是固定第二维为 6，自动算行数
- `arr.ravel()`：优先视图；`arr.flatten()`：一定返回拷贝
- 修改视图里面的值，原数组会跟着变；这是高频踩坑点

> 判断是否连续：`arr.flags.contiguous` 如果是 False，reshape 大概率触发拷贝

## 四、索引与切片（重中之重）

两类索引行为完全不一样：

1. 简单切片（冒号）

   ：返回视图，共享内存

   ```
   sub = arr[0:2, 1:]
   ```

2. 花式索引 / 整数数组索引

   ：直接返回拷贝

   ```
   sub = arr[:, [1,3,5]]
   ```

   ✨ 你 T2 用到的这种，就是拷贝，不是视图！

- 多维索引：`arr[i,j]` 等价 `arr[i][j]`，推荐前者
- 切片左闭右开：`arr[1:3]` 只取下标 `1,2`

强制拿到独立拷贝：

```
sub_copy = arr[:, [1,3,5]].copy()
```

## 五、运算、逐元素乘、点积、外积

- `a * b`：element-wise，对应位置相乘（哈达玛积），不是矩阵乘法
- `a @ b`：`np.matmul`，矩阵乘法 / 向量点积
- 如果是两个一维向量：`@` 得到标量（点积）
- 如果一个 `(N,1)`、一个 `(1,M)`：`@` 得到 `(N,M)`，也就是外积
- `a.T`：一维数组转置无效！一维没有行列之分；想要列向量必须 `reshape(-1,1)`

## 六、拼接 hstack /vstack/concatenate

- `np.hstack([a,b])`：横向拼接，增加列
- `np.vstack([a,b])`：纵向拼接，增加行
- **必须把待拼接数组包进列表 `[]`**，不能直接传多个参数
- `np.concatenate`：更通用，指定 `axis`，h/vstack 本质是它的封装

## 七、广播 broadcast（AI 最常用）

规则：从最后一维往前对齐

- 如果维度长度相等：兼容
- 如果其中一个维度长度等于 1：可以自动扩展
- 如果两者都不为 1、还不相等：报错
- ⭐广播不会在内存里复制数组；只是在读取的时候虚拟扩展，节省内存
- 典型场景：给矩阵每一行加一个向量、每一列加一个向量

示例：

```
A = np.array([[1,2,3],[4,5,6]])   # shape (2,3)
b = np.array([10,20,30])          # shape (3,)，按行广播
c = np.array([[1],[2]])           # shape (2,1)，按列广播
```

## 八、视图 vs 拷贝，一张总结

表格

| 操作                | 返回 | 是否共享内存 |
| ------------------- | ---- | ------------ |
| 简单切片            | view | ✅            |
| reshape（连续内存） | view | ✅            |
| reshape（非连续）   | copy | ❌            |
| 花式索引            | copy | ❌            |
| .copy()             | copy | ❌            |

## 九、高频踩坑清单（面试常问）

1. `arange` 左闭右开，很容易少一个数字
2. 默认浮点数是 float64，模型推理一般只用 float32
3. 一维数组没有转置，想做外积一定要 reshape
4. `*` 和 `@` 极易搞混
5. 视图修改会污染原数据，踩坑玄学源头
6. 广播对齐是从后往前，不是从前往后
7. hstack 参数忘记套方括号，直接抛错
8. 切片是视图，但花式索引是拷贝

## 十、思考题标准答案归档

Q1：reshape 会不会一定拷贝内存？什么时候会拷贝？

> 不一定。底层内存连续时，reshape 返回视图，不拷贝；如果数组内存不连续（比如花式索引取出之后），reshape 必须拷贝。

Q2：广播的时候，是真的把数组复制出来一份吗？

> 不会。广播只是计算时改变寻址逻辑，不会开辟新内存复制元素。

Q3：ndarray 的切片返回的是视图还是拷贝？如果想强制拷贝怎么做？

> 普通冒号切片返回视图；整数数组花式索引返回拷贝；强制拷贝调用 `.copy()`。

Q4：`np.array([1,2,3]) @ np.array([[1],[2],[3]])` 结果是什么？

> 标量 `14`，等价 \(1\times 1 + 2\times 2 + 3\times 3\)。

## 十一、和 PyTorch 的对应关系（提前铺垫 day2）

- `np.ndarray` ↔ `torch.Tensor`
- `reshape` ↔ `reshape` / `view`
- `strides` ↔ Tensor 的 stride
- 广播规则几乎一模一样
- PyTorch 区分 CPU / CUDA；NumPy 只有 CPU
- PyTorch 的 `view` 要求内存连续，比 numpy reshape 限制更严




# M0 Week2 Day2：PyTorch Tensor（CPU）
> 前置：已经学会 NumPy，本小节只跑 CPU 张量

## 1. 张量是什么
Tensor = PyTorch 的多维数组，对标 np.ndarray。
额外多出两个核心属性：
- `requires_grad`：是否开启梯度追踪，给自动微分 autograd 使用
- `device`：数据在哪（cpu / cuda:0）

常用属性
- `.shape` / `.size()`：维度
- `.dtype`：数据类型（float32 / float64 / int64 …）
- `.device`：设备
- `.is_contiguous()`：内存是否连续

## 2. 创建张量
```python
torch.tensor([...])      # 从列表构造
torch.ones(shape)
torch.zeros(shape)
torch.arange(start, end, step)
torch.from_numpy(arr)    # 和 numpy 共享内存！
```

## 3. 变形：view /reshape

- ```
  tensor.view(new_shape)
  ```

  - ✅ 返回视图，共享内存
  - ❗ 必须连续，否则报错

- ```
  tensor.reshape(new_shape)
  ```

  - 如果连续：等价 view
  - 如果不连续：自动拷贝，返回新张量

## 4. 索引、切片

规则和 numpy 几乎一致：

- `tensor[:, 0::2]`：步长取列
- `tensor[start:end, ...]`：左闭右开
- 切片出来的结果，很多时候是视图

## 5. 广播（broadcasting）

和 NumPy 广播规则完全一样：

- 从最后一维向前对齐
- 维度相等，或者其中一个为 1，则可以广播
- 如果维度无法对齐，直接报错

典型场景：用一维向量给矩阵每一行加；用形状 `(N,1)` 的向量给每一列加。

## 6. 拷贝、视图、梯度相关

- **视图（view）**：不拷贝内存，多个名字指向同一块数据；修改会互相影响
- `clone()`：深拷贝，全新内存；保留 requires_grad
- `detach()`：视图，脱离 autograd 计算图，不再追踪梯度；共享内存
- `detach().clone()`：独立副本，不带梯度，最常用于取出中间值打印保存

> 重要禁令： 不要原地修改一个 requires_grad=True 的叶子张量的 detach 视图，会抛出运行时异常。

## 7. 和 NumPy 互转

```
# tensor -> numpy（无梯度）
arr = t.detach().numpy()
# numpy -> tensor
t = torch.from_numpy(arr)
```

GPU 张量必须先搬到 CPU，才能转 numpy。

## 8. 高频踩坑清单

1. `view` 遇到不连续张量直接炸，拿不准就用 reshape
2. `from_numpy` 共享内存，改一个两边都变
3. `requires_grad=True` 不能直接 `.numpy()`
4. `detach` 只是解绑计算图，不是拷贝
5. 切片、索引很多返回视图，不经意修改会污染原始张量

## 思考题回顾

- view /reshape 的差异
- from_numpy 的内存共享
- detach /clone/detach ().clone () 使用时机
- 带梯度张量不能直接转 numpy
- GPU 张量无法直接和 numpy 互转



# Week2 Day3 — PyTorch Tensor 常用运算、拼接、规约、索引
> 面向 AI Infra 零基础；所有函数附带入参、返回值

## 1. 张量拼接 stack / cat
### torch.cat
作用：沿着已有维度拼接，不新增维度
```python
torch.cat(tensors, dim=0, *, out=None)
```

- tensors：必选，张量序列；除 dim 外其余维度 shape 一致
- dim：可选，默认 0，沿此维度拉长
- out：可选，指定输出
- 返回：Tensor，尽量视图

### torch.stack

作用：新增一维，把多个同 shape 张量打包

```
torch.stack(tensors, dim=0, *, out=None)
```

- tensors：必选，全部张量 shape 完全相同
- dim：新维度插入位置
- 返回：Tensor，维度 +1

## 2. 规约操作 sum /max

### Tensor.sum

```
sum(dim=None, keepdim=False, dtype=None)
```

- dim：沿哪些维度求和；不传则全局求和
- keepdim：False（默认），去掉被规约维度；True，保留长度为 1
- 返回：规约后张量

### Tensor.max

两种形式：

1. `max()`：全局最大值，返回单个张量

2. ```
   max(dim, keepdim=False)
  ```

   ：沿维度求最大

   - 返回 `(values, indices)`

## 3. 布尔掩码索引

`x > 0 / x < 0` 生成同 shape bool mask

- `x[mask]`：取出满足条件元素，返回一维拷贝
- `x[mask] = val`：原地赋值
- `(cond).sum()`：统计满足条件元素数量

比较算子：`gt, lt, ge, le, eq, ne`

## 4. 乘法区分

- `A * B`：逐元素乘（Hadamard），要求可广播
- `A @ B / torch.matmul`：广义矩阵乘法，自动适配向量、矩阵、批量
- `A.mm(B)`：仅限二维 × 二维
- `A.mv(v)`：二维矩阵 × 一维向量

## 5. 维度重排 permute /transpose

### permute

```
permute(*dims)
```

- *dims：完整维度重排序列，数量等于总维数
- 返回：视图，经常 non-contiguous

### transpose

```
transpose(dim0, dim1)
```

- 只交换两个维度，等价 permute 的特例

## 6. clamp 截断

```
torch.clamp(input, min=None, max=None, *, out=None)
```

- min：下界，None 不限
- max：上界，None 不限
- 返回：截断后的张量；`clamp_` 原地版本

## 7. 重要概念

- **视图（view）**：共享底层内存；只改元信息（shape/stride），不一定连续
- **contiguous**：内存按顺序排布；permute /transpose 出来经常不连续，需要 `.contiguous()`
- **in-place 操作（后缀 _）**：直接改写内存；autograd 中风险很高，容易断计算图
- **keepdim**：广播神器，保留长度为 1 的维度，避免形状不匹配

## 面试思考题要点

1. cat：拉长现有维度；stack：插入新维度
2. keepdim 控制规约之后是否保留 singleton 维度，影响广播
3. 布尔掩码索引返回拷贝，不是视图
4. permute：全维度重排；transpose：仅交换两个
5. in-place 修改会破坏 autograd 计算图
6. clamp：限制数值范围；LLM 常用于 logits 防溢出



# Week2 Day4 — Autograd 自动求导
> AI Infra M0 Week2；所有函数附带功能、入参、返回值、坑点

## 核心总述
Autograd 是 PyTorch 的自动微分引擎。正向运算时构建一张有向无环计算图；调用 `backward()`，沿着链式法则反向传播，算出叶子张量的梯度，存入 `.grad`。

前提：只有浮点张量可以生成有效梯度；整数、布尔张量即便打开 `requires_grad=True`，反向之后 `.grad` 依旧为空。

## 1. torch.tensor
```python
torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
```

- **功能**：新建张量
- **入参**
  - `data`（必选）：列表、嵌套列表、numpy array，初始化数据
  - `dtype`（可选）：指定元素类型
  - `device`（可选）：`cpu` / `cuda`
  - `requires_grad`（可选，bool，默认 False）：是否开启梯度追踪
  - `pin_memory`（可选，bool，默认 False）：锁页内存，GPU 传输加速
- **返回值**：新建 Tensor
- **坑点**：整型数据无法生成有效梯度

## 2. Tensor.requires_grad_(mode)

```
tensor.requires_grad_(mode)
```

- **功能**：原地修改梯度追踪开关
- **入参**
  - `mode`（必选，bool）：True 开启追踪，False 关闭
- **返回值**：自身（原地操作）
- **坑点**：不能抹除已经建好的计算图，只会影响后续运算

## 3. Leaf Tensor（叶子张量）

判定：用户直接创建、不是算子输出，并且 `requires_grad=True`。

- `is_leaf`：只读属性
- 只有叶子张量默认保存 `.grad`
- 算子产出的中间张量：非叶子，反向后默认丢弃梯度，节约内存

## 4. Tensor.backward

```
tensor.backward(gradient=None, retain_graph=False, create_graph=False)
```

- **功能**：启动反向传播，链式求导，梯度累加到叶子节点 `.grad`
- **入参**
  - `gradient`（可选，Tensor，默认 None）：上游梯度 \(\partial \mathcal L/\partial y\)；输出非标量时必须传入
  - `retain_graph`（可选，bool，默认 False）：反向结束销毁计算图；True 保留图，可以多次 backward，内存开销更大
  - `create_graph`（可选，bool，默认 False）：为求导过程再建一张计算图，用于高阶导数
- **返回值**：None
- **坑点**：梯度是累加，不会自动覆盖

## 5. Tensor.grad.zero_

```
tensor.grad.zero_()
```

- **功能**：原地清空该张量的梯度
- **入参**：无
- **返回值**：自身
- **补充**：`optimizer.zero_grad()` 本质遍历参数调用该接口

## 6. torch.ones_like

```
torch.ones_like(input, dtype=None, device=None, requires_grad=False)
```

- **功能**：生成与 input 同 shape，全部填充 1 的张量
- **入参**
  - `input`（必选，Tensor）：参考形状
  - `dtype` / `device` / `requires_grad`：可选
- **返回值**：新张量
- **配套**：`torch.zeros_like`，填充 0

## 7. Tensor.detach

```
tensor.detach()
```

- **功能**：生成脱离计算图的视图，共享底层内存，`requires_grad=False`
- **入参**：无
- **返回值**：新张量（视图）
- **坑点**：如果修改 detach 返回张量的值，会影响原张量正向计算

## 8. torch.no_grad()

```
with torch.no_grad():
    ...
```

- **功能**：上下文管理器；块内所有运算不构建计算图
- **入参**：无
- **返回值**：上下文对象
- **收益**：降低内存占用、加速；推理、验证阶段标配
- **坑点**：仅在 `with` 内部生效

## 标准训练流程

```
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)
x.grad.zero_()
```

## 高频面试要点

1. `detach`：切断单个张量链路，正向依旧建图；适合取出中间结果
2. `no_grad`：关闭一块代码的建图；适合推理
3. 非标量 loss 的 backward，必须传入上游梯度，常用 `ones_like`
4. 如果忘记 `zero_grad`，梯度持续累加，训练发散
5. 如果想要拿到中间张量梯度，需要 `register_hook`



# M0 Week2 Day5 笔记：广播、原地操作、梯度钩子
> 归属路线：AI Infra M0 前置基础 / PyTorch 张量与自动微分
> 配套练习：week2_day5_tensor.py

## 一、广播 Broadcast
### 核心规则
两个张量做逐元素运算时，遵循「靠右对齐」：
1. 将两个张量的维度从最右侧开始一一对应对齐；维度不够的，在左边自动补维度 1
2. 对应位置维度，满足下面二者之一就可以广播：
   - 维度相等
   - 其中一个维度等于 1
3. 如果某一维两者都不为 1，并且数值不一样 → 广播失败，直接报错
4. 广播之后，维度为 1 的轴会自动复制扩张，生成同形状张量再逐元素计算

### 函数：torch.broadcast_tensors
```python
torch.broadcast_tensors(*tensors)
```

- **功能**：返回广播对齐之后的一组新张量
- **入参**：若干张量
- **返回值**：元组，全部张量扩张到同一个 shape

示例

```
a = torch.tensor([[10.0], [20.0], [30.0]])   # [3, 1]
b = torch.tensor([[1.0, 2.0, 3.0, 4.0]])     # [1, 4]
aa, bb = torch.broadcast_tensors(a, b)
# aa.shape = [3, 4]
# bb.shape = [3, 4]
```

### 易错点

广播只是逻辑上的扩张，不会立刻复制全部数据；PyTorch 在底层按需读取，节省内存。

## 二、原地操作 In-place

### 识别标记

PyTorch 约定：函数名以下划线 `_` 结尾代表原地操作，例如：`add_`、`mul_`、`fill_`。
原地操作直接修改原张量，不分配新内存。

### 关键限制

如果张量是叶子节点，并且 `requires_grad=True`，绝大多数原地操作会抛出 `RuntimeError`。

> 
> 原因：自动微分计算图依赖叶子节点原始数值；原地覆盖原值之后，反向传播无法还原，导数计算失效。

- ✅：`requires_grad=False` 的张量，可以安全原地修改
- ❌：带梯度追踪的叶子张量，禁止原地改写

```
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x.add_(1.0)   # 报错
y = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
y.add_(5.0)   # 正常执行
```

## 三、梯度钩子 register_hook

### 函数

```
handle = tensor.register_hook(hook_fn)
```

- **功能**：在张量反向传播经过它的时候，挂载回调函数，拦截流过该张量的梯度
- **入参**
  - `hook_fn`：回调函数，签名 `def hook_fn(grad): ...`
    - `grad`：上游传递到当前张量的梯度 \(\partial L/\partial y\)，不是局部导数
    - 如果返回张量，则这个新张量会替换原有梯度，继续向上流动；可以用来缩放、裁剪梯度
- **返回值**：`RemovableHandle` 句柄

### 使用规范

- 使用完毕，必须调用 `handle.remove()` 卸载钩子
- 如果忘记移除：钩子会一直存活；循环多次 backward 时反复触发，篡改梯度、打印垃圾信息，甚至内存泄漏
- 钩子只在反向传播的时候触发，前向不会运行

### 示例

```
x = torch.tensor([3.0], requires_grad=True)
y = x ** 3
z = 2 * y

def hook_fn(grad):
    print("钩子收到梯度：", grad)
    return grad

handle = y.register_hook(hook_fn)
z.backward()
handle.remove()
```

执行顺序：

1. 前向算出 y、z，构建计算图
2. `z.backward()`，默认起点上游梯度为 1
3. 梯度流到达 y，触发钩子
4. 梯度继续流向叶子节点 x，存入 `x.grad`

## 四、梯度流向补充

调用 `tensor.backward()`，不传参数时：默认给这个张量传入上游梯度 `tensor([1.0])`，作为反向传播的起点。

## 五、思考题标准答案

Q1：shape `[2, 3, 4]` 和 `[3, 1]` 是否可以广播？

> 
> 可以。靠右对齐：`[2, 3, 4]` 与 `[1, 3, 1]`；每一组维度满足广播规则；输出 shape `[2, 3, 4]`。

Q2：为什么 `requires_grad=True` 的叶子张量大多不能原地操作？

> 
> 反向传播依赖叶子节点原始值；原地修改会覆盖原值，导致无法正确求导，PyTorch 主动拦截。

Q3：钩子拿到的梯度，是局部导数，还是上游传进来的梯度？

> 
> 上游传进来的梯度 \(\partial L/\partial y\)。

Q4：循环 backward，如果忘记 remove 钩子会怎样？

> 
> 钩子持续挂载，每次反向都会触发，造成梯度被反复修改、日志混乱；不断堆积句柄，引发内存泄漏。

## 六、高频坑汇总

1. 不要拿浮点数张量用 `torch.eq` 判断相等，浮点精度会翻车
2. 原地操作、自动求导不要混用
3. `register_hook` 只捕获流过该张量的梯度，不能直接拿到局部导数
4. 广播靠右对齐，不是靠左对齐，很容易写反维度预期



# M0 Week2 Day6 笔记：张量常用算子
> 归属路线：AI Infra M0 前置基础 / PyTorch 张量算子
> 配套练习：week2_day6_tensor.py

## 一、逐元素算子 Element-wise
特点：位置一一对应运算，支持广播，输出 shape = 广播后 shape

### torch.add
```python
torch.add(input, other, *, alpha=1, out=None)
```

- 功能：`input + alpha * other`，等价运算符 `+`
- input：主张量
- other：加数，标量 / 张量，支持广播
- alpha：other 的缩放系数
- out：可选，存放输出
- 返回：结果张量

### torch.mul

```
torch.mul(input, other, *, out=None)
```

- 功能：逐元素相乘，等价运算符 `*`
- ⚠️：不是矩阵乘法！矩阵乘用 `torch.matmul` / `@`
- 返回：逐元素乘积张量

### torch.pow

```
torch.pow(input, exponent, *, out=None)
```

- 功能：逐元素求幂
- input：底数
- exponent：指数，可以标量 / 张量，广播
- 返回：每个位置 `base ** exponent`

### torch.exp / torch.log

```
torch.exp(input, *, out=None)
torch.log(input, *, out=None)
```

- `exp`：逐元素自然指数 \(e^{x}\)
- `log`：逐元素自然对数 \(\ln(x)\)；输入不能含负数
- 返回：同 shape 张量

## 二、归约算子 Reduction

特点：沿着指定维度合并元素，可以压缩维度；`keepdim` 控制是否保留长度为 1 的维度

### torch.sum

```
torch.sum(input, dim=None, keepdim=False, *, dtype=None, out=None)
```

- dim：目标维度；`dim=None`：全部元素求和得到标量
- keepdim：True → 保留该维（长度为 1）；False → 删除
- dtype：指定输出类型，防止整数溢出

### torch.mean

```
torch.mean(input, dim=None, keepdim=False, *, dtype=None, out=None)
```

- 参数和 sum 一致；一般要求输入浮点

### torch.max（两个版本）

1. 归约版本（必须指定 dim）
```
torch.max(input, dim, keepdim=False, *, out=None)
```

返回元组 `(values, indices)`：最大值、最大值下标
2. 逐元素比较版本
```
torch.max(input, other, *, out=None)
```

### torch.argmax

```
torch.argmax(input, dim=None, keepdim=False, *, out=None)
```

- 只返回最大值下标，long 张量
- torch.min/torch.argmin 对称

## 三、比较算子 + torch.where

### 比较系列（全部逐元素、支持广播）

- `torch.gt(a,b)` → `a > b`
- `torch.lt(a,b)` → `a < b`
- `torch.ge(a,b)` → `a >= b`
- `torch.le(a,b)` → `a <= b`
- `torch.eq(a,b)` → `a == b`（浮点数慎用）
- `torch.ne(a,b)` → `a != b`

返回 bool 张量，也就是 mask（掩码），常用于 Attention、padding。

### torch.where

```
torch.where(condition, x, y, *, out=None)
```

- condition：bool 掩码
- 规则：条件 True → 取 x；False → 取 y
- x、y、condition 三者先尝试广播；广播失败报错
- 返回广播合并后的张量

## 四、torch.gather

```
torch.gather(input, dim, index, *, out=None, sparse_grad=False)
```

- 功能：沿着 dim，按 index 从 input 捞取元素
- input：源张量（读）
- dim：采集的维度
- index：long 类型张量；shape 需要兼容
- 返回：和 index 同 shape 的采集结果

理解公式：
`out[..., i, ...] = input[..., index[..., i, ...], ...]`
index 放在 dim 这一维。

配套反向算子：`torch.scatter`，按索引往张量里面写入数据。

## 五、思考题标准答案

Q1：torch.mul 是逐元素乘；矩阵乘法用哪几个函数？

> 
> `torch.mm`（仅限二维）；`torch.matmul` / `@`（通用，支持批量高维矩阵乘）

Q2：torch.sum (x, dim=None) 的效果是什么？

> 
> 将张量所有元素求和，返回一个标量张量。

Q3：torch.where 的三个参数如果形状不一样，会发生什么？

> 
> 先尝试广播；满足广播规则就正常运行，广播冲突才抛出异常。

Q4：gather 的 index 如果不是 long 类型，会怎么样？

> 
> PyTorch 直接抛出类型错误，索引张量强制为 long（int64）。

## 六、高频坑汇总

1. `*` / `mul`：逐元素；矩阵乘法必须用 matmul / @
2. `max` 归约版本必须传 dim，不能省略
3. `gather / scatter` 的 index 必须是 long
4. `where` 依赖广播，不是要求形状严格相等
5. `keepdim` 很关键：丢掉维度之后后续广播容易翻车