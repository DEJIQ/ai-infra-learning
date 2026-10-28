# AI-Infra Learning Repo
> 学习路线参考：https://caomaolufei.github.io/AIInfraGuide/
目标：初级AI Infra（算子开发 + LLM推理引擎方向）
基线：仅会C++ HelloWorld，本地无NVIDIA GPU，按需租用按量付费T4云GPU
周期：6个月
硬件分工
- 本地：Windows + WSL2 Ubuntu，写C++/Python、CPU单元测试、代码管理、文档
- 云GPU（T4 16G按量付费）：CUDA编译、ncu profiling、TensorRT、端到端推理benchmark，用完关机

## 整体阶段
- M0：前置基础（4周）✅ 进行中
- M1：CUDA算子优化（4周）
- M2：ONNX + TensorRT 推理基础（4周）
- M3：核心项目：极简推理引擎（4周，简历主项目）
- M4：选修：LLM推理进阶 / 分布式训练（二选一，4周）
- M5：面试复盘查漏补缺（4周）

## 仓库规范
1. 每周目录自带README：记录当日任务、踩坑、感悟、自测题
2. Git提交规范：`M0-W1-Dx: xxx`，单日代码单独commit，不攒批量提交
3. 所有CUDA相关代码：配套CPU reference实现，用于数值校验
4. 黑名单（前期不碰）：MLIR/TVM、完整vLLM源码精读、MoE、NCCL底层源码

## 进度看板
- [ ] M0 Week1 C++内存基础
- [ ] M0 Week2 WSL Linux + Python/PyTorch CPU
- [ ] M0 Week3 线性代数直觉 + Transformer Decoder
- [ ] M0 Week4 PyTorch Autograd + 并行基础概念