#include <iostream>
#include <cstddef>  // offsetof
using namespace std;

struct S1 {
    char c;
    int  i;
};
struct S2 {
    int i;
    char c;
};
struct S3 {
    char a;
    short b;
    char c;
};

int main()
{
    // 在这里写打印逻辑
    // 输出格式：
    // S1: sizeof=8, alignof=4
    //   c offset: 0
    //   i offset: 4
    cout<<"S1: sizeof="<<sizeof(S1)<<", alignof="<<alignof(S1)<<endl;
    cout<<"  c offset:"<<offsetof(S1,c)<<endl;
    cout<<"  i offset:"<<offsetof(S1,i)<<endl;
    cout<<endl;
    cout<<"S2: sizeof="<<sizeof(S2)<<", alignof="<<alignof(S2)<<endl;
    cout<<"  i offset:"<<offsetof(S2,i)<<endl;
    cout<<"  c offset:"<<offsetof(S2,c)<<endl;
    cout<<endl;
    cout<<"S3: sizeof="<<sizeof(S3)<<", alignof="<<alignof(S3)<<endl;
    cout<<"  c offset:"<<offsetof(S3,a)<<endl;
    cout<<"  s offset:"<<offsetof(S3,b)<<endl;
    cout<<"  c offset:"<<offsetof(S3,c)<<endl;

    return 0;
}

/*
思考题（代码跑出来之后回答）

1. S1 和 S2 成员完全一样，顺序不同，sizeof 一样吗？为什么？
一样，只是padding的位置不同，S1的padding位置是1，2，3，s2的padding位置是5，6，7
2. S3 的总大小是多少？画出内存布局，标出 padding 位置
总大小6B，内存布局依次是a 1B,Padding 1B,b 2B,c 1B,padding 1B
3. 为什么 CUDA 算子、Tensor 底层都优先保证内存对齐？
内存对齐才能更快的访问内存
你回答的方向对，这里补全 AI Infra/CUDA 视角完整版（面试常用）
CPU/GPU 的硬件**宽位加载**：硬件一次读 4/8/16/32 字节（cacheline、GPU 的 memory transaction）；不对齐会触发**两次内存访问 + 拼接数据**，延迟翻倍；
CUDA 硬件限制：global memory、shared memory 访问，非对齐访问会产生额外开销，部分旧架构不支持非对齐访问；
Tensor 底层（比如 TensorRT、cuBLAS）：向量化指令（SIMD/PTX 向量 load）**强制要求对齐**，对齐才能用上向量化加速；
对齐也利于 cache 预取，减少 cache miss。
*/