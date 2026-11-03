//#define T1
#define T2
//#define T3
//#define T4

#include <iostream>
#include <cstddef>
using namespace std;

#ifdef T1
// ========== OJ1 day6_align1: S1 S2 S3 基础对齐 ==========
// 需求：打印sizeof、alignof、每个成员offset，严格遵循输出格式
struct S1 { char c; int  i; };
struct S2 { int i; char c; };
struct S3 { char a; short b; char c; };

int main()
{
    cout<<"S1: sizeof="<<sizeof(S1)<<", alignof="<<alignof(S1)<<endl;
    cout<<"  c offset: "<<offsetof(S1,c)<<endl;
    cout<<"  i offset: "<<offsetof(S1,i)<<endl;
    cout<<endl;
    cout<<"S2: sizeof="<<sizeof(S2)<<", alignof="<<alignof(S2)<<endl;
    cout<<"  i offset: "<<offsetof(S2,i)<<endl;
    cout<<"  c offset: "<<offsetof(S2,c)<<endl;
    cout<<endl;
    cout<<"S3: sizeof="<<sizeof(S3)<<", alignof="<<alignof(S3)<<endl;
    cout<<"  a offset: "<<offsetof(S3,a)<<endl;
    cout<<"  b offset: "<<offsetof(S3,b)<<endl;
    cout<<"  c offset: "<<offsetof(S3,c)<<endl;
    return 0;
}
#endif

#ifdef T2
// ========== OJ2 day6_align2: A B C D packed对比 ==========
// 需求：对比成员顺序影响、packed关闭对齐效果
struct A { char x; long long y; };
struct B { long long y; char x; };
struct __attribute__((packed)) C { char x; long long y; };
struct D { char a; char b; int c; short d; };

int main()
{
    cout<<"A: sizeof="<<sizeof(A)<<", alignof="<<alignof(A)<<endl;
    cout<<"  x offset: "<<offsetof(A,x)<<endl;
    cout<<"  y offset: "<<offsetof(A,y)<<endl;
    cout<<endl;
    cout<<"B: sizeof="<<sizeof(B)<<", alignof="<<alignof(B)<<endl;
    cout<<"  y offset: "<<offsetof(B,y)<<endl;
    cout<<"  x offset: "<<offsetof(B,x)<<endl;
    cout<<endl;
    cout<<"C: sizeof="<<sizeof(C)<<", alignof="<<alignof(C)<<endl;
    cout<<"  x offset: "<<offsetof(C,x)<<endl;
    cout<<"  y offset: "<<offsetof(C,y)<<endl;
    cout<<endl;
    cout<<"D: sizeof="<<sizeof(D)<<", alignof="<<alignof(D)<<endl;
    cout<<"  a offset: "<<offsetof(D,a)<<endl;
    cout<<"  b offset: "<<offsetof(D,b)<<endl;
    cout<<"  c offset: "<<offsetof(D,c)<<endl;
    cout<<"  d offset: "<<offsetof(D,d)<<endl;
    return 0;
}
#endif

#ifdef T3
// ========== OJ3 day6_align3: 嵌套结构体 ==========
// 需求：理解嵌套结构体作为成员时的对齐传递规则（Tensor元结构体高频场景）
struct Inner {
    char ch;
    int num;
};
struct Outer {
    char tag;
    Inner in;
    short val;
};

int main()
{
    cout<<"Inner: sizeof="<<sizeof(Inner)<<", alignof="<<alignof(Inner)<<endl;
    cout<<"  ch offset: "<<offsetof(Inner,ch)<<endl;
    cout<<"  num offset: "<<offsetof(Inner,num)<<endl;
    cout<<endl;
    cout<<"Outer: sizeof="<<sizeof(Outer)<<", alignof="<<alignof(Outer)<<endl;
    cout<<"  tag offset: "<<offsetof(Outer,tag)<<endl;
    cout<<"  in offset: "<<offsetof(Outer,in)<<endl;
    cout<<"  val offset: "<<offsetof(Outer,val)<<endl;
    return 0;
}
#endif

#ifdef T4
// ========== OJ4 day6_align4: 数组对齐（选做） ==========
// 需求：观察结构体数组的连续存储特性，数组元素之间不会额外插入padding
struct E {
    char c;
    int i;
};

int main()
{
    E arr[3];
    cout<<"E: sizeof="<<sizeof(E)<<", alignof="<<alignof(E)<<endl;
    cout<<"arr[0] offset in array: 0"<<endl;
    cout<<"arr[1] offset in array: "<<sizeof(E)<<endl;
    cout<<"arr[2] offset in array: "<<2*sizeof(E)<<endl;
    cout<<"sizeof(arr) = "<<sizeof(arr)<<endl;
    return 0;
}
#endif

/*
===== Day6 全部思考题（写完代码在这里作答） =====
1. S1、S2成员相同顺序不同，sizeof相同，是否代表所有结构体调换成员顺序size都不变？举反例。
2. S3末尾为什么要补1字节padding？
3. A 和 B 的sizeof相等吗？对比OJ1的S1/S2，总结规律。
4. C为什么sizeof=9、alignof=1？packed结构体有什么性能/硬件限制（CUDA视角）。
5. 画出D完整内存布局，标出所有padding位置。
6. 嵌套结构体的对齐值怎么算？Inner作为Outer的一个成员，它的对齐要求是谁决定的？
7. 如果把Outer里面顺序改成 Inner in; char tag; short val;，总大小会不会变？
8. 数组里每个结构体之间会不会额外插入padding？为什么Tensor一维连续存储，只需要保证数组首地址对齐？
9. 为什么CUDA的Global Memory、Shared Memory访问强烈推荐16B/32B/128B对齐？
10. SIMD（CPU）/GPU向量load指令，对不对齐有硬性要求吗？非对齐访问会发生什么？
11. 什么时候我们会主动用packed？什么时候绝对不能用packed？
12. 能不能用offsetof读取带虚函数的类成员偏移？为什么？
*/
