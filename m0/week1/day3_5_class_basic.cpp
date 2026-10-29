// 切换题目：只保留一个#define，其余注释
//#define T1
//#define T2
#define T3

#include <iostream>
using namespace std;

// 你的代码写在这里
class Box{
    public:
    int w;
    Box(){
        w = 0;
    }
};

class HeapBox{
    public:
    int* buf;
    HeapBox(){
        buf = new int[5];
    }
    ~HeapBox(){
        delete[] buf;
    }
};

class ReadBox{
    private:
    int num = 10;
    public:
    int get_num() const{
        return num;
    }
};

#ifdef T1
// T1：基础类 + 构造初始化
int main()
{
    Box b;
    cout << b.w << endl; // 预期输出0
    return 0;
}
#endif

#ifdef T2
// T2：构造申请堆内存，析构释放
int main()
{
    HeapBox h;
    h.buf[0] = 99;
    cout << h.buf[0] << endl;
    return 0;
}
#endif

#ifdef T3
// T3：const成员函数练习
int main()
{
    ReadBox r;
    cout << r.get_num() << endl;
    return 0;
}
#endif

/*
思考题（写完填笔记）

1. 构造函数是手动调用吗？什么时候执行？
不需要手动调用；创建（实例化）对象的一瞬间自动执行
2. 析构函数的核心作用是什么？
对象销毁时自动执行，释放该对象在构造阶段申请的堆资源，避免内存泄漏（不止是构造申请的，也包含对象生命周期内申请的堆内存）。
3. 成员函数后面加 const 代表什么？
承诺：这个成员函数**不会修改类内任何成员变量**；编译器会检查，一旦尝试修改成员直接报错；只读接口优先加 const。
*/
