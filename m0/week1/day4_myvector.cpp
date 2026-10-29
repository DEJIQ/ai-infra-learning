// 切换题目：只保留一个#define，其余注释
//#define T1
#define T2

#include <iostream>
using namespace std;

// 在这里写你的 MyVector 模板类实现
template <typename T>
class MyVector{
    private:
    T* data_;
    size_t size_;
    size_t capacity_;
    public:
    MyVector():data_(nullptr),size_(0),capacity_(0){}
    ~MyVector(){delete[] data_;}
    size_t size() const{
        return size_;
    }
    size_t capacity() const{
        return capacity_;
    }
    T& operator[](size_t idx){
        return *(data_ + idx);
    }
    void push_back(const T& val){
        size_t new_cap = 1;
        if(size_ == capacity_){
            if(capacity_ == 0){new_cap = 1;}
            else{
                new_cap = capacity_ * 2;
            }
            T *new_data_ = new T[new_cap];
            for(size_t i = 0;i<size_;i++){
                new_data_[i] = data_[i];
            }
            delete[] data_;
            data_ = new_data_;
            capacity_ = new_cap;
        }
        data_[size_ ++] = val;
    }
};


#ifdef T1
int main()
{
    MyVector<int> v;
    for(int i = 0; i < 10; i++)
    {
        v.push_back(i);
        cout << "size=" << v.size() << " cap=" << v.capacity() << endl;
    }
    for(size_t i = 0; i < v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
    return 0;
}
#endif

#ifdef T2
int main()
{
    MyVector<int> v;
    v.push_back(100); // 先分配内存，buffer不再是空
    int* p_old = &v[0];
    cout << "old ptr: " << p_old << endl;
    for(int i = 0; i < 10; i++)
    {
        v.push_back(i);
    }
    int* p_new = &v[0];
    cout << "new ptr: " << p_new << endl;
    cout << "ptr equal? " << (p_old == p_new) << endl;
    return 0;
}

#endif
/*
## 思考题完整答案（你第二条已经答对了，帮你补齐第一条，直接抄进笔记）

1. 现在这个版本，执行 `MyVector<int> v2 = v1;` 会发生什么？为什么？

> 
> 编译器会生成默认拷贝构造函数，做**浅拷贝**：直接把 `v1.data_、v1.size_、v1.capacity_` 的值原样复制给 `v2`。
> 结果：`v1.data_` 和 `v2.data_` 指向**同一块堆内存**。等到两个对象生命周期结束、各自调用析构函数时，会对同一块内存连续两次 `delete[]` → **double free，程序崩溃**。

2. 析构函数里 `delete[] data_`，当 `data_ == nullptr` 的时候安全吗？

> 
> C++ 标准明确：`delete[] nullptr` 是合法安全的，不会触发崩溃，什么都不做。你的回答正确。
*/
