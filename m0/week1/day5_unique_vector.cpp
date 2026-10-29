#define T1

#include <iostream>
#include <memory>
#include <vector>

using namespace std;

template <typename T>
class UniqueVector{
    private:
    unique_ptr<T[]> data_;
    size_t size_;
    size_t capacity_;
    public:
    UniqueVector():data_(nullptr),size_(0),capacity_(0){}
    size_t size() const {return size_;}
    size_t capacity() const {return capacity_;}
    T& operator[] (size_t idx){
        return data_[idx];
    }
    void push_back(const T& val){
        size_t new_cap = 1;
        if(size_ == capacity_){
            if(capacity_ == 0){new_cap = 1;}
            else{
                new_cap = capacity_ * 2;
            }
            unique_ptr<T[]> new_data_ = make_unique<T[]>(new_cap);
            for(size_t i =0;i<size_;i++){
                new_data_[i] = data_[i];
            }
            data_ = std::move(new_data_);
            capacity_ = new_cap;
        }
        data_[size_ ++] = val;
    }
};

#ifdef T1
int main()
{
    UniqueVector<int> v;
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
    UniqueVector<int> v1;
    v1.push_back(100);
    UniqueVector<int> v2 = v1; // 预期：编译报错
    return 0;
}
#endif

/*
1. 把 data_换成 unique_ptr<T []> 之后，还需要手动写析构函数吗？为什么？
不需要，自动析构，释放内存
2. unique_ptr 为什么不能直接拷贝？std::move 做了什么？
禁止浅拷贝，unique_ptr删掉了默认构造函数中的拷贝函数
不用等到double free再内存崩溃，std::move移交unique_ptr所有权
3. 什么时候我们会用`.get()`拿到底层裸指针？有什么风险？
在算子读取一些buf时要用到，风险不知道
*/