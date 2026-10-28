#include <iostream>
int main()
{
    // new 在堆上分配单个int
    int* p = new int;
    *p = 100;
    std::cout << "*p = " << *p << std::endl;

    // new[] 分配int数组
    int* arr = new int[4]{1,2,3,4};
    for(int i = 0; i < 4; i++)
    {
        std::cout << arr[i] << " "<<std::endl;
    }
    std::cout << "\n";

    // 释放单个对象
    delete p;
    p = nullptr; 

    // 释放数组，带[]
    delete arr;
    arr = nullptr;
    return 0;
}
