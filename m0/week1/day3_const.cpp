// 切换题目：取消注释对应那一行，其他注释掉
//#define T1
//#define T2
#define T3

#include <iostream>
using namespace std;

#ifdef T1
// ========== T1 代码写在这个区域 ==========
int main()
{
    int x = 10,y=20;
    const int* p = &x;
    cout<<"指针p指向的变量值不能改变，此时指向的x值为："<<*p<<endl;
    p = &y;
    cout<<*p<<endl;
    //*p = 999;
    cout<<*p<<endl;
    return 0;
}
#endif

#ifdef T2
// ========== T2 代码写在这个区域 ==========
int main()
{
    int num = 50;
    int* const ptr = &num;
    *ptr = 60;
    cout<<*ptr<<endl;
    /*
    int x = 10;
    ptr = &x;
    cout<<*ptr<<endl;
    return 0;
     */
    
}
#endif

#ifdef T3
// ========== T3 代码写在这个区域 ==========

void print_arr(const int* arr,int n){
    for(int i = 0;i<n;i++){
        cout<<*(arr + i)<<endl;
    }
}

int main()
{
    const int* arr = new int[]{1,3,5,7};
    print_arr(arr,4);
    delete[] arr;
    return 0;
}
#endif
