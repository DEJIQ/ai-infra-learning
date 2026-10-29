// 切换题目：只保留一个#define，其余注释
//#define T1
//#define T2
#define T3

#include <iostream>
using namespace std;

#ifdef T1
int main()
{
    int a = 100;
    int &ref = a;
    cout<<a<<" "<<ref<<endl;
    ref = 200;
    cout<<a<<" "<<ref<<endl;
    int b = 999;
    ref = b;
    cout<<b<<" "<<ref<<endl;
    return 0;
}
#endif

#ifdef T2

void swap_val(int a,int b){
    a += b;
    b = a - b;
    a = a - b;
}

void swap_str(int *a,int *b){
    *a += *b;
    *b = *a -*b;
    *a -= *b;
}

void swap_ref(int &a,int &b){
    a += b;
    b = a - b;
    a = a - b;
}

int main()
{
    int x = 1,y = 2;
    cout<<"原值："<<x<<" "<<y<<endl;
    swap_val(x,y);
    cout<<"值传递swap之后："<<x<<" "<<y<<endl;
    swap_str(&x,&y);
    cout<<"指针swap之后："<<x<<" "<<y<<endl;
    swap_ref(x,y);
    cout<<"引用swap之后："<<x<<" "<<y<<endl;
    return 0;
}
#endif

#ifdef T3
void add(const int* a,const int* b,int &res){
    res = *a + *b;
}

int main()
{
    int *arr = new int[]{3,4};
    int ans;
    add(arr,arr+1,ans);
    //add(3,4,ans);
    cout<<ans<<endl;
    delete[] arr;
    return 0;
}
#endif
