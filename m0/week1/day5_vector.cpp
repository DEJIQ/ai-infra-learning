// 切换题目：只保留一个#define，其余注释
#define T1
//#define T2
//#define T3

#include <iostream>
#include <vector>   // vector必须包含这个头文件！
using namespace std;

#ifdef T1
int main()
{
    vector<int> vec{5,0,6,7};
    int *ptr = vec.data();
    for(int i = 0;i<vec.size();i++){
        cout<<ptr[i]<<endl;
    }
    vec.push_back(5);
    for(int i = 0;i<vec.size();i++){
        cout<<vec[i]<<endl;
    }
    return 0;
}
#endif

#ifdef T2
void print_buffer(const int* buf, int n)
{
    for(int i = 0;i<n;i++){
        cout<<*(buf + i)<<endl;
    }
}

int main()
{
    vector<int> vec(10);
    for(int i = 0;i<vec.size();i++){
        vec[i] = i + 1;
    }
    print_buffer(vec.data(),6);
    return 0;
}
#endif

#ifdef T3
void square_inplace(vector<int>& vec)
{
    for(int i = 0;i<vec.size();i++){
        vec[i] *= vec[i];
    }
}

int main()
{
    vector<int> v{1,2,3,4};
    square_inplace(v);
    for(int i = 0;i<v.size();i++){
        cout<<v[i]<<" ";
    }
    return 0;
}
#endif
