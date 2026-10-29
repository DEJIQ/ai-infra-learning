// 切换题目：只保留一个#define，其余注释
//#define T1
//#define T2
#define T3

#include <iostream>
#include <vector>
using namespace std;

#ifdef T1
int main()
{
    vector<int> v;
    for(int i = 0; i < 10; i++)
    {
        v.push_back(i);
        cout << "size=" << v.size() << " cap=" << v.capacity() << endl;
    }
    return 0;
}
#endif

#ifdef T2
int main()
{
    vector<int> v;
    v.reserve(10);
    for(int i = 0; i < 10; i++)
    {
        v.push_back(i);
        cout << "size=" << v.size() << " cap=" << v.capacity() << endl;
    }
    return 0;
}
#endif

#ifdef T3
int main()
{
    vector<int> v;
    int *p_old = v.data();
    cout << "扩容前 p_old = " << p_old << endl;

    for(int i = 0; i < 10; i++)
    {
        v.push_back(i);
    }

    int *p_new = v.data();
    cout << "扩容后 p_new = " << p_new << endl;
    cout << "p_old == p_new ? " << (p_old == p_new) << endl;
    // 不再去 *p_old，避免崩溃
    return 0;
}
#endif
