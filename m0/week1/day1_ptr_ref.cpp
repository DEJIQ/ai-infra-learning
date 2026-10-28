#include <iostream>
int main()
{
    int a = 10;
    int *p = &a;   // 指针：存放变量的内存地址，可以独立存在
    int &ref = a;  // 引用：a的别名，从定义起就必须绑定一个有效变量

    std::cout << "a = " << a << std::endl;
    std::cout << "&a = " << &a << std::endl;

    std::cout << "*p = " << *p << std::endl;
    std::cout << "p = " << p << std::endl;

    std::cout << "ref = " << ref << std::endl;
    std::cout << "&ref = " << &ref << std::endl;

    ref = 20;
    std::cout << "修改ref之后 a = " << a << std::endl;
    return 0;
}
