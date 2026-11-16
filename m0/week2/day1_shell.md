# T1

 ```shell
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ touch t1.txt
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ echo -e "apple\nbanana\norange">>t1.txt
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ cat t1.txt
 apple
 banana
 orange
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ head -1 t1.txt
 apple
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ tail -1 t1.txt
 orange
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ echo "grape" >> t1.txt
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ cp t1.txt t1_bak.txt
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ mv t1_bak.txt ~/
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ ls
 log.txt  t1.txt
 djq@LAPTOP-SH5R5OFE:~/ai_exp$ cd ..
 djq@LAPTOP-SH5R5OFE:~$ ls
 CUDA_Lesson       ai_infra
 CUDA_Lesson1      ai_infra_study
 MD                course1
 ai-infra-odyssey  cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
 ai_exp            t1_bak.txt
 djq@LAPTOP-SH5R5OFE:~$ rm t1_bak.txt
 djq@LAPTOP-SH5R5OFE:~$ ls
 CUDA_Lesson       ai_infra
 CUDA_Lesson1      ai_infra_study
 MD                course1
 ai-infra-odyssey  cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
 ai_exp
 djq@LAPTOP-SH5R5OFE:~$
 ```

# T2

```shell
djq@LAPTOP-SH5R5OFE:~$ ls /usr/bin/ | grep "python"
pybabel-python3
python3
python3.10
djq@LAPTOP-SH5R5OFE:~$ ls /usr/bin/ | grep g++ | wc -l
4
djq@LAPTOP-SH5R5OFE:~$ ps aux | grep python
root         195  0.0  0.2  30140 19524 ?        Ss   Oct29   0:00 /usr/bin/python3 /usr/bin/networkd-dispatcher --run-startup-triggers
root         232  0.0  0.2 107196 22120 ?        Ssl  Oct29   0:00 /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal
djq      1078067  0.0  0.0   4036  2396 pts/5    S+   09:12   0:00 grep --color=auto python
djq@LAPTOP-SH5R5OFE:~$
```

# T3

```shell 
djq@LAPTOP-SH5R5OFE:~$ ls -l ~/ai_exp/t1.txt
-rw-r--r-- 1 djq djq 26 Nov  4 08:45 /home/djq/ai_exp/t1.txt
djq@LAPTOP-SH5R5OFE:~$ chmod 640 ~/ai_exp/t1.txt
djq@LAPTOP-SH5R5OFE:~$ ls -l ~/ai_exp/t1.txt
-rw-r----- 1 djq djq 26 Nov  4 08:45 /home/djq/ai_exp/t1.txt
djq@LAPTOP-SH5R5OFE:~$
```

# T4

```shell 
djq@LAPTOP-SH5R5OFE:~/ai_exp$ echo "hello dou bao!" >> ./t1_bak.txt
djq@LAPTOP-SH5R5OFE:~/ai_exp$ cat ./t1_bak.txt
hello dou bao!
djq@LAPTOP-SH5R5OFE:~/ai_exp$
```





### 思考题（写在 md 末尾）

1. `>` 和 `>>` 的本质区别？ >是覆盖写入 >>是行尾追加
2. `less` 和 `cat` 适用场景分别是什么？  less是分页看日志，空格翻页，q退出，cat是一次性输出文件内容
3. WSL 里 `/mnt/e` 和 `~/` 底层存储区别，为什么后面深度学习代码尽量不要放 `/mnt/e`？   权限不同，挂载盘权限777，不利于权限管理
4. `chmod 755`、`chmod 644` 分别代表什么权限？ 755指文件所有者可读可写可执行，同组人可读可执行，其他人可读可执行。644指文件所有者可读可写，同组人可读，其他人可读。



## 二、思考题批改 & 补全

1. **`>` 和 `>>`**

> 你的答案：`>`覆盖写入，`>>`行尾追加 ✔正确 补充：`>`会直接截断清空原有文件；`>>`在文件末尾追加，不删除旧内容。

1. **cat vs less**

> 你的答案 ✔正确

- `cat`：一次性全部打印，适合小文件；
- `less`：分页浏览，适合大日志文件，`空格`翻页，`q`退出。

1. **`/mnt/e`（Windows 挂载盘）vs `~/`（WSL 原生 Linux 家目录）**

> 你只回答了权限，补充完整两点： 1）**性能**：`/mnt/e`是跨系统 9P 协议挂载，IO 读写速度慢，跑 PyTorch、加载数据集会明显变慢；`~/`是 WSL 虚拟磁盘 ext4 原生文件系统，IO 性能高。 2）**权限**：挂载盘强制固定为 777，Linux 原生权限模型失效，容易出现权限玄学 bug；`~/`遵循标准 Linux 权限。 结论：深度学习实验代码、数据集尽量放在 WSL 家目录`~/`。

1. **chmod 755 / 644**

> 你的答案✔正确

- `755`：所有者 rwx，组 rx，其他 rx；常用于可执行脚本 / 程序；
- `644`：所有者 rw，组 r，其他 r；普通文本文件默认权限。

## 三、本次踩坑汇总

1. bash 中普通`echo`不会解析`\n`换行符，要加参数`echo -e`；
2. `>`是覆盖，多次执行会清空旧内容；追加务必使用`>>`；
3. T4 相对路径理解失误：`./`代表**当前目录**；`../`代表上一级目录；
4. WSL 挂载 Windows 磁盘`/mnt/*`性能差，深度学习项目不要放在这里。



