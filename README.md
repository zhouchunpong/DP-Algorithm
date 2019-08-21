
# 动态规划问题 Dynamic Programming Algorithm
## Abstract
>**动态规划**（英语：Dynamic programming，简称DP）通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。动态规划常常适用于有重叠子问题和最优子结构性质的问题，动态规划方法所耗时间往往远少于朴素解法。动态规划背后的基本思想非常简单。大致上，若要解一个给定问题，我们需要解其不同部分（即子问题），再根据子问题的解以得出原问题的解。

>通常许多子问题非常相似，为此动态规划法试图仅仅解决每个子问题一次，从而减少计算量：一旦某个给定子问题的解已经算出，则将其记忆化存储，以便下次需要同一个子问题解之时直接查表。这种做法在重复子问题的数目关于输入的规模呈指数增长时特别有用。                    **From 维基百科**

## Introduction
看下面的一个例子，我们需要从以下带时间窗的任务中，选择总价值最大的，每次只能做一个任务，而且任务之间的时间不能冲突。每一个矩形上面标的数字代表该任务完成的value。如下所示：
<center><img src="https://github.com/zhouchunpong/test/blob/master/1.PNG" width = "500" alt="1" align=center></center>


**解决思想：选 或 不选**

设$OPT(i)$为当考虑任务i时的最优选择。举个例子，我们先考虑第8个任务，对于任务8，我们有两种选择：
1. 选择执行任务8，则可以获得其收益，$value(8)=4$; 同时在执行完任务8后，又面临选择，是否去执行任务4，即是$OPT(4)$。故执行任务8的收益可以写作：$4 + OPT(5)$
2. 不去执行任务8：这种情况就判断下一个任务，即任务7。此时的收益是：$OPT(7)$

综上，我们需要保证最大收益，$OPT(8)$可以表示如下：
$$OPT(8) = max \begin{cases} 
4 + OPT(5) &if\ choose\ the\ task\ 8\\\\ 
OPT(7) & if\ not\ choose\ the\ task\ 8 
\end{cases}$$

然后我们就可以推广到任务i了，依然有两种选择，做任务i或者不做，再去取两种情况的最大值，公式描述如下：
$$OPT(i) = max \begin{cases} 
value(i) + OPT(prev(i)) &if\ choose\ the\ task\ i\\ 
OPT(i-1) & if\ not\ choose\ the\ task\ i 
\end{cases}$$
其中$prev(i)$表示如果选择了任务i，那么在之前能选择的最近任务（确保时间不冲突）。例如选择做任务8，那么之前最近能选择任务5；如果选择做任务7，之前能做的最近任务是任务3。所有的$prev(i)$如下表所示：

$prev(8)$|$prev(7)$|$prev(6)$|$prev(5)$|$prev(4)$|$prev(3)$|$prev(2)$
:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:
5|3|2|0|1|0|0

下面开始进行任务的选择：

1. 当i=1时，只有一个任务，肯定选择执行，故$OPT(1)=value(1)=5$
2. 当i=2时，任务1和2之间选择，$max\{value(2),OPT(1)\}=OPT(1)=5$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以此类推.......

7. 当i=7时，$OPT(7)= max\{2+OPT(3),OPT(6)\}=2+OPT(3)=10$
8. 当i=8时，$OPT(8)= max\{4+OPT(5),OPT(7)\}=4+OPT(5)=13$

算完所有的i后，就得到如下的表，可以发现当i=8时，此时收益最大为$OPT(8)=13$，选取的任务是[1,4,8]。
$i$|$prev(8)$|$OPT(i)$|task
:--:|:--:|:--:|:--:
1|0|$value(1)=5$|[1]
2|0|$value(1)=5$|[1]
3|0|$value(3)=8$|[3]
4|1|$value(1,4)=9$|[1,4]
5|0|$value(1,4)=9$|[1,4]
6|2|$value(1,4)=9$|[1,4]
7|3|$value(3,7)=10$|[3,7]
8|5|$value(1,4,8)=13$|[1,4,8]

## Code
用递归的思想, python代码如下所示（注意Python索引是从0开始，而不是1）：
```python
value=[5,1,8,4,6,3,2,4]
prev =[0,0,0,1,0,2,3,5]

def rec_opt(i):
    if i==0:
        return 0
    elif i==1:
        return value[0]
    elif i==2:
        return max(value[0],value[1])
    else:
        choose_i=value[i-1]+rec_opt(prev[i-1])
        not_choose_i=rec_opt(i-1)
        return max(choose_i,not_choose_i)
for i in range(1,9):
    print(rec_opt(i),end=' ')
```
输出如下：
```
5 5 8 9 9 9 10 13 
```
虽然递归能解决，但里面存在着许多的重复计算，下面换种思路，使用非递归的。主要思想就是把每个任务的$OPT(i)$存起来，而不用每次都计算。代码如下：
```python
def dp_opt(i):
    opt=[0]*(i+1)
    opt[0]=0
    opt[1]=value[0]
    opt[2]=max(value[0],value[1])
    for i in range(2,i+1):
        choose_i=value[i-1]+opt[prev[i-1]]
        not_choose_i=opt[i-1]
        opt[i]=max(choose_i,not_choose_i)
    return opt[1:]
print(dp_opt(8))
```
输出如下：
```
[5, 5, 8, 9, 9, 9, 10, 13]
```

## Exercise One
在数组arr=[1,2,4,1,7,8,3]中选出一堆不相邻的数，使之选出的数字之和最大。例如选择数字2,1,8,他们之间就互不相邻，我们想找出最大的数字组合。

**解决思想：选 或 不选** 

对于第i个数字，如果选了则下一步只能考虑第(i-2)个数字，因为不能相邻；如果不选i，就考虑i-1。
用公式表示如下：
$$OPT(i) = max \begin{cases} 
arr(i) + OPT(i-2) &if\ choose\ the\ nunber\\ 
OPT(i-1) & if\ not\ choose\ the\ number 
\end{cases}$$

```Python
arr=[1,2,4,1,7,8,3]

def rec_opt(i): #递归
    if i==0:
        return arr[0]
    elif i==1:
        return max(arr[0],arr[1])
    else:
        choose_i=arr[i]+rec_opt(i-2)
        not_choose_i=rec_opt(i-1)
        return max(choose_i,not_choose_i)
print(rec_opt(6))

def dp_opt(i): #非递归
    opt=[0]*i
    opt[0]=arr[0]
    opt[1]=max(arr[0],arr[1])
    for i in range(2,i):
        choose_i=arr[i]+opt[i-2]
        not_choose_i=opt[i-1]
        opt[i]=max(choose_i,not_choose_i)
    return opt[-1]
print(dp_opt(7))
```
无论使用递归或者非递归，结果都是15.

## Exercise Two
给定一个数组arr=[3,34,4,12,5,2]和整数S=9，能否在数组arr=中选取一堆数字，使得这些数字之和等于给定整数，如果存在，返回True；否则返回False。

**解决思想：选 或 不选** 
我们把原问题拆分成子问题，原问题定义为Subset(i,S)，其中i表示考虑到前i个数，S是我们需要满足的整数。对于第i个数，如果选了，则S=S-arr(i)；如果没选，则考虑前i-1个数，S保持不变。依次递归，只要有一种满足条件就返回True。用公式描述如下： 
$$Subset(i,S) = OR \begin{cases} 
Subset(i-1,S-arr(i)) &if\ choose\ the\ number\\ 
Subset(i-1,S) & if\ not\ choose\ the\ number 
\end{cases}$$

```Python
arr=[3,34,4,12,5,2]
def rec_subset(i,s): #递归
    if s==0:
        return True
    elif i==0: #只剩一个数的时候
        return arr[0]==s
    elif arr[i]>s: #当前数字比s大时
        return rec_subset(i-1,s)
    else:
        choose_i=rec_subset(i-1,s-arr[i])
        not_choose_i=rec_subset(i-1,s)
        return choose_i or not_choose_i
for i in range(10,15):
    print( rec_subset(len(arr)-1,i))

import numpy as np
def dp_subset(S): #非递归
    subset=np.zeros( (len(arr),S+1) ,dtype=bool) #二维数据，来保存每个Subset的状态
    subset[:,0]=True
    subset[0,:]=False
    subset[0,arr[0]]=True
    for i in range(1,len(arr)):
        for s in range(1,S+1):
            if arr[i]>s:
                subset[i,s]=subset[i-1,s]
            else:
                choose_i=rec_subset(i-1,s-arr[i])
                not_choose_i=rec_subset(i-1,s)
                subset[i,s] = choose_i or not_choose_i
    r,c=subset.shape
    return subset[r-1,c-1]
for i in range(10,15):
    print( dp_subset(i))
```
输出结果都是：True True True False True

## Reference
1. <https://www.bilibili.com/video/av16544031>
2. <https://www.bilibili.com/video/av18512769>
