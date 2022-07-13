# 【学习笔记】GitModel统计分析

> &#8195;&#8195;本文是Datawhale和GitModel开源项目的学习记录，一方面是梳理知识框架，另一方面是提升的代码熟练度。这里十分感谢Datawhle贡献者提供的项目资源。
>- 项目源：https://github.com/Git-Model/Modeling-Universe/tree/main/Data-Story
>- 开始时间：2022年7月9日
>- 进度：建模基础（Task2进行中）

# 学习路径

本次项目的主要目标有三个：
- 学习**建模基础知识**，了解从数据导入、探索性数据分析和建立回归/分类模型的过程；
- 阅读资料和代码练习，提高对Pandas、Scipy、Matplotlib等**常用库的实践能力**；
- 通过项目提供的案例，举一反三，提升**统计分析应用能力**。

因此，进一步拆解本次学习项目的目标，可以得到具体的学习任务（这里并不完全按照Datawhale组队学习的Task）。每完成一个学习任务，就在任务后作标记，本文也会随着学习的进度更新。

## 1. Python基础
目标是学习环境搭建，熟悉基础语法，会使用相应的库。
  - 成功**搭建Python环境**，配置好需要使用的Packages。==【Done】==
  - **快速阅读**[**JoyfulPandas**](http://joyfulpandas.datawhale.club/Content/index.html)，关注目录内容，以便后期翻阅。==【Done】==
  - **完成“ Task1：Pandas 动手学入门”的35道题目**。==【Done】==

## 2. 建模基础
目标是学习建模的主要过程，包括EDA、可视化、回归/分类模型。
  - 熟悉EDA过程，**完成“Task2：EDA初体验”**。==【Doing】==
  - 实现可视化，**完成“Task5：数据可视化”**。
  - 学习回归分析原理，**完成 “Task3：统计分析之回归分析 ”**。
  - 学习回归分类原理，**完成“Task4：统计分析之分类分析”**。

## 3. 项目实战
目标是将上述的知识融会贯通，举一反三。
  - 选择一道赛题，**完整走一遍建模流程**。



# 一、Python基础
## （一）环境搭建
> 系统：**Win10**
> 版本：**Python3.7**
>工具：**PyCharm**、**Anaconda(Jupyter Lab)**、**Typora**

**Anaconda**对于新人来说挺友好的，内置了Numpy 、Pandas等常用科学库，安装即可使用。在做数据处理与分析时，通常会使用Jupyter Notebook或Jupyter Lab。个人更推荐后者，其模块化管理功能优秀，支持分屏和Markdown目录，适合作文档管理工具和文本阅读，Code & Write。

**PyCharm**是Python的IDE工具，主要是用于Python开发，功能更多但配置也较为复杂。于个人而言，PyCharm主要是用来看源代码与说明文档，代码规范化和存储为.py文件，偶尔用作远程访问服务器。另外，PyCharm支持使用Anaconda的开发环境，对各类包或是环境管理起来也是比较方便。

**Typora**是基于Markdown的写作工具，优点是所见即所得，而缺点是付费（早期是免费的）。关于支持Markdown的工具，个人比较推荐飞书，免费、功能更多且存储方便，缺点是没办法离线写作。
## （二）Pandas 动手学入门
Pandas基于Numpy进行扩展，是数据处理与数据分析的重要工具。Datawhale的[**JoyfulPandas**](http://joyfulpandas.datawhale.club/Content/index.html)十分适合Pandas入门者去阅读，章节知识划分详细且有具体例子。另外，去年有写过关于Numpy和Pandas数据结构的特点（[链接](https://blog.csdn.net/weixin_44790239/article/details/115033911?spm=1001.2014.3001.5501)），`list`、`set`、`dict`、`ndarray`、`series`、`dataframe`的转化过程，涉及到深度学习，还会接触到`tensor`。这些数据结构是处理数据的基础，因此在学习Pandas时需要体会其每一次变化的逻辑。
- 本项目中的35道题、JoyfulPandas的习题以及[100-pandas-puzzles](https://github.com/ajcr/100-pandas-puzzles)都可用于锻炼Pandas的使用能力。
>本人具备有一定的Python基础，因此直接跳过基础语法部分。
>这里也给大家推荐一些可以学习基础语法的路径。
>入门：B站“Python”教程，任意一个视频。
>进阶：[Intermediate Python(英文)](https://github.com/yasoob/intermediatePython)、[Intermediate Python(中译)](http://shouce.jb51.net/Python_jj/index.html)，需要时再阅读。

## （三）习题摘录
练习题目总共有35道，本文就不逐一展示，挑部分比较有趣的题目讲解一下。（附：[习题个人答案](https://github.com/Hoveii/datawhale_model)）
- **题目12：选择 `age` 在 2 到 4 之间的数据（包含边界值）**

数据结构如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/f67227e227b54b01a3d7acd82ddbfbaa.png)

复合条件查询，在实际问题经常会遇到。有以下三种方法可以解决。
1. 利用索引函数`loc`和`iloc`是较为常见的思路。需要区分两种的场景，前者按索引名（更常用），后者按索引下标。
2. `query(expr)`函数是较为特殊的用法，直接使用列名和匹配信息，以字符串的形式输入，查询思路与索引函数类似。
3. `apply(func)`可以对某一维度`axis`的每个元素进行`func`运算（注：`DataFrame`才有`axis`参数），灵活性很强且常用于各种复杂条件运算，第20题同样可以使用该函数。
```python
# 方法一：直接索引
df.loc[(df['age'] >= 2)&(df['age'] <= 4)]

# 方法二：apply函数
df[df['age'].apply(lambda x: 2 <= x <= 4)]

# 方法三：query函数
df.query("2<= age <= 4")
```
- **题目16：新增一行数据 k，数据自定义，然后再删除新追加的 k 行**

如增加一行，`index为k`，`value=['cat', 2.5, 1, 'yes']`。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f67227e227b54b01a3d7acd82ddbfbaa.png)

对已有数据进行新增行/列。有以下两种方法解决。
1. 索引函数，上面已经讲过。但需要注意的是，不能以`df[col]`方式复合索引赋值，应该使用索引函数，前者会创建副本（不影响原数据），或者是创建视图（影响原数据）。
2. `concat`合并函数。这种方法可以新增多行/列，但要求合并的维度数一致。
```python
# 方法一：索引函数
df.loc['k', :] = ['cat', 2.5, 1, 'yes']

# 方法二：concat合并
tmp = pd.DataFrame(data=[['cat', 2.5, 1, 'yes']], columns=df.columns, index=['k'])
pd.concat([df, tmp], axis=0)
```
- **题目29：等距分组求和**

数据结构如下图所示，A是0到100之间的数，将A分组（组距为10），求每个组中B的和。

![在这里插入图片描述](https://img-blog.csdnimg.cn/6c79368a49344485a197175b18bbb938.png)

本题主要有两个点，等距分组、分组汇总。
1. 分组函数可以自定义，也可以使用`cut`函数，事先给出分割点。
2. 汇总函数主要是`groupby`，类似于SQL中的`group by `

```python
# 第一种方法：自定义函数分组，这里简单粗暴地举个例子
def cut_df(x):
    if x <= 10:
        return '(0, 10]'
    elif x <= 20:
        return '(10, 20]'
    elif x <= 30:
        return '(20, 30]'
    elif x <= 40:
        return '(30, 40]'
    elif x <= 50:
        return '(40, 50]'
    elif x <= 60:
        return '(50, 60]'
    elif x <= 70:
        return '(60, 70]'
    elif x <= 80:
        return '(70, 80]'
    elif x <= 90:
        return '(80, 90]'
    else:
        return '(90, 100]'
df.groupby(by=df['A'].apply(cut_df))['B'].sum()

# 第二种方法：cut函数，简洁很多
df.groupby(by=pd.cut(df['A'], [i * 10 for i in range(11)]))['B'].sum()
```

- **第30至第35题：数据清洗**

原始数据是：

![在这里插入图片描述](https://img-blog.csdnimg.cn/c609b687184345fc9ccee260047c4278.png)

清洗完后数据是：
`From_To`字段拆分且修改字符大小写，`FlightNumber`字段是填充缺失值，
`Airline`字段是提取关键信息，`RecentDelays`是按元素位置拆分成多列。

![在这里插入图片描述](https://img-blog.csdnimg.cn/328d7f1e055643f9a532deaf29281cc0.png)

1. `From_To`字段处理
主要是使用`Series.str.StringMethods`处理，要求该序列中的数据类型为字符串形式。[JoyfulPandas第三章节](http://joyfulpandas.datawhale.club/Content/ch8.html#id8)对该部分已有比较详细的讲述。针对这道题目，先按"_"拆分字符成列，然后仅首字母大写即可。
```python
# 拆分列
tmp = pd.DataFrame(df['From_To'].str.split('_').tolist(), columns=['From', 'To'])

# 仅首字母大写
tmp = tmp.apply(lambda x: x.str.capitalize(), axis=0)

# 删除与合并
df.drop(columns=['From_To'], inplace=True)  # 注意修改原数据要inplace=True，否则只会创建副本
df = pd.concat([tmp, df], axis=1)
```
2. `FlightNumber`字段处理
该字段下存在缺失值，常见的缺失值处理方法有：删除、填充，而填充缺失值一般又分为邻近填充（向前、向后）和插值填充。这里要求的是按前后平均进行填充，即**线性插值**，使用的是`interpolate(method='linear')`函数，具体的`method`需看官方文档。
```python
# 线性填充
df['FlightNumber'] = df['FlightNumber'].interpolate(method='linear').astype('int')
```
3. `Airline`字段处理
该字段目标是**提取单词**，清洗无关符号和数字。因此，个人觉得比较好的方式是用正则表达式，具体思路是：先使用`replace`剔除非字母和空格的字符`[^A-Za-z\s]`，再使用`strip`去除首尾空格。
```python
# 字符串处理
df['Airline'] = df['Airline'].str.replace('[^A-Za-z\s]', '', regex=True).str.strip()
```
4. `RecentDelays`字段处理
该字段下存在`list`类型（长度不一），即复合结构，需要对list分列，并按每个list元素下标记录到对应列中。
个人的思路是，利用了`Ndarray`和`DataFrame`的**广播机制**，将长度不一的`list`扩展成长度一致的二维数组结构，自动进行空值填充。这道题中，如果字段存入的是`key:value`多个键对字符串的形式，按`key:value`分列则是需要考虑使用`json.loads`。（这也算一个小拓展）
```python
delays = pd.DataFrame(df["RecentDelays"].tolist())  # 列表转DF
delays.columns = ['delays_{}'.format(i + 1) for i in range(len(delays.columns))]  # 列名修改
df.drop(columns=['RecentDelays'], inplace=True)  # 删除列
df = pd.concat([df, delays], axis=1)  # 合并列
```
