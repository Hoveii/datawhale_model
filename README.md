# 【学习笔记】GitModel统计分析

> &#8195;&#8195;本文是Datawhale和GitModel开源项目的学习记录，一方面是梳理知识框架，另一方面是提升的代码熟练度。这里十分感谢Datawhale和GitModel贡献者提供的项目资源。
>- 项目源：https://github.com/Git-Model/Modeling-Universe/tree/main/Data-Story
>- 开始时间：2022年7月9日
>- 进度：项目实战

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
  - 熟悉EDA过程，**阅读“Task2：EDA初体验”**。==【Done】==
  - 实现可视化，**完成“Task5：数据可视化”**。 ==【Done】==
  - 学习回归分析原理，**完成 “Task3：统计分析之回归分析 ”**。==【Done】==
  - 学习回归分类原理，**完成“Task4：统计分析之分类分析”**。==【Done】==

## 3. 项目实战
目标是将上述的知识融会贯通，举一反三。
  - 选择一道赛题，**完整走一遍建模流程**。==【Doing】==



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
# 二、流程认识与EDA入门
## （一）数据挖掘流程
根据以往的实践经验来看，数据挖掘主要是围绕着What、Why、How这一过程展开。
###  1. **What**
这一部分主要是数据挖掘的前期准备，同时也是最容易忽略的部分。这一阶段中，我们需要明确要解决的业务问题是什么，如何将一个比较复杂或是难以描述业务问题**转化为简单且可量化的业务问题**。针对这个问题，需要什么变量或是什么类型的数据，即**找到这一问题的度量和维度**；由此，我们可以**获取**到哪些数据，并考虑用什么方式去**存储**。

![在这里插入图片描述](https://img-blog.csdnimg.cn/7490644058894180a4457fffeef3af7d.png)
### 2. **Why**
这一部分是数据挖掘中最常见的部分，也是各类竞赛中的核心步骤。下面的流程框架是参考了Datawhale的《如何打一个数据挖掘比赛——入门版》。
![在这里插入图片描述](https://img-blog.csdnimg.cn/0ce9bb280b774af5ae459764c0441d92.png)
### 3. **How**
这一部分主要是数据挖掘的最终部分，主要关注的是如何解决最先提出的业务问题，不仅仅是给出一个较优的模型方法，还需要给出模型可解释性，判断在实际业务中应用、模块化或是产品化的可能。
![在这里插入图片描述](https://img-blog.csdnimg.cn/dca83ff416834bd2a5d7c00d967f1a4e.png)



## （二）EDA入门
一般来说，数据分析前期流程主要包括数据采集、数据存储。像Kaggle、阿里云天池、科大讯飞iFLY等比赛平台通常会提供数据集，数据分析的前期工作基本满足需要，因此“What”部分略过，主要关注“Why”部分。
### 1. 数据读取与解释
- 竞赛数据集一般是以csv格式存储，通常可以使用pd.read_csv导入，如果导入的中文出现乱码时，需要额外加上参数encoding='gbk'。项目中的读取例子如下：
```python
# 加载必要库
import pandas as pd # 数据分析库
import numpy as np # 矩阵计算
import matplotlib.pyplot as plt # 画图

import plotly.express as px # 画图
from statsmodels.graphics.gofplots import qqplot # 统计模型
import seaborn as sns # 统计绘图
%matplotlib inline

# 避免产生报警告
import warnings 
warnings.filterwarnings('ignore')

house = pd.read_csv("./data/boston.csv")
house.head() # 读前五行

# from IPython.display import display, HTML
# display(HTML(house.head().to_html(index=False))) # 读前五行，输出不显示行索引
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/3b80486644ff42709b8446f06de7753b.png)
- 读取数据前需要了解数据集基本信息，如样本量、字段名/字段数量、字段类型、字段含义。本次使用的数据集是波士顿房价(Boston house prices dataset)。该数据集共有506条数据，每条数据里面共有14个字段（包括需要预测的房价）。
- 	字段含义如下所示（来源：`sklearn.datasets.load_boston`）：

```
  - CRIM     per capita crime rate by town
  - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
  - INDUS    proportion of non-retail business acres per town
  - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
  - NOX      nitric oxides concentration (parts per 10 million)
  - RM       average number of rooms per dwelling
  - AGE      proportion of owner-occupied units built prior to 1940
  - DIS      weighted distances to five Boston employment centres
  - RAD      index of accessibility to radial highways
  - TAX      full-value property-tax rate per $10,000
  - PTRATIO  pupil-teacher ratio by town
  - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
  - LSTAT    % lower status of the population
  - MEDV     Median value of owner-occupied homes in $1000'
```

```python
# 查看字段信息
print("字段数：{}\n样本数：{}\n".format(house.shape[1], house.shape[0]))
house.info()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/94021c8a4f494cc698eea3c8cc76e9e7.png)

### 2. 数据预处理
数据读取后并不能直接进行模型训练，缺失值和文本变量会对模型训练产生影响。因此对于存在上述情况的字段或记录，需要作出相应的调整，如直接剔除、填补、转换等。下面按项目的例子，主要总结一下缺失值的处理过程。
- 识别数据的缺失值。
第一种方法：数据描述性统计，可以通过`count`一行比较出缺失数量。
第二种方法：`data.isnull()`函数，`isnull().sum(axis=0)`思路可以统计每列的缺失值的数量，进一步可以实现缺失数据可视化。
```python
# 模拟含有缺失的数据
null = pd.read_csv("./data/boston_null.csv")

# 方法一：数据描述性统计
null.describe()
# 方法二：data.isnull()函数
null.isnull()
# 缺失数据可视化
missing = null.isnull().sum()
missing.plot.bar()  # 柱状图
sns.heatmap(null.isnull())  # 热力图
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/e748007add9048bba9d3379d10c0571e.png)


- 缺失数据处理
在处理前需要做缺失比例统计，查看每一个字段的缺失值占比，若是缺失占比非常小时可使用填充法进行填补，若缺失占比非常大时可以考虑直接剔除。
```python
# 缺失占比统计
data = pd.DataFrame(null.columns, columns=['Features'])
data['Miss_rate'] = (null.isnull().sum(axis=0) / len(null)).values
# 均值填补缺失
null.fillna(null.mean())
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/41cfdc65615b4b37b1ab2b57ac759741.png)


### 3. 探索性数据分析
主要分为单变量分析与多变量分析。在分析前需要先识别变量是离散/分类型还是连续型，然后根据变量的类型再进一步分析，如连续型单变量分析、离散型单变量分析、连续型-离散型组合分析、连续型-连续型组合分析、离散型-离散型组合分析等。
- 识别变量的类型，项目实例中给出的思路是计算每个字段值去重后的个数，若个数很小则视为离散型变量，否则视为连续型变量。
```python
data = pd.DataFrame(null.columns, columns=['Features'])
data['Unique'] = (null.nunique()).values
data  # CHAS和RAD为典型的离散型变量
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/5c02613754bc42d99a056440313f0829.png)


- 单变量分析
离散型变量看**频数分布**（柱状图、饼图）、连续型变量看**直方分布**（直方图、KDE）。
```python
# 项目中使用的代码，连续型变量使用直方分布图+KDE估计图
plt.figure()
sns.distplot(house["MEDV"],
            bins=100,  # 柱子的个数
            hist=True,  # 是否绘制直方图
            kde=True,  # 是否绘制密度图
            rug=True,  # 数据分布标尺
            color='r',  # 颜色
#             vertical=False,  # 是否水平绘制
#             norm_hist=False,  # 标准化，kde为True时自动标准化
            axlabel=None,  # x轴标注
            label=None,  # 图例标签，通过plt.legend()显示
            ax=None,
            )

# 除了distplot，也可以直接使用kdeplot
# KDE估计图
r = sns.kdeplot(data=house, x="MEDV")
# 提取峰值信息
x = r.lines[0].get_xdata()
y = r.lines[0].get_ydata()
maxid = np.argmax(y)
print('数据主要集中在{}'.format(np.round(x[maxid], 0)))
plt.vlines(x[maxid], ymin=0, ymax=y[maxid], colors='red')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/e0497a9b4caf4f2a894ad36ec589ae40.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/9a82eb5f5f6841f6b90c04cca23eebaf.png)

- 多变量分析通常做相关分析。常见方法如下：
（1）变量相关系数 + 热力图，一般使用`sns.heatmap(df.corr())`，色彩深浅与相关性强调有关
（2）散点图，一般使用`sns.pairplot(df)`，对角线为单变量直方图、其他为散点图。可以通过散点的疏密和走向看出关联程度。
```python
# 单变量分布+散点图
sns.pairplot(house[["AGE", "DIS"]], size = 3)

# 相关系数 + 热力图
plt.figure(figsize=(12, 9))  # 设置画布大小
sns.heatmap(house.corr(), cmap='YlGn')  # cmap为配色
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/d0e85ec6d0924168a8526cf926f93324.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/bddf4d8b2b1648e7bad70bd99179754c.png)
# 三、回归分析
## （一）线性回归的概念
本期项目中有提到，数据分析的目的是找到x与y的关系，并用模型显式表示出来。回归分析则是一种寻找被解释变量和解释变量之间函数关系的数学方法，那么解释变量和被解释变量的关系如下所示。
$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p
$$
一般来说，解释变量视为随机变量，那么上述的模型可以表示为$F_{y|x}$，即给定$x$可以通过确定的$F$得到$y$的状态。然而现实数据并不一定能完美地得到这样一个关系，实际上可能会存在干扰项（随机误差），模型可以改成下面的形式，显然$y$也应该是随机变量。
$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + u
$$
显然，我们很难在真实情况中得到$\{ y|x\}$，随机误差是无法精确计算出来的。因此为了合理地得到$y$与$x$的关系，我们通常选择对$y$取期望。
$$
E(y) =  E(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p) + E(u)
$$
我们的目标是给定$x$得到$y$，那么可以得到条件期望：
$$
E(y|x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + E(u)
$$
上述说到，$u$是随机误差，即$u \sim N(0, \sigma^2)$，带入$E(u)=0$可得：
$$
E(y|x) =  \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p 
$$
因此，我们可以得到一般回归模型。
$$
y = E(y|x) + u
$$
从上面的式子可以看出，**回归模型建模本质是条件均值建模**，最终是通过各种方法求出$E(y|x)$中的各个参数（系数），从而得到$x$和$y$的一种量化关系。

## （二）线性回归模型的假设
对于初学者来说，刚开始看回归模型的假设条件时会感到十分困惑，为什么用这么简单的模型需要设定这么多假设。这次项目中的原理和例子详细地讲述了各个假设是什么、为什么这样设置以及不这样设置会导致什么问题，十分通俗易懂。有兴趣的同学可以阅读GitModel[统计分析：第三章](https://github.com/Git-Model/Modeling-Universe/tree/main/Data-Story)或者伍德里奇的《计量经济学导论》，这里只简单记录假设含义与意义。
### 1. 线性假定
- 具体形式表现为：$y = \beta^T x + u$，其中w的元素均为常数。
- 顾名思义，解释变量和被解释变量具有线性关系（这里仅关注狭义的线性回归模型）。
- 但实际上这种线性关系可以拓展到非线性的情况，如经济学中有个著名的函数“Cobb-Douglas生产函数”，即：$Y=AK^\alpha L^\beta$，两边取自然对数后可以得到：$\ln(Y) = \ln(A) + \alpha \ln(K) + \beta \ln(L)$，这就可以使用线性回归的方法估计出弹性系数$\alpha$和$\beta$。

### 2. 严格外生性假定
- 具体表现为：$E(u|x) = 0$，即扰动项独立于观测数据
- 该假定也可以称为随机误差条件均值零假定，当线性模型中存在遗漏解释变量、解释变量测量误差以及反向因果的情况时，会导致干扰项与解释变量存在相关性，这也是常说的“内生性问题”。该假定是为了保证系数估计的无偏性。
### 3. 随机抽样假定
- 具体表现：样本$\{x\}$是随机、独立的。
- 这里涉及的是统计基础和抽样问题，样本随机独立是数理计算的前提条件。
### 4. 非完全共线性假定
- 具体表现为：不存在严格（完全）的多重共线性，即不存在某一解释变量可由其余解释变量线性表示。
- 解释变量之间存在较强相关性，则会共线性问题，影响的是系数的估计方差。
### 5. 球形扰动项假定
- 具体表现为：$Var(u|x) = \sigma^2$，即扰动项条件同方差和无自相关。
- 这个又可以称为同方差假定，这一假定是为了保证系数显著性检验的准确性。
- 在使用**t检验**法做系数显著性检验时，需要用残差估计随机误差的方差来计算检验统计量
$$
\operatorname{se}\left(\hat{\beta}_{j}\right)=\hat{\sigma} \sqrt{\left(X^{\prime} X\right)^{-1}} \\
t = \frac{\hat\beta_j - \beta_j}{\operatorname{se}\left(\hat{\beta}_{j}\right)} \sim t_{n-k-1}
$$
在同方差的情况下可以得到随机误差的方差的无偏估计：$\sigma^2 = \frac{1}{df}\sum\limits_{i=1}^{n}{(\hat u)^2}$
$$
\hat{\sigma}^{2}=\frac{1}{n-k-1} \sum_{i=1}^{n}\left(y_{i}-\hat{\beta}_{0}-\hat{\beta}_{1} x_{i 1}-\cdots-\hat{\beta}_{k} x_{i k}\right)^{2}=\frac{RSS}{n-k-1}
$$
- 当存在异方差时，意味着随机误差的方差估计是有偏的，并导致检验统计量不准确，系数的显著性检验失效。

### 6. 正态性假定
- 具体表现为：$u|x \sim N(0, \sigma^2)$
- 该假定是狭义回归的重要特征，在给定$x$的情况下等价于$y|x \sim N(E(y|x), \sigma^2)$，表明了被解释变量与残差是相近的分布（正态分布），基于正态假定才有后面“Z检验”、“T检验”和“F检验”这些显著性检验方法。
- 实际上，由中心极限定理可知，大样本情况下可以看作近似服从正态分布，因此在大样本且分布是非正态的情况下也能满足这条假定。
- 当$y$服从二项分布、泊松分布等情况时，可以对线性模型进行拓展，引入“指数分布族”的概念并打破了正态性的假定，从而构建出应用更广泛的“广义线性模型”。狭义线性模型可以视为广义线性模型的一种特殊情况。

## （三）线性回归模型的优劣势
- 线性回归模型是一个基础的模型，但绝不是一个简单的模型。线性回归模型强大的原因在于模型的解释性，一方面可以给出模型每个变量的系数（重要性程度，影响方向），另一方面提供了每个系数的有效性（显著性检验），这两个优势决定了它在重解释的影响因素分析领域内有所成就。因此，在使用线性回归模型时要重视其回归分析的解释性功能，善用假设检验的方式挖掘有效变量。
- 但线性回归模型的假设条件过于严格，同时对于复杂非线性的情况下十分乏力，这就造成了线性回归模型在预测能力上不及树模型等。

## （四）作业答案参考
本节的作业题目十分有趣，个人的答案如下所示。
### 题目1
![请添加图片描述](https://img-blog.csdnimg.cn/a4b3c1465db049c3a8ed0d54adcb53c0.png)
- 导入相应的库，调整页面设置
```python
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文正常显示
plt.rcParams['axes.unicode_minus'] = False  # "-"符号正常显示
```
- 读取数据集，注意这里的数据文件是`dta`格式，来源于`stata`软件
```python
# 读取数据集, pandas支持读取dta文件
data = pd.read_stata('./data/bwght2.dta')
```
- 建立线性回归模型，`lbwght`为被解释变量，`npvis`和`npvissq`为解释变量（后者是前者的平方）
```python
res = sm.formula.ols('lbwght~npvis+npvissq', data=data).fit()
res.summary()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/1aef0a4e9ac440d4af73a6b3256514d1.png)
- 上述结果回答了第一个问题，二次项`npvissq`系数为`-0.0004`且通过显著性检验。模型结果如下所示：
$$
\log(bwght) = 7.9579 +0.0189 \cdot npvis - 0.0004 \cdot npvis^2
$$
- 显然，根据原模型推导，可以得到如下的式子。
$$
\log(bwght) = \beta_0 + \beta_1 \cdot npvis + \beta_2 \cdot npvis^2 \\
\frac{\Delta(\log(bwght))}{\Delta(npvis)} = \beta_1 + 2 \beta_2 \cdot npvis \\
 \frac{\Delta(\log(bwght))}{\Delta(npvis)}=0 \Longrightarrow npvis = - \frac{\beta_1}{2\cdot \beta_2}
$$
- 上述结果回到了第二个问题，当$\beta_2$小于0时存在极大值，带入系数结果可求得`npvis`，这里可以用python模拟出结果。
```python
def func(x, b0, b1, b2):
    return b0 + b1 * x + b2 *x ** 2


x = np.linspace(0, 100, num=1000)
y = func(x, *res.params)
max_ind = np.argmax(y)

fig, ax = plt.subplots(figsize=(12,8))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.plot(x, y)
plt.scatter(x[max_ind], y[max_ind], color='red')
plt.vlines(x[max_ind], ymin=0, ymax=y[max_ind], colors='red', linestyle='dashed')
plt.hlines(y[max_ind], xmin=-100, xmax=x[max_ind], colors='red', linestyle='dashed')
plt.annotate("npvis = {}，lbwght = {}".format(np.round(x[max_ind]), np.round(y[max_ind])), (x[max_ind], y[max_ind]), 
             xytext=(-65, 15), textcoords='offset points', bbox=dict(boxstyle='round', pad=0.5, fc='white'), color='black', weight='heavy', size=15) 
plt.tick_params(labelsize=15)
plt.xlabel('npvis', size=15)
plt.ylabel('lbwght', size=15)
plt.xlim(xmin=0)
plt.ylim(ymin=0, ymax=9)
plt.savefig('res01.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/25b7ec0d810849d0bdaf6897b81883a4.png)

- 上述结果可以看到，`lbwght`和`npvis`的关系呈现抛物线的形态，这就意味着`npvis`对`lbwght`的**边际贡献递减**。当$npvis > 22$时，边际贡献为负数，一种可能是确实是存在边际贡献为负的情况，另一种可能是$npvis>22$的样本数很少而导致估计不准确，还有一种可能是该变量本身不适合使用二次项。结合下面统计的结果可以发现，产前检查次数超过22次的实际样本量略少。
这里有两个不一样的角度看：
	- 一种角度将产检次数与怀孕状况关联起来，即产检次数反映怀孕的状况，产检次数过多有可能是怀孕状况较差，产下的婴儿体重会比正常情况低也是有可能的，但这种情况无法解释产检次数越多，婴儿体重越大的合理性。
	- 另一种角度是将产检次数与怀孕状况剥离开来，不是因为产检次数多或少导致婴儿体重下降，而是怀孕情况影响产检次数和产后婴儿的体重，在这里就不是一种因果关系，而是相关关系；只有剥离怀孕状况的差异才能有效估计产检次数对婴儿体重的影响，因此无法解释产检次数对婴儿体重的影响关系。
	- 综合以上两种来看，无论哪一种角度都是无法解释产检次数对婴儿体重影响的关系，其系数不具备现实意义，但需要注意到的是，产检次数存在一个临界点，即超过22次时反映了怀孕状况很差，这一点是可以进行讨论且具备实际意义的，产检次数超过一定次数时视为怀孕状况存在异常，需要做好婴儿/孕妇产前的护理准备，临界点具备一定的现实意义。
```python
# npvis超过22的样本比例
(data['npvis'] > 22).value_counts().apply(lambda x: '{}%'.format(np.round(x / len(data) * 100, 2)))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/032a2c6007a949a89d790f058d2c362c.png)
- 在上述模型的基础上引入`mage`及其二次方项`magesq`，得到系数和显著性。
$$
\log(bwght) = 7.5837 + 0.0254 \cdot mage - 0.0004 \cdot magesq + 0.0180 \cdot npvis - 0.0004 \cdot npvis^2
$$
```python
res = sm.formula.ols('lbwght~mage+magesq+npvis+npvissq', data=data).fit()
data['resid'] = res.resid
res.summary()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/fc1e05633e9e471fa5aedbf7f60e85c2.png)
- 从上述结果可以看到，`mage`及其二次项的系数均通过检验，可计算出临界点。
```python
# 回归模型函数
def func(x, b0, b1, b2, b3, b4):
    return b0 + b1 * x + b2 * x ** 2 + b3 * 1 + b4 * 1

x = np.linspace(0, 100, num=1000)
y = func(x, *res.params)
max_ind = np.argmax(y)

fig, ax = plt.subplots(figsize=(12,8))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.plot(x, y)
plt.scatter(x[max_ind], y[max_ind], color='red')
plt.vlines(x[max_ind], ymin=0, ymax=y[max_ind], colors='red', linestyle='dashed')
plt.hlines(y[max_ind], xmin=-100, xmax=x[max_ind], colors='red', linestyle='dashed')
plt.annotate("npvis = {}，mage = {}".format(np.round(x[max_ind]), np.round(y[max_ind])), (x[max_ind], y[max_ind]), 
             xytext=(-65, 15), textcoords='offset points', bbox=dict(boxstyle='round', pad=0.5, fc='white'), color='black', weight='heavy', size=15) 
plt.tick_params(labelsize=15)
plt.xlabel('npvis', size=15)
plt.ylabel('mage', size=15)
plt.xlim(xmin=0)
plt.ylim(ymin=0, ymax=9)
plt.savefig('res02.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/8d4fa322ceed4d68aca6899cd19d89bf.png)

- 从上述结果可以看到，母亲的年龄对婴儿体重存在边际递减效应，而$mage=31$是一个重要的临界点。`mage`在20到31之间时，婴儿体重有较高的水平；而`mage`在31以后，婴儿体重随着`mage`下降。
	- 从数据比例的角度看，年龄超过31岁的样本比例在33.02%，样本数量足够多，检验具备可靠性。
	- 从医学的角度看，23到30岁之间是女性生育的最佳年龄段，因此递增那一部分是合理的；而30岁以后，随年龄的增长，不利于生育的因素会增多，低出生体重儿发病率会越高，因此递减那一部分也是合理的。
	- 综上来看，$mage=31$这一临界点具备现实意义。

- 从上面的模型结果来看，$R^2$只有0.026，说明当前模型并不能解释`log(gwght)`大部分的变异，因此还需要考虑加入其他的变量。

### 题目2
![请添加图片描述](https://img-blog.csdnimg.cn/9e06689fb6234569a419d14a53784b10.png)
- 读取数据，注意数据是txt格式
```python
data = pd.read_table('./data/P176.txt')
```
- 分别使用WLS和OLS估计`X`对`Y`的系数，WLS中$h(x)$为$x^2$
```python
res_ols = sm.formula.ols('Y~X', data=data).fit()
res_wls = sm.formula.wls('Y~X', data=data, weights=1/data['X']**2).fit()
print("OLS Result：\n", res_ols.summary().tables[1])
print("WLS Result：\n", res_wls.summary().tables[1])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/96ae7bafcd1f4891bda858604e50d69f.png)
- 画出两个模型的散点图+回归直线图
```python
# 散点图 + 回归直线
y_ols = res_ols.fittedvalues
y_wls = res_wls.fittedvalues
plt.figure(figsize=(12, 6))
plt.scatter(x=data.X, y=data.Y, color='blue')
plt.plot(data.X, y_ols, 'r-', label='OLS')
plt.plot(data.X, y_wls, 'g-', label='WLS')
plt.legend(prop={'size':15})
plt.tick_params(labelsize=15)
plt.xlabel('X', size=15)
plt.ylabel('Y', size=15)
plt.savefig('res03.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/5732ba206f0549ad82e80edbf0c17191.png)
- 画出残差图，显然可以发现两种模型的残差仍然具有异方差。
```python
# 残差散点图
plt.figure(figsize=(12, 6))
plt.scatter(x=data.X, y=res_ols.resid, color='red', label='ols_resid')
plt.scatter(x=data.X, y=res_wls.resid, color='green', label='wls_resid')
plt.legend(prop={'size':15})
plt.tick_params(labelsize=15)
plt.xlabel('X', size=15)
plt.ylabel('Y', size=15)
plt.savefig('res04.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/955aae3e44a94d9cb48ef8444f66a107.png)
- 使用FGLS模型，中间推导过程如下所示：
$$
Var(u|x) = \sigma^2 \cdot \exp(\delta_0 +\delta_1 \cdot x_1 + \delta_2 \cdot x_2) \\
E(u^2|x) = \sigma^2 \cdot \exp(\delta_0 +\delta_1 \cdot x_1 + \delta_2 \cdot x_2) \\
\log(u^2) = \delta_0 +\delta_1 \cdot x_1 + \delta_2 \cdot x_2 + \log(\sigma^2)
$$
```python
# FGLS
# 对y拟合回归
res_ols = sm.formula.ols('Y~X', data=data).fit()
# 对残差平方取对数
data['lresidsq'] = res_ols.resid.apply(lambda x: np.log(x ** 2))
# 拟合对数残差平方拟合回归
resid_ols = sm.formula.ols('lresidsq~X', data=data).fit()
# 通过回归预测值反推计算得到h(x)值
data['h'] = resid_ols.fittedvalues.apply(lambda x: np.exp(x))
# 以h为权重拟合WLS
res_fgls = sm.formula.wls('Y~X', weights=data['h'], data=data).fit()
# 绘出残差散点图
plt.figure(figsize=(12, 6))
plt.scatter(x=data.X, y=res_ols.resid, color='red', label='ols_resid')
plt.scatter(x=data.X, y=res_wls.resid, color='green', label='wls_resid')
plt.scatter(x=data.X, y=res_fgls.resid, color='blue', label='fgls_resid')
plt.legend(prop={'size':15})
plt.tick_params(labelsize=15)
plt.xlabel('X', size=15)
plt.ylabel('Y', size=15)
plt.savefig('res05.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/431314b72a8a49d2b54ec2bea511c220.png)
- 从上面的结果可以看到，三种模型都没办法消除异方差。这时候我们应该考虑异方差产生的三种情况。
  - 总体分布并非正态分布，可能存在偏态（QQ图+KS检验）。
  - 样本量较少且存在较大的测量误差（对数化）。
  - 遗漏变量（加入其他解释变量）。
- 对被解释变量和解释变量进行正态性检验，这里仅展示被解释变量。
```python
from scipy import stats

# 绘制QQ图
fig,ax = plt.subplots(figsize=(12, 8), dpi=100)
stats.probplot(data.Y, plot=ax)
plt.title('Quantile-Quantile plot')

# KS检验
mu = np.mean(data.Y)
sigma = np.std(data.Y)
stats.kstest(data.Y, 'norm', args=(mu, sigma))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/3965beb7d8af40bc966c81cb398c38b0.png)![请添加图片描述](https://img-blog.csdnimg.cn/dbb497d04ff2492c984d5c451d26e8da.png)
- 从QQ图结果可以看到被解释变量偏离点较多，尽管通过了KS检验（有可能是因为样本过少）。因此只能考虑第二种做法：降低测量误差，即采用对数化方法。这是需要考虑三种情况：
  - 对被解释变量取对数，解释变量保持不变
  - 对解释变量取对数，被解释变量保持不变
  - 对解释变量和被解释变量同时取对数（展示这种方案）
$$
y = \beta_0 + \beta_1 \cdot x + u 
\rightarrow
\log(y) = \beta_0 + \beta_1 \cdot \log(x) + u 
$$
```python
# 对数后OLS
res2_ols = sm.formula.ols('np.log(Y)~np.log(X)', data=data).fit()
res2_ols.summary()

# 残差散点图
plt.figure(figsize=(12, 6))
plt.scatter(x=data.X, y=res2_ols.resid)
plt.tick_params(labelsize=15)
plt.xlabel('X', size=15)
plt.ylabel('Y', size=15)
plt.savefig('res_log01.png', dpi=300)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/bc3c3249c52a4761864a6b91083fc8a6.png)
![请添加图片描述](https://img-blog.csdnimg.cn/e8eaf3f34b964e6db18cc8b18beb3338.png)
- 经过两边取对数化后发现，残差图中看不出明显的异方差，这里可以进一步使用bp异方差检验。显然，结果拒绝了原假设（默认$\alpha=0.05$），即存在异方差。
```python
def bp_test(res, X):
    result_bp_test = sm.stats.diagnostic.het_breuschpagan(res, X)
    bp_lm_statistic = result_bp_test[0]
    bp_lm_pval = result_bp_test[1]
    bp_F_statistic= result_bp_test[2]
    bp_F_pval = result_bp_test[3]
    bp_test_output=pd.Series(result_bp_test[0:4],index=['bp_lm_statistic','bp_lm_pval','bp_F_statistic','bp_F_pval'])    
    return bp_test_output

res2_ols_md = sm.formula.ols('np.log(Y)~np.log(X)', data=data)
res2_ols = res2_ols_md.fit()
bp_test(res2_ols.resid, res2_ols_md.exog)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/0a9201a4d9e8410390ee55fc51de6b62.png)


- 下一步则需要考虑其他对数形式的模型，反复上述过程。最终选出以下的模型，通过了系数显著性检验、模型显著性检验和BP检验，这里加入$x^2$是因为残差散点图中可以发现有点像抛物线的样子，结合异方差的定义可以考虑引入二次项做回归。这里也可以认为是加入了被遗漏的变量。
$$
y = \beta_0 + \beta_1 \cdot x + u 
\rightarrow
\log(y) = \beta_0 + \beta_1 \cdot x + \beta_2 \cdot x^2 + u 
$$
```python
res2_ols = sm.formula.ols('np.log(Y)~ X + I(X**2)', data=data).fit()
res2_ols.summary()

plt.figure(figsize=(12, 6))
plt.scatter(x=data.X, y=res2_ols.resid)
plt.tick_params(labelsize=15)
plt.xlabel('X', size=15)
plt.ylabel('Y', size=15)
plt.savefig('res_log02.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/73ebb29f509f4b1094fcae3893c9a347.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/e95a588351c842b0b538e143165bd72e.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/81bf15a7f5e94ce4a3131acf5077f7e6.png)
- 从上述的结果可以发现，存在异方差时来说依赖于WLS或是FGLS做模型是不够的，仍需要考虑解释变量（特征）是否能够解释变异，通过增加变量或是降低测量误差的方式都时重要的方法。

# 四、分类
## （一）线性回归与二分类问题
分类问题可以将被解释变量看成是一个选择函数，如二分类问题中假定被解释变量只有0和1两种情况。
$$
y= \begin{cases}1, & 发生 \\ 0, & \text { 不发生 }\end{cases}
$$
一般来说，我们会更关注给定解释变量时，被解释变量取某个值的概率，即有：
$$
P(y=i|x) ，i=1或0
$$
对于二分类变量，假定$P(y=1|x) = g(h(x;\beta))$。$g(x)$是一个$(0,1)$区间内关于解释变量$x$和参数$\beta$的函数，这时候我们可以进一步假定$h(x;\beta)=\beta_0 + \beta_1x_1 +... +\beta_nx_n$。因此可以考虑使用线性回归模型拟合参数得到事件发生的概率，而$g(x)$称为**连接函数**。
- 这里需要注意一点的是，$h(x)$和$u$有着相似的分布，$u$可以不再服从正态性假设。（这一点可以结合广义线性模型的原理进行理解）
## （二）Logistic回归与Probit回归
### 1. Logistics回归原理
在学习Logistics回归模型前需要先了解一个概念：Odds（一般被称为优势比或几率比），有如下定义：
$$
Odds= \frac{P(y=1|x)}{P(y=0|x)} = \frac{P(y=1|x)}{1 - P(y=1|x)} =  \frac{p}{1 - p}
$$
记$p=P(y=1|x)$，有$p \in (0,1)$，可以画出$Odds$的图像：
```python
import numpy as np
import matplotlib.pyplot as plt

# 定义odds函数
def odds(x):
    return x / (1 - x)

# 模拟0到1之间的数，注意不可取1
x = 1 - np.linspace(0, 1, num=100, endpoint=False)
y = odds(x)

# 1*2图设置
fig, axes = plt.subplots(1, 2, figsize=(16,8))

# 正常图像
axes[0].plot(x, y)
axes[0].tick_params(labelsize=15)
axes[0].set_xlabel('p', size=15)
axes[0].set_ylabel('odds', size=15)

# y轴对数压缩
axes[1].plot(x, y)
axes[1].set_yscale('log')  # y轴对数化
axes[1].tick_params(labelsize=15)
axes[1].set_xlabel('p', size=15)
axes[1].set_ylabel('odds', size=15)
```
![请添加图片描述](https://img-blog.csdnimg.cn/a6a22613e11f46ac8e6cffcacf35e561.png)
显然，当$odds$对数化后，曲线变得十分光滑，即有：
$$
\log(odds) = \log\frac{p}{1 - p}
$$
```python
# 对数化
x = 1 - np.linspace(0, 1, num=100, endpoint=False)
y = np.log(odds(x))

plt.figure(figsize=(12,8))
plt.plot(x, y)
plt.tick_params(labelsize=15)
plt.xlabel('p', size=15)
plt.ylabel('lodds', size=15)
plt.title(r'$\log(odds) = \log(\frac{p}{1-p})$', size=15)
plt.savefig('lodds.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/f390d47198934192bcdb168d8e8acdb8.png)
这里就引出了**Logit函数**的概念，这里为什么叫Logit，有个有趣是说法就是Log-it，it为odds（这部分参考文章：[Logit究竟是个啥？](https://zhuanlan.zhihu.com/p/27188729)）。这时候，有如下两个式子：
$$
P(y=1|x) = p \\
Logit(odds) = log(\frac{p}{1-p})
$$
已知$Logit(odds)$在$p\in(0,1)$上的值域为$(-\infty, +\infty)$，进一步计算出反函数，该反函数称为**sigmoid函数**。其中，反函数与原函数关于$y=x$对称。
$$
p = \frac{1}{1+\exp(-h)} = g(h) \\
h = Logit(odds)
$$
```python
# 定义odds函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x1 = 1 - np.linspace(0, 1, num=100, endpoint=False)
x2 = np.linspace(-4, 4, num=100, endpoint=False)
y1 = np.log(odds(x1))
y2 = sigmoid(x2)

# 1*2画布
fig, axes = plt.subplots(1, 2, figsize=(20,8))

# sigmoid函数
axes[0].plot(x2, y2, 'r')
axes[0].tick_params(labelsize=15)
axes[0].set_xlabel('p', size=15)
axes[0].set_ylabel('h', size=15)
axes[0].tick_params(labelsize=15)
axes[0].set_xlabel('x', size=15)
axes[0].set_ylabel('y', size=15)
axes[0].set_title('sigmoid函数', size=20)

# 两个函数的关系
axes[1].plot(x1, y1, 'b', label=r'$h = \log(\frac{p}{1-p})$')
axes[1].plot(x2, y2, 'r', label=r'$p = \frac{1}{1 + \exp{-h}}$')
axes[1].tick_params(labelsize=15)
axes[1].set_xlabel('x', size=15)
axes[1].set_ylabel('y', size=15)
axes[1].set_xlim(xmin=-4, xmax=4)
axes[1].set_ylim(ymin=-4, ymax=4)
axes[1].legend(prop={'size':15})
axes[1].set_title('两个函数的关系', size=20)
plt.savefig('sigmoid.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/03e0775d62574efb942dacde5629893d.png)


根据线性回归在二分类问题上应用的概念，可以设$h(x;\beta) = \beta^Tx$，那么就有如下的式子，推导出Logistic回归模型。
$$
\begin{aligned}
&P(y=1|x) = p \\
\Rightarrow  &P(y=1|x)  = \frac{1}{1+\exp(-h)} \\
\Rightarrow  &P(y=1|x)  =\frac{1}{1+\exp(- \beta^Tx)}
\end{aligned}
$$
Logistic回归模型如下所示：
$$
P(y=1|x)  =\frac{1}{1 + \exp(- \beta^Tx)}
$$

### 2. Probit回归原理
**Sigmoid函数**其实是将$(-\infty, +\infty)$映射到$(0, 1)$且曲线十分光滑（意味着容易求导），同样的我们可以考虑一种在数理统计中十分常见的函数：**累积分布函数**(Cumulative Distribution Function，CDF)。需要注意区分三个函数的区别：
$$
\begin{aligned}
概率密度函数PDF&：f(v;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\{-\frac{(x-\mu)^2}{2\sigma^2}\}\\
累积分布函数CDF&：F(x) = P(X\le x) = p =  \int_{-\infty}^{x} f(v;\mu,\sigma^2)dv \\
逆累积分布函数PPF&：x = F^{-1}(p)
\end{aligned}
$$
```python
from scipy.stats import norm

x1 = np.linspace(-4, 4, num=100, endpoint=False)
x2 = np.linspace(-4, 4, num=100, endpoint=False)
x3 = np.linspace(0, 1, num=100, endpoint=False)
y1 = norm.pdf(x1)
y2 = norm.cdf(x2)
y3 = norm.ppf(x3)

# 1*2画布
fig, axes = plt.subplots(1, 3, figsize=(30,8))

# pdf
axes[0].plot(x1, y1, 'r')
axes[0].tick_params(labelsize=15)
axes[0].tick_params(labelsize=15)
axes[0].set_xlabel('v', size=15)
axes[0].set_ylabel('f', size=15)
axes[0].set_title('PDF函数', size=20)

# cdf
axes[1].plot(x2, y2, 'r')
axes[1].tick_params(labelsize=15)
axes[1].tick_params(labelsize=15)
axes[1].set_xlabel('x', size=15)
axes[1].set_ylabel('p', size=15)
axes[1].set_title('CDF函数', size=20)

# ppf
axes[2].plot(x3, y3, 'r')
axes[2].tick_params(labelsize=15)
axes[2].tick_params(labelsize=15)
axes[2].set_xlabel('p', size=15)
axes[2].set_ylabel('x', size=15)
axes[2].set_title('PPF函数', size=20)
fig.suptitle('概率函数的三种表示', fontsize=20)
plt.savefig('概率函数的三种表示.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/d52ff972612b4bfa944468260cac70c8.png)
累积分布函数与Sigmoid函数有着相近的值域和定义域，适合作为连接函数。Probit回归的连接函数则是标准正态分布的CDF。Sigmoid函数与标准正态分布的CDF差异如下图所示：
```python
fig, axes = plt.subplots(1, 1, figsize=(8,8))

x = np.linspace(-4, 4, num=100, endpoint=False)
y1 = norm.cdf(x)
y2 = sigmoid(x2)

axes.plot(x, y1, 'b', label='Sigmoid函数')
axes.plot(x, y2, 'r', label='CDF函数')
axes.tick_params(labelsize=15)
axes.set_xlabel('h', size=15)
axes.set_ylabel('p', size=15)
axes.set_xlim(xmin=-4, xmax=4)
axes.set_ylim(ymin=0, ymax=1)
axes.legend(prop={'size':15})
plt.savefig('cdf和sigmoid.png', dpi=300)
```
![请添加图片描述](https://img-blog.csdnimg.cn/1c47eeccfd8a4b1daa51f9d9ce611996.png)
Probit回归模型中，令$p = \Phi(h)$，其中$\Phi(h)$为标准正态分布CDF，模型如下所示：
$$
\begin{aligned}
&P(y=1|x) = p \\
\Rightarrow  &P(y=1|x)  =\Phi(h) \\
\Rightarrow  &P(y=1|x)  =\Phi(\beta^Tx)
\end{aligned}
$$
### 3. 两种线性模型的异同
- 相同点
两种模型都是用于解决分类问题，从事件发生概率的角度出发，通过连接函数建立起线性模型，均可以使用统计检验方法做推断。
- 不同点
最直观的差异就是两种模型使用的连接函数不一样，除此之外两者的差异还在于残差的分布假设。
回到最开始的问题，$y$的分段函数可以用换成另一种写法：
	- 存在一个连续变量$h$，使得$h>0$时，有$y=i$，即：
$$
y|x = \begin{cases}1, & h >0 \\ 0, & h \le 0 \end{cases}
$$
那么事件$y=1$发生的概率：
$$
\begin{aligned}
P(y=1|x) 
&= P(h > 0|x) \\
&= P(\beta^Tx> 0|x) \\
&= P(u > -\beta^Tx|x) \\
&=P(u \le \beta^Tx|x)
\end{aligned}
$$
从这一步可以看出，在给定$x$的情况下，残差$u$的分布与$y$的分布一致。根据**线性回归均值建模**原理可以推导出$p$为随机变量$y$的均值，样本数据用于估计$p$。
$$
y = y_i |x \sim B(1,p)\\
\begin{aligned}
&P(y=y_i|x) = p^{y_i} \cdot (1 - p)^{1 - y_i} \\
\Rightarrow &E(y=y_i|x) = p \\
\Rightarrow &P(y=1|x) = E(y=y_i|x) + u
\end{aligned}
$$
	- 当连接函数$h=sigmoid(\beta^Tx+u)$时，残差项$u$在$sigmoid$函数中服从的是**Logistic分布**（注：当残差项移到$sigmoid$函数外时服从的是**0-1分布**）。
$$
P(y=1|x)  =  \frac{1}{1 + \exp(- \beta^Tx)} +u，u|x\sim B(1,sigmoid(\beta^Tx))
$$
	- 当连接函数为$h=\Phi(\beta^Tx+u)$时，残差项$u$服从的是**标准正态分布**。
（注：当残差项移到CDF函数外时服从的是**0-1分布**）
$$
P(y=1|x)  =F(\beta^Tx) + u，u|x \sim B(1, \Phi( \beta^Tx))
$$
## （三）二分类与多分类(补充ing)
### 1. 多分类Sigmoid推导
二分类问题中，Logistic回归的连接函数为Logit函数，即：$h(x)=log(\frac{p(x)}{1-p(x)})=\beta^Tx$。其中$\frac{p(x)}{1-p(x)}=Odds$为优势比，详细地说是$y=1$对于$y=0$的优势比。
多分类问题中，解释变量$y$有多个取值，根据二分类方法的思路可以对多分类问题进行拓展。选择一类作为**基类**，计算其他类对于基类的优势比，记$y=0,1,...,M$。
$$
h(x)_{m} = \log(\frac{P(y=m|x)}{P(y=0|x)})  = \beta^Tx，m=1,...,M \\
\begin{aligned}
e^{h_m} &= \frac{P(y=m|x)}{P(y=0|x)} \\
\Rightarrow P(y=m|x) &=  e^{h_m}  \cdot P(y=0|x)\\
\Rightarrow P(y=m|x) &=  e^{h_m}  \cdot (1 - \sum\limits_{i=1}^{M}P(y=i|x)) \\
因此有：\sum\limits_{j=1}^{M}P(y=j|x) &= 
 \sum\limits_{j=1}^{M}e^{h_j}  \cdot (1 - \sum\limits_{i=1}^{M}P(y=i|x)) \\
 \Rightarrow \sum\limits_{j=1}^{M}P(y=j|x) &= 
\frac{\sum\limits_{j=1}^{M}e^{h_j}}{1+ \sum\limits_{i=1}^{M}e^{h_i}}  \\
 \Rightarrow P(y=m|x) &=  \frac{e^{h_m}}{1+ \sum\limits_{i=1}^{M}e^{h_i}}，m=1, ...,M
\end{aligned}
$$
上述为多分类Sigmoid函数，除此之外还有一个比较相近且常用的函数：Softmax函数。
Sigmoid函数是基于二项分布推导，而Softmax是基于多项分布推导，在二分类问题上两者是一致的，而在多分类的情况下两者的损失函数会存在差异。
$$
Sigmoid函数：P(y=m|x) =  \frac{e^{h_m}}{1+ \sum\limits_{i=1}^{M}e^{h_i}}，m=1, ...,M \\
Softmax函数：P(y=m|x) =  \frac{e^{h_m}}{ \sum\limits_{i=0}^{M}e^{h_i}}，m=0, ...,M 
$$


通过上述推导，可以得到每个类别的条件概率，在实际预测是需要选出$M-1$个概率中最大概率的类别，再将类别与阈值$\alpha$比较。
$$
\begin{aligned}
k &= \underset{m \in \{1,2,...,M\}}{\argmax} P(y=m|x) \\
y &= \begin{cases}k, & P(y=k|x) \ge \alpha \\ 0, & P(y=k|x) < \alpha \end{cases}
\end{aligned}
$$
### 2. OVO与OVR
并非所有模型像Logistic模型一样推导出多分类的Logit函数，通常来说二分类转向多分类时主要采用两者策略：**一对一**（**OVO**，One VS One）、**一对其余**（**OVR**，One VS Rest）。假设解释变量$y$共有$K$个类，不同的策略方法如下所示：
- OVO
先按解释变量将数据集（训练集）拆成K份，然后两两组合成新的数据集并构建出一个分类器，因此总共需要构建出$C_{2}^{K}$个分类器，最后根据这些分类器的结果按投票法选出数量最多的类别。
- OVR
先按解释变量将数据集（训练集）拆成K份，但与OVO不同的是，每次只需要选出一个类别（记为1），其余类别（记为0）合并成一个数据集并构建一个分类器，这就意味着每个类别只需要构建一个分类器，因此总共需要构建$K$个分类器，最后根据这些分类器选出正样本概率最大的类别。
## （四）作业答案参考
### 题目1
![在这里插入图片描述](https://img-blog.csdnimg.cn/18c8793c0c8e49e18cfa738b4eea8192.png)
- 导入数据，剔除缺失数据
```python
# 导入数据
loan=pd.read_stata('./data/loanapp.dta')

# 选取要用的变量组成新的数据集
loan=loan[["approve","white","hrat","obrat","loanprc","unem","male","married","dep","sch","cosign","chist","pubrec","mortlat1","mortlat2","vr"]]

# 去除含缺失值样本
loan = loan.dropna() 
```
- 考虑普通OLS回归模型，$\beta_1$代表的是`white=1`时的效应，即歧视效应。当$\beta_1$显著且很大时，代表着歧视效应明显。从拟合的结果来看，$\beta_1$是显著，说明存在一定的歧视效应，其正向符合表面白人在贷款审批时具有优势。
$$
approve = \beta_0 + \beta_1 \cdot white + u
$$
```python
md_ols = sm.formula.ols('approve~white', data=loan).fit()
md_ols.summary()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/0291af4912564b2ebd7a23c47c3a6054.png)
- 引入其他控制变量（如，是否为男性`male`、是否已婚`married`），继续拟合OLS模型。从显著性报告结果来看，歧视效应仍旧显著。
$$
approve = \beta_0 + \beta_1 \cdot white + \beta_2 \cdot male + \beta_3 \cdot married + u
$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/753d6cd6be9049bfa425c312a831af4a.png)
- 进一步，引入`obrat`债务占比，并分析是否该变量与`white`的交互效应。注意，需要排除掉共线性的影响，这里考虑对交互项进行**去中心化**处理。从结果上看，主效应和交互效应均显著。交互效应正向显著，意味着在高债务占比的情况下，歧视效应越加明显。
$$
approve = \beta_0 + \beta_1 \cdot white + \beta_2 \cdot obrat +
 \beta_3 \cdot (white -\overline{white} ) \cdot(obrat -\overline{obrat}) + u
$$
```python
# 构建去中心化变量
loan['white_1'] = loan['white']-np.mean(loan['white'])
loan['obrat_1'] = loan['obrat']-np.mean(loan['obrat'])
md3_ols = sm.formula.ols('approve~white+obrat+I(white_1 * obrat_1)', data=loan).fit()
print(md3_ols.summary())
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/26d64c89dd39434baf8d386f7936fad2.png)
- 上述使用的OLS对系数估计和显著性检验会存在较大问题，主要是因为这里的解释变量是二元变量(0/1)，违背了普通OLS假设的正态性假设，因此可以考虑使用Logitstic模型。从结果上看，交互效应变得不显著，但主效应依旧显著且系数为正，说明在信贷审批中存在显著歧视效应。
```python
md4_logit = sm.formula.logit('approve~white+obrat+I(white_1 * obrat_1)', data=loan).fit()
print(md4_logit.summary())
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/b9e0c081ea484590948218b7aa69e824.png)
### 题目2
![在这里插入图片描述](https://img-blog.csdnimg.cn/bc85c52e94a4479bb1c8cc4a56510816.png)
- 导入依赖库和数据
```python
from sklearn.datasets import load_iris

# 下载/读取鸢尾花数据集
iris_dataset=load_iris()

# 提取数据集中的自变量集与标签集
iris_data=iris_dataset['data'] # 自变量
iris_target=iris_dataset['target'] # 标签集
```
- 使用`trian_test_split`划分训练集和测试集
```python
from sklearn.model_selection import train_test_split

# 按3：1划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(iris_data, iris_target, test_size=0.25)
```
- 使用`LogisticRegression`模型，分别计算训练集和测试集准确率。从上述结果来看，训练集的准确率要高于测试集的准确率。原因在于模型训练中使用的是训练集，模型可以反映训练集的信息，但测试集有可能包含训练集未能反映的信息，模型不一定可以捕捉到，从而导致测试集的准确率要低于训练集的准确率。这样也说明了测试集的结果一定程度上反映了模型泛化能力。
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 模型训练
md = LogisticRegression(n_jobs=-1)
md.fit(train_x, train_y)

# 模型预测
train_pred = md.predict(train_x)
test_pred = md.predict(test_x)

# 计算正确率
tmp = [['Train', accuracy_score(train_y, train_pred)],
       ['Test', accuracy_score(test_y, test_pred)]]
res = pd.DataFrame(data=tmp, columns=['type', 'Accuracy'])
print(res)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/fd34d08e0a614f0d8c3b73cbbc2cf2dc.png)
- 计算测试集的混淆矩阵、精确率、召回率和F1score
```python
from sklearn.metrics import confusion_matrix

# 计算测试集的混淆矩阵
confusion_matrix(test_y, test_pred)

# 返回综合报告
print(classification_report(test_y, test_pred))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2118b99315994558bb44a976676b0119.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/57c5712ab38d449183ef016a88b525fd.png)
